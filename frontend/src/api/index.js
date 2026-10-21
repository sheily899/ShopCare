/**
 * ShopCare 前端 API 边界。
 * 只负责把 Vue 视图请求映射到 FastAPI；不在浏览器端复制意图、RAG 或 Agent 逻辑。
 */

// 后端开发默认运行在 8010；生产环境可通过 VITE_API_BASE_URL 覆盖。
const API_BASE = import.meta.env?.VITE_API_BASE_URL || 'http://127.0.0.1:8010'

function requestError(payload, status) {
  const detail = payload?.detail
  const message = typeof detail === 'string'
    ? detail
    : (detail?.message || detail?.msg || `请求失败（${status}）`)
  const error = new Error(message)
  if (detail && typeof detail === 'object') error.details = detail
  return error
}

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      ...(options.body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
      ...(options.headers || {}),
    },
  })
  const payload = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw requestError(payload, response.status)
  }
  return payload
}

export function login(username, password) {
  return request('/auth/login', { method: 'POST', body: JSON.stringify({ username, password }) })
}

export function registerConsumer(username, password, displayName) {
  return request('/auth/register', { method: 'POST', body: JSON.stringify({ username, password, display_name: displayName }) })
}

export function sendChat({ message, userId = 'anonymous', convId, token }) {
  return request('/chat', {
    method: 'POST',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body: JSON.stringify({ message, user_id: userId, conv_id: convId }),
  })
}

export function getConsumerMessages(convId, token, afterSeq = null) {
  const query = afterSeq == null ? '' : `?after_seq=${encodeURIComponent(String(Math.max(0, Number(afterSeq) || 0)))}`
  return request(`/chat/${encodeURIComponent(convId)}/messages${query}`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  })
}

export function answerSlot(convId, payload, token) {
  return request(`/chat/${encodeURIComponent(convId)}/slots`, {
    method: 'POST',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body: JSON.stringify(payload),
  })
}

export async function streamChat({ message, userId = 'anonymous', convId, token, onEvent }) {
  const response = await fetch(`${API_BASE}/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}) },
    body: JSON.stringify({ message, user_id: userId, conv_id: convId }),
  })
  if (!response.ok || !response.body) throw new Error(`请求失败（${response.status}）`)
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { value, done } = await reader.read()
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done })
    const blocks = buffer.split('\n\n')
    buffer = blocks.pop() || ''
    for (const block of blocks) {
      const event = block.match(/^event:\s*(.+)$/m)?.[1] || 'message'
      const data = block.match(/^data:\s*(.+)$/m)?.[1] || '{}'
      const payload = JSON.parse(data)
      onEvent?.(event, payload)
    }
    if (done) break
  }
}

export function getHealth() {
  return request('/health')
}

export function searchKnowledge(query, topK = 5, token, domain) {
  return request('/search', {
    method: 'POST',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body: JSON.stringify({ query, top_k: topK, domain }),
  })
}

export function uploadKnowledge(file, token, domain) {
  const form = new FormData()
  form.append('file', file)
  form.append('domain', domain)
  return request('/knowledge/upload', {
    method: 'POST',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body: form,
  })
}

export function getTraces(token) {
  return request('/trace/tools', { headers: token ? { Authorization: `Bearer ${token}` } : {} })
}

export function getServiceSessions(token) {
  return request('/service/sessions', { headers: token ? { Authorization: `Bearer ${token}` } : {} })
}

export function getServiceDashboard(token) {
  return request('/service/dashboard', { headers: token ? { Authorization: `Bearer ${token}` } : {} })
}

export function getServiceSession(convId, token, afterSeq = null) {
  const query = afterSeq == null ? '' : `?after_seq=${encodeURIComponent(String(Math.max(0, Number(afterSeq) || 0)))}`
  return request(`/service/sessions/${encodeURIComponent(convId)}${query}`, { headers: token ? { Authorization: `Bearer ${token}` } : {} })
}

export function refreshServiceHandoff(convId, token) {
  return request(`/service/sessions/${encodeURIComponent(convId)}/refresh`, {
    method: 'POST',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  })
}

export async function streamServiceHandoffEvents({ convId, token, lastEventId = '', onEvent, signal }) {
  const headers = { Accept: 'text/event-stream' }
  if (token) headers.Authorization = `Bearer ${token}`
  if (lastEventId) headers['Last-Event-ID'] = String(lastEventId)
  const response = await fetch(`${API_BASE}/service/sessions/${encodeURIComponent(convId)}/handoff/events`, {
    headers,
    signal,
  })
  if (!response.ok || !response.body) throw new Error(`交接事件连接失败（${response.status}）`)

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let cursor = String(lastEventId || '')
  const dispatch = (block) => {
    let eventType = 'message'
    const data = []
    for (const line of block.split(/\r?\n/)) {
      if (!line || line.startsWith(':')) continue
      const separator = line.indexOf(':')
      const field = separator < 0 ? line : line.slice(0, separator)
      const value = separator < 0 ? '' : line.slice(separator + 1).replace(/^ /, '')
      if (field === 'id' && !value.includes('\0')) cursor = value
      else if (field === 'event') eventType = value || 'message'
      else if (field === 'data') data.push(value)
    }
    if (!data.length) return
    const raw = data.join('\n')
    let payload
    try { payload = JSON.parse(raw) } catch { payload = raw }
    onEvent?.(eventType, payload, cursor)
  }

  while (true) {
    const { value, done } = await reader.read()
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done })
    const blocks = buffer.split(/\r?\n\r?\n/)
    buffer = blocks.pop() || ''
    for (const block of blocks) dispatch(block)
    if (done) break
  }
  if (buffer.trim()) dispatch(buffer)
  return { lastEventId: cursor }
}

export function sendServiceMessage(convId, content, agentId = 'agent-demo', token) {
  return request(`/service/sessions/${encodeURIComponent(convId)}/messages`, {
    method: 'POST',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body: JSON.stringify({ agent_id: agentId, content }),
  })
}

export function updateServiceTicket(convId, fields, token, expectedVersion) {
  return request(`/service/sessions/${encodeURIComponent(convId)}/ticket`, {
    method: 'PATCH',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body: JSON.stringify({ fields, expected_version: expectedVersion ?? null }),
  })
}

export function confirmServiceTicket(convId, token) {
  return request(`/service/sessions/${encodeURIComponent(convId)}/ticket/confirm`, {
    method: 'POST',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  })
}
