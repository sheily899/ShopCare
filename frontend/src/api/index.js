/**
 * ShopCare 前端 API 边界。
 * 只负责把 Vue 视图请求映射到 FastAPI；不在浏览器端复制意图、RAG 或 Agent 逻辑。
 */

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

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
    throw new Error(payload.detail || `请求失败（${response.status}）`)
  }
  return payload
}

export function sendChat({ message, userId = 'anonymous', convId }) {
  return request('/chat', {
    method: 'POST',
    body: JSON.stringify({ message, user_id: userId, conv_id: convId }),
  })
}

export function getHealth() {
  return request('/health')
}

export function searchKnowledge(query, topK = 5) {
  return request('/search', {
    method: 'POST',
    body: JSON.stringify({ query, top_k: topK }),
  })
}

export function uploadKnowledge(file) {
  const form = new FormData()
  form.append('file', file)
  return request('/knowledge/upload', { method: 'POST', body: form })
}

export function reloadSkills() {
  return request('/skills/reload', { method: 'POST' })
}

export function getSkills() {
  return request('/skills')
}

export function getTraces() {
  return request('/trace/tools')
}
