<script setup>
import { computed, nextTick, ref, watch } from 'vue'
import { sendChat } from '../api/index.js'

const emit = defineEmits(['sessions-changed'])
const STORAGE_KEY = 'shopcare_sessions'
const USER_ID = 'anonymous'
const quickPrompts = ['退款申请', '技术排查', '发票咨询']

function createId() {
  return globalThis.crypto?.randomUUID?.() || `conv-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function readSessions() {
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')
    return Array.isArray(stored) ? stored : []
  } catch {
    return []
  }
}

function createSession() {
  return { id: createId(), createdAt: new Date().toISOString(), messages: [] }
}

function escapeHtml(value) {
  return String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#39;')
}

function renderInlineMarkdown(value) {
  let html = value
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>')
  html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+|mailto:[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>')
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
  html = html.replace(/__([^_]+)__/g, '<strong>$1</strong>')
  html = html.replace(/~~([^~]+)~~/g, '<del>$1</del>')
  return html.replace(/\*([^*\n]+)\*/g, '<em>$1</em>')
}

function splitTableRow(line) {
  const trimmed = line.trim().replace(/^\|/, '').replace(/\|$/, '')
  return trimmed.split('|').map((cell) => cell.trim())
}

function isTableSeparator(line) {
  const cells = splitTableRow(line)
  return cells.length > 0 && cells.every((cell) => /^:?-{3,}:?$/.test(cell))
}

function renderTable(lines) {
  const headers = splitTableRow(lines[0]).map((cell) => `<th>${renderInlineMarkdown(escapeHtml(cell))}</th>`).join('')
  const body = lines.slice(2).map((line) => {
    const cells = splitTableRow(line)
    return `<tr>${cells.map((cell) => `<td>${renderInlineMarkdown(escapeHtml(cell))}</td>`).join('')}</tr>`
  }).join('')
  return `<table><thead><tr>${headers}</tr></thead><tbody>${body}</tbody></table>`
}

function renderMarkdown(value) {
  const lines = String(value ?? '').replace(/\r\n?/g, '\n').split('\n')
  const output = []
  let index = 0
  while (index < lines.length) {
    const line = lines[index]
    if (!line.trim()) { index += 1; continue }

    if (/^\s*```/.test(line)) {
      const codeLines = []
      index += 1
      while (index < lines.length && !/^\s*```/.test(lines[index])) {
        codeLines.push(lines[index])
        index += 1
      }
      if (index < lines.length) index += 1
      output.push(`<pre><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`)
      continue
    }

    if (index + 1 < lines.length && line.includes('|') && isTableSeparator(lines[index + 1])) {
      const tableLines = [line, lines[index + 1]]
      index += 2
      while (index < lines.length && lines[index].trim() && lines[index].includes('|')) {
        tableLines.push(lines[index])
        index += 1
      }
      output.push(renderTable(tableLines))
      continue
    }

    const heading = line.match(/^\s*(#{1,3})\s+(.+)$/)
    if (heading) {
      output.push(`<h${heading[1].length}>${renderInlineMarkdown(escapeHtml(heading[2]))}</h${heading[1].length}>`)
      index += 1
      continue
    }

    const unordered = line.match(/^\s*[-*+]\s+(.+)$/)
    if (unordered) {
      const items = []
      while (index < lines.length) {
        const item = lines[index].match(/^\s*[-*+]\s+(.+)$/)
        if (!item) break
        items.push(`<li>${renderInlineMarkdown(escapeHtml(item[1]))}</li>`)
        index += 1
      }
      output.push(`<ul>${items.join('')}</ul>`)
      continue
    }

    const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/)
    if (ordered) {
      const items = []
      while (index < lines.length) {
        const item = lines[index].match(/^\s*\d+[.)]\s+(.+)$/)
        if (!item) break
        items.push(`<li>${renderInlineMarkdown(escapeHtml(item[1]))}</li>`)
        index += 1
      }
      output.push(`<ol>${items.join('')}</ol>`)
      continue
    }

    const paragraph = []
    while (index < lines.length && lines[index].trim()) {
      const current = lines[index]
      if (paragraph.length && (/^\s*```/.test(current) || /^\s*(#{1,3})\s+/.test(current) || /^\s*[-*+]\s+/.test(current) || /^\s*\d+[.)]\s+/.test(current))) break
      paragraph.push(escapeHtml(current))
      index += 1
    }
    output.push(`<p>${renderInlineMarkdown(paragraph.join('<br>'))}</p>`)
  }
  return output.join('')
}

function formatTime(value) {
  return new Intl.DateTimeFormat('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }).format(new Date(value))
}

function formatLatency(value) {
  const milliseconds = Number(value)
  if (value === null || value === undefined || value === '' || Number.isNaN(milliseconds) || milliseconds < 0) return '—'
  const seconds = milliseconds / 1000
  if (seconds < 60) return `${seconds.toFixed(2)} 秒`
  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = Math.round(seconds % 60)
  if (remainingSeconds === 0) return `${minutes} 分钟`
  return `${minutes} 分 ${String(remainingSeconds).padStart(2, '0')} 秒`
}

const sessions = ref(readSessions())
if (!sessions.value.length) sessions.value = [createSession()]
const activeId = ref(sessions.value[0].id)
const input = ref('')
const sending = ref(false)
const errorMessage = ref('')
const stream = ref(null)

const activeSession = computed(() => sessions.value.find((session) => session.id === activeId.value) || sessions.value[0])
const messages = computed(() => activeSession.value?.messages || [])

function sessionSummary(session) {
  const firstUserMessage = session.messages.find((item) => item.role === 'user')?.content || '新会话'
  return { id: session.id, title: firstUserMessage, createdAt: formatTime(session.createdAt) }
}

function emitSessionSummaries() {
  emit('sessions-changed', sessions.value.map(sessionSummary))
}

function persist() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions.value))
  emitSessionSummaries()
}

function updateSession(session) {
  sessions.value = sessions.value.map((item) => (item.id === session.id ? session : item))
  persist()
}

function startNewConversation() {
  const session = createSession()
  sessions.value = [session, ...sessions.value]
  activeId.value = session.id
  input.value = ''
  errorMessage.value = ''
  persist()
}

function selectSession(id) {
  if (sessions.value.some((session) => session.id === id)) activeId.value = id
  errorMessage.value = ''
}

async function submit(message = input.value) {
  const text = String(message || '').trim()
  if (!text || sending.value || !activeSession.value) return
  const session = { ...activeSession.value, messages: [...activeSession.value.messages, { role: 'user', content: text, createdAt: new Date().toISOString() }] }
  updateSession(session)
  input.value = ''
  errorMessage.value = ''
  sending.value = true
  try {
    // 以浏览器请求生命周期计时，覆盖网络传输、后端处理与响应渲染前等待，
    // 这样展示给用户的耗时才对应真实体感，而不是只反映后端某个内部阶段。
    const clientStartedAt = performance.now()
    const result = await sendChat({ message: text, userId: USER_ID, convId: session.id })
    const clientLatencyMs = Math.max(0, performance.now() - clientStartedAt)
    const assistantMessage = {
      role: 'assistant',
      content: result.response || '当前没有可展示的回复。',
      createdAt: new Date().toISOString(),
      // UI 使用 client_latency_ms；server_latency_ms 保留后端 API 计时，便于定位网络与服务端耗时差异。
      metadata: { ...result, latency_ms: clientLatencyMs, client_latency_ms: clientLatencyMs, server_latency_ms: result.latency_ms },
    }
    updateSession({ ...session, id: result.conv_id || session.id, messages: [...session.messages, assistantMessage] })
    if (result.conv_id && result.conv_id !== activeId.value) activeId.value = result.conv_id
  } catch (error) {
    errorMessage.value = error.message || '请求未完成，请稍后重试。'
  } finally {
    sending.value = false
  }
}

function onKeydown(event) {
  if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
    event.preventDefault()
    submit()
  }
}

watch(sessions, emitSessionSummaries, { deep: true, immediate: true })
watch(messages, async () => {
  await nextTick()
  if (stream.value) stream.value.scrollTop = stream.value.scrollHeight
}, { deep: true })

defineExpose({ startNewConversation, selectSession })
</script>

<template>
  <section class="chat-workspace" aria-label="会话工作区">
    <section class="chat-vessel">
      <div ref="stream" class="message-stream" aria-live="polite">
        <div v-if="!messages.length" class="welcome-block">
          <div class="welcome-badge" aria-hidden="true"><span></span></div>
          <h1>从一个客户问题开始</h1>
          <p class="welcome-subtitle">下面的快捷问题只是起点，你也可以直接输入自己的测试用例。</p>
          <div class="quick-prompts" aria-label="常用问题">
            <button v-for="prompt in quickPrompts" :key="prompt" class="chip-button" type="button" @click="submit(prompt)">{{ prompt }}</button>
          </div>
        </div>

        <article v-for="(message, index) in messages" :key="`${message.createdAt}-${index}`" class="message-row" :class="message.role">
          <div class="message-bubble">
            <div v-if="message.role === 'assistant'" class="message-content markdown-content" v-html="renderMarkdown(message.content)"></div>
            <div v-else class="message-content">{{ message.content }}</div>
            <div v-if="message.role === 'assistant'" class="message-meta">
              节点: {{ message.metadata?.agent_type || '未知' }} · 端到端耗时: {{ formatLatency(message.metadata?.latency_ms) }} · 知识库: {{ message.metadata?.knowledge_used ? '已引用' : '无' }}
            </div>
          </div>
          <div v-if="message.role === 'assistant' && (message.metadata?.escalated || message.metadata?.agent_recommended_handoff)" class="handoff-card">
            提示：当前诉求涉及人工复核，建议转接人工坐席处理。
          </div>
        </article>

        <div v-if="sending" class="loading-copy" role="status">正在整理答复</div>
        <div v-if="errorMessage" class="error-copy" role="alert">{{ errorMessage }}</div>
      </div>

      <form class="composer" @submit.prevent="submit()">
        <label class="sr-only" for="chat-input">输入售后问题</label>
        <textarea id="chat-input" v-model="input" rows="2" placeholder="输入你的售后问题" @keydown="onKeydown"></textarea>
        <div class="composer-footer">
          <span class="composer-hint">⌘ / Ctrl + Enter 发送</span>
          <button class="primary-button send-button" type="submit" :disabled="sending || !input.trim()">发送</button>
        </div>
      </form>
    </section>
  </section>
</template>
