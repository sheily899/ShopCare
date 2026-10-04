<script setup>
import { computed, nextTick, ref, watch } from 'vue'
import { sendChat } from '../api/index.js'

const STORAGE_KEY = 'shopcare_sessions'
const USER_ID = 'anonymous'
const quickPrompts = ['退换货运费险规定', '设备开机故障排查', '发票开具时效说明', '退款审核规则']

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
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;')
}

function renderMarkdown(value) {
  let html = escapeHtml(value).replaceAll('\n', '<br>')
  html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+|mailto:[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>')
  html = html.replace(/(^|<br>)(#{1,3})\s([^<]+)/g, '$1<strong>$3</strong>')
  return html
}

function formatTime(value) {
  return new Intl.DateTimeFormat('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }).format(new Date(value))
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

function persist() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions.value))
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
  activeId.value = id
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
    const result = await sendChat({ message: text, userId: USER_ID, convId: session.id })
    const assistantMessage = {
      role: 'assistant',
      content: result.response || '当前没有可展示的回复。',
      createdAt: new Date().toISOString(),
      metadata: result,
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
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    submit()
  }
}

watch(messages, async () => {
  await nextTick()
  if (stream.value) stream.value.scrollTop = stream.value.scrollHeight
}, { deep: true })
</script>

<template>
  <section class="chat-layout" aria-label="会话工作区">
    <aside class="session-sidebar">
      <button class="outline-button new-session" type="button" @click="startNewConversation">开启新对话</button>
      <div class="sidebar-label">历史会话</div>
      <div class="session-list" role="list">
        <button
          v-for="session in sessions"
          :key="session.id"
          class="session-item"
          :class="{ selected: session.id === activeId }"
          type="button"
          role="listitem"
          @click="selectSession(session.id)"
        >
          <span class="session-title">{{ session.messages.find((item) => item.role === 'user')?.content || '新会话' }}</span>
          <span class="session-time">{{ formatTime(session.createdAt) }}</span>
        </button>
      </div>
    </aside>

    <section class="chat-panel">
      <div ref="stream" class="message-stream" aria-live="polite">
        <div v-if="!messages.length" class="welcome-block">
          <p>售后服务通道。可咨询订单物流、退换政策与设备排障。</p>
          <div class="quick-prompts" aria-label="常用问题">
            <button v-for="prompt in quickPrompts" :key="prompt" class="chip-button" type="button" @click="submit(prompt)">
              {{ prompt }}
            </button>
          </div>
        </div>

        <article v-for="(message, index) in messages" :key="`${message.createdAt}-${index}`" class="message-row" :class="message.role">
          <div class="message-bubble">
            <div v-if="message.role === 'assistant'" class="message-content markdown-content" v-html="renderMarkdown(message.content)"></div>
            <div v-else class="message-content">{{ message.content }}</div>
            <div v-if="message.role === 'assistant'" class="message-meta">
              节点: {{ message.metadata?.agent_type || '未知' }}　耗时: {{ message.metadata?.latency_ms ?? '—' }}ms　知识库: {{ message.metadata?.knowledge_used ? '已引用' : '无' }}
            </div>
          </div>
          <div v-if="message.role === 'assistant' && (message.metadata?.escalated || message.metadata?.agent_recommended_handoff)" class="handoff-card">
            {{ message.metadata.intent === 'refund' || message.metadata.intent === 'payment_issue' ? '当前涉及退款与账单核验，建议转人工客服处理。' : message.metadata.intent === 'account_security' ? '当前涉及账号与资金安全，建议转人工客服处理。' : message.metadata.intent === 'technical_crash' ? '当前涉及系统异常报错，建议转人工客服处理。' : message.metadata.intent === 'complaint' ? '当前涉及服务投诉与纠纷，建议转人工客服处理。' : '当前问题较为复杂，建议转人工客服处理。' }}
          </div>
        </article>

        <div v-if="sending" class="loading-copy" role="status">正在整理答复</div>
        <div v-if="errorMessage" class="error-copy" role="alert">{{ errorMessage }}</div>
      </div>

      <form class="composer" @submit.prevent="submit()">
        <label class="sr-only" for="chat-input">输入售后问题</label>
        <textarea id="chat-input" v-model="input" rows="2" placeholder="输入你的售后问题" @keydown="onKeydown"></textarea>
        <button class="primary-button" type="submit" :disabled="sending || !input.trim()">发送</button>
      </form>
    </section>
  </section>
</template>
