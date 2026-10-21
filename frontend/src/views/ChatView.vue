<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { answerSlot, getConsumerMessages, streamChat } from '../api/index.js'
import { mergeCustomerServiceMessages } from '../lib/consumer-message-sync.js'
import { latestQuestionAssistant, unansweredSlotQuestions } from '../lib/slot-questions.js'

const emit = defineEmits(['sessions-changed'])
const props = defineProps({ userId: { type: String, default: 'consumer-demo' }, token: { type: String, default: '' } })
const STORAGE_KEY = 'shopcare_sessions'
const quickPrompts = ['成分与肤质适配', '订单物流查询', '退货退款申请']

function agentLabel(value) {
  return { consult: 'Consult Agent', order: 'Order Agent', after_sales: 'After Sales Agent', escalation: '人工接管' }[value] || value || '未知'
}

function createId() {
  return globalThis.crypto?.randomUUID?.() || `conv-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function readSessions() {
  try {
    const stored = JSON.parse(localStorage.getItem(`${STORAGE_KEY}:${props.userId}`) || '[]')
    return Array.isArray(stored) ? stored : []
  } catch {
    return []
  }
}

function createSession() {
  return { id: createId(), createdAt: new Date().toISOString(), messages: [], lastMessageSeq: 0 }
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

function isHorizontalRule(line) {
  const normalized = String(line ?? '').trim()
  return normalized === '---' || normalized === '***' || normalized === '___'
    || /^(?:-{3,}|\*{3,}|_{3,})$/.test(normalized)
}

function renderMarkdown(value) {
  const lines = String(value ?? '').replace(/\r\n?/g, '\n').split('\n')
  const output = []
  let index = 0
  while (index < lines.length) {
    const line = lines[index]
    if (!line.trim()) { index += 1; continue }

    // Markdown 水平分隔线必须作为独立块处理，避免被当作普通文本显示为三个短横线。
    if (isHorizontalRule(line)) {
      output.push('<hr>')
      index += 1
      continue
    }

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
      if (paragraph.length && (/^\s*```/.test(current) || /^\s*(#{1,3})\s+/.test(current) || /^\s*[-*+]\s+/.test(current) || /^\s*\d+[.)]\s+/.test(current) || isHorizontalRule(current))) break
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
const streamStatus = ref('正在准备处理')
const progressSteps = ref([
])
let messageSyncTimer = null
let messageSyncInFlight = false

const activeSession = computed(() => sessions.value.find((session) => session.id === activeId.value) || sessions.value[0])
const messages = computed(() => activeSession.value?.messages || [])
const latestAssistantMessage = computed(() => latestQuestionAssistant(messages.value))

function sessionSummary(session) {
  const firstUserMessage = session.messages.find((item) => item.role === 'user')?.content || '新会话'
  return { id: session.id, title: firstUserMessage, createdAt: formatTime(session.createdAt) }
}

function emitSessionSummaries() {
  emit('sessions-changed', sessions.value.map(sessionSummary))
}

function persist() {
  localStorage.setItem(`${STORAGE_KEY}:${props.userId}`, JSON.stringify(sessions.value))
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
  progressSteps.value = progressSteps.value.map((step, index) => ({ ...step, state: index === 0 ? 'active' : 'pending' }))
  const progressId = `progress-${Date.now()}`
  const safetyOpening = /(烂脸|发红|红肿|起疹|刺痛|瘙痒|过敏|肿胀)/.test(text)
    ? '收到，请别担心，我先帮您把安全和后续处理理清楚。'
    : '收到，我先帮您核对这件事，马上给您下一步。'
  let progressMessage = { role: 'assistant', content: '正在读取会话上下文…', createdAt: new Date().toISOString(), localId: progressId, progress: true }
  updateSession({ ...session, messages: [...session.messages, progressMessage] })
  try {
    // 以浏览器请求生命周期计时，覆盖网络传输、后端处理与响应渲染前等待，
    // 这样展示给用户的耗时才对应真实体感，而不是只反映后端某个内部阶段。
    const clientStartedAt = performance.now()
    streamStatus.value = safetyOpening
    let result = null
    let streamedText = ''
    await streamChat({
      message: text,
      userId: props.userId,
      convId: session.id,
      token: props.token,
      onEvent: (event, payload) => {
        if (event !== 'done') {
          if (event === 'delta') {
            streamedText += payload.text || ''
            progressMessage = { ...progressMessage, content: streamedText }
          } else {
            if (['context', 'intent', 'knowledge', 'tools', 'response'].includes(event)) {
              const labels = { context: '读取会话上下文', intent: '意图识别', knowledge: '知识库调用', tools: '工具调用', response: '整理回复' }
              const next = progressSteps.value.filter((step) => step.key !== event)
              next.push({ key: event, label: labels[event], state: 'done' })
              progressSteps.value = next.map((step, index) => ({ ...step, state: index === next.length - 1 ? 'active' : 'done' }))
            }
          }
          streamStatus.value = payload.message || streamStatus.value
          if (event !== 'delta') progressMessage = { ...progressMessage, content: streamStatus.value }
          const current = sessions.value.find((item) => item.id === session.id)
          if (current) updateSession({ ...current, messages: current.messages.map((item) => item.localId === progressId ? progressMessage : item) })
        }
        if (event === 'done') result = payload
        if (event === 'error') throw new Error(payload.message || '请求未完成')
      },
    })
    if (!result) throw new Error('服务未返回完整答复')
    if (result.intent_diagnostics?.handoff_suppressed) {
      const current = sessions.value.find((item) => item.id === session.id) || session
      updateSession({
        ...current,
        messages: current.messages.filter((item) => item.localId !== progressId),
      })
      return
    }
    const clientLatencyMs = Math.max(0, performance.now() - clientStartedAt)
    const assistantMessage = {
      role: 'assistant',
      content: result.response || '当前没有可展示的回复。',
      createdAt: new Date().toISOString(),
      // UI 使用 client_latency_ms；server_latency_ms 保留后端 API 计时，便于定位网络与服务端耗时差异。
      metadata: { ...result, latency_ms: clientLatencyMs, client_latency_ms: clientLatencyMs, server_latency_ms: result.latency_ms },
    }
    const current = sessions.value.find((item) => item.id === session.id) || session
    const ackMessageId = result.intent_diagnostics?.ack_message_id
    const existingAck = ackMessageId && current.messages.some(
      (item) => item.metadata?.intent_diagnostics?.ack_message_id === ackMessageId,
    )
    updateSession({
      ...current,
      id: result.conv_id || session.id,
      messages: existingAck
        ? current.messages.filter((item) => item.localId !== progressId)
        : current.messages.map((item) => item.localId === progressId ? assistantMessage : item),
    })
    if (result.conv_id && result.conv_id !== activeId.value) activeId.value = result.conv_id
  } catch (error) {
    errorMessage.value = error.message || '请求未完成，请稍后重试。'
    const current = sessions.value.find((item) => item.id === session.id)
    if (current) {
      updateSession({
        ...current,
        messages: current.messages.filter((item) => item.localId !== progressId),
      })
    }
  } finally {
    sending.value = false
    progressSteps.value = []
    streamStatus.value = '正在准备处理'
  }
}

async function chooseSlot(message, question, option) {
  if (sending.value || !activeSession.value) return
  try {
    const result = await answerSlot(activeSession.value.id, {
      question_id: question.question_id,
      slot_id: question.slot_id,
      value: option.value,
    }, props.token)
    const current = sessions.value.find((item) => item.id === activeSession.value.id)
    if (!current) return
    const confirmation = `已记录：${option.label}`
    updateSession({ ...current, messages: [...current.messages, { role: 'user', content: option.label, createdAt: new Date().toISOString(), slotAnswer: true, slotId: question.slot_id, questionId: question.question_id }, { role: 'assistant', content: confirmation, createdAt: new Date().toISOString(), metadata: { slotAnswer: true, case_state: result.case_state } }] })
  } catch (error) {
    errorMessage.value = error.message || '选项暂时无法提交，请稍后重试。'
  }
}

async function syncConsumerMessages() {
  const session = activeSession.value
  if (!session?.id || !props.token || messageSyncInFlight) return
  messageSyncInFlight = true
  try {
    const response = await getConsumerMessages(session.id, props.token, session.lastMessageSeq || 0)
    if (activeSession.value?.id !== session.id) return
    const current = sessions.value.find((item) => item.id === session.id)
    if (!current) return
    const merged = mergeCustomerServiceMessages(
      current.messages,
      current.lastMessageSeq || 0,
      response.messages || [],
      response.last_message_seq,
    )
    if (merged.lastMessageSeq !== (Number(current.lastMessageSeq) || 0) || merged.messages.length !== current.messages.length) {
      updateSession({ ...current, messages: merged.messages, lastMessageSeq: merged.lastMessageSeq })
    }
  } catch {
    // 轮询只补齐另一端消息；会话尚未写入服务端或短暂断网时不打断消费者操作。
  } finally {
    messageSyncInFlight = false
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
watch(activeId, () => { void syncConsumerMessages() })

onMounted(() => {
  void syncConsumerMessages()
  messageSyncTimer = window.setInterval(syncConsumerMessages, 2000)
})
onUnmounted(() => {
  if (messageSyncTimer) window.clearInterval(messageSyncTimer)
})

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
            <div v-if="message.role === 'customer_service'" class="message-sender-label">人工客服</div>
            <details v-if="message.progress" class="progress-details" open>
              <summary>已收到，正在处理</summary>
              <div class="progress-steps">
                <div v-for="step in progressSteps" :key="step.key" class="progress-step" :class="step.state">
                  <span class="progress-dot">{{ step.state === 'done' ? '✓' : step.state === 'active' ? '·' : '○' }}</span>
                  <span>{{ step.label }}</span>
                </div>
              </div>
            </details>
            <div v-else-if="message.role === 'assistant'" class="message-content markdown-content" v-html="renderMarkdown(message.content)"></div>
            <div v-else class="message-content">{{ message.content }}</div>
            <div v-if="message.role === 'assistant' && !message.progress" class="message-meta">
              节点: {{ agentLabel(message.metadata?.agent_type) }} · 端到端耗时: {{ formatLatency(message.metadata?.latency_ms) }} · 知识库: {{ message.metadata?.knowledge_used ? '已引用' : '无' }}
            </div>
            <details v-if="message.role === 'assistant' && !message.progress && message.metadata" class="result-trace">
              <summary>查看本次真实处理结果</summary>
              <p>意图：{{ message.metadata.intent || '未返回' }}</p>
              <p>知识库：{{ message.metadata.knowledge_used ? '已实际引用' : '本次未调用' }}</p>
              <p>工具：{{ message.metadata.tools_used?.length ? message.metadata.tools_used.join('、') : '本次未调用工具' }}</p>
            </details>
          </div>
          <div v-if="message.role === 'assistant' && !message.progress && (message.metadata?.escalated || message.metadata?.agent_recommended_handoff)" class="handoff-card">
            已为您转接人工，请稍等。客服会继续查看当前会话，不需要您重复描述。
          </div>
          <div v-if="message === latestAssistantMessage && unansweredSlotQuestions(message, messages).length" class="follow-up-options">
            <span class="follow-up-label">为了更快帮您处理，可以直接选择：</span>
            <template v-for="question in unansweredSlotQuestions(message, messages)" :key="question.question_id">
              <span class="follow-up-label">{{ question.text }}</span>
              <button v-for="option in question.options" :key="`${question.question_id}-${option.value}`" class="chip-button" type="button" @click="chooseSlot(message, question, option)">{{ option.label }}</button>
            </template>
          </div>
        </article>

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
