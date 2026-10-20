<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref } from 'vue'
import { confirmServiceTicket, getServiceSessions, getServiceSession, refreshServiceHandoff, sendServiceMessage, streamServiceHandoffEvents, updateServiceTicket } from '../api/index.js'
import {
  emotionPresentation as presentEmotion,
  businessTextLabel,
  departmentLabel,
  formatHandoffLatency,
  formatHandoffReasons,
  handoffReasonLabel,
  handoffSourceLabel,
  sessionStatusLabel,
  ticketSubtaskStatusLabel,
  ticketTypeLabel,
  urgencyPresentation as presentUrgency,
} from '../lib/handoff-presentation.js'

const sessions = ref([])
const selectedSession = ref(null)
const messageList = ref(null)
const followsLatestMessage = ref(true)
const serviceReply = ref('')
const busy = ref(false)
const errorMessage = ref('')
const ticketStatus = ref('待审核')
const assignee = ref('质量')
const editingTicket = ref(false)
const editingFields = ref({})
const ticketError = ref('')
let refreshTimer = null
let handoffReconnectTimer = null
let handoffStreamController = null
let handoffStreamConvId = ''
let handoffEventCursor = '0'
const refreshing = ref(false)
const props = defineProps({ token: { type: String, default: '' }, agentId: { type: String, default: 'agent-demo' } })
const sortMode = ref('latest')
const handoffSummary = computed(() => selectedSession.value?.case_state?.handoff_summary || null)
const activeHandoff = computed(() => selectedSession.value?.handoff || null)
const handoffBlocks = computed(() => activeHandoff.value?.blocks || {})
const ticketDraft = computed(() => selectedSession.value?.case_state?.ticket_draft || null)
const suggestedReplies = computed(() => {
  const reply = handoffBlocks.value.next_steps?.payload?.suggested_reply
  if (activeHandoff.value) return reply ? [reply] : []
  return selectedSession.value?.case_state?.suggested_replies || []
})
const handoffEmotion = computed(() => handoffBlocks.value.emotion_urgency?.payload?.emotion?.value)
const handoffTemporaryUrgency = computed(() => handoffBlocks.value.emotion_urgency?.payload?.temporary?.value)
const handoffFinalUrgency = computed(() => handoffBlocks.value.emotion_urgency?.payload?.final?.value)
const displayMessages = computed(() => {
  const source = selectedSession.value?.messages || []
  return source.filter((message, index) => {
    const previous = source[index - 1]
    return !previous || previous.sender_type !== message.sender_type || previous.content !== message.content
  })
})
const emotionLabel = computed(() => {
  const value = handoffEmotion.value || selectedSession.value?.case_state?.emotion || handoffSummary.value?.emotion
  return presentEmotion(value).label
})
const priorityLabel = computed(() => handoffFinalUrgency.value || handoffTemporaryUrgency.value || handoffSummary.value?.urgency || selectedSession.value?.urgency || 'P3')
const emotionDisplay = computed(() => presentEmotion(handoffEmotion.value || selectedSession.value?.case_state?.emotion || handoffSummary.value?.emotion))
const priorityDisplay = computed(() => presentUrgency(priorityLabel.value))
const handoffReasonDisplay = computed(() => handoffReasonLabel(activeHandoff.value?.reason || handoffSummary.value?.handoff_reason))
const fieldLabels = {
  order_id: '订单号', product_name: '商品', sku: '规格', shade: '色号', quantity: '数量', symptom: '症状',
  symptom_area: '不适部位', onset_time: '出现时间', stopped_use: '是否停用', medical_visit: '是否就医',
  missing_item: '缺少商品', wrong_item: '错发商品', damaged_area: '破损部位', tracking_no: '物流单号',
  return_reason: '退货原因', opened_or_used: '拆封/使用', refund_amount: '退款金额', price_difference: '价差',
}
function fieldLabel(field) { return fieldLabels[field] || '其他信息' }
function sessionPriorityClass(session) {
  const level = presentUrgency(session?.urgency || 'P3').label
  return ['P0', 'P1', 'P2', 'P3'].includes(level) ? `priority-${level}` : 'priority-P3'
}
function ticketErrorText(error) {
  const missing = error?.details?.missing_fields
  if (Array.isArray(missing) && missing.length) {
    return `请先补充必填字段：${missing.map(fieldLabel).join('、')}`
  }
  return error?.message || '请先补充工单必填字段。'
}

const handoffStatusLabels = {
  pending: '生成中', ready: '已完成', degraded: '降级结果', failed: '生成失败',
}
function blockStatusLabel(block) {
  return handoffStatusLabels[block?.status] || '等待生成'
}
function isPostHandoffMessage(message) {
  const sourceSeq = Number(activeHandoff.value?.source_message_seq || 0)
  return message.sender_type === 'consumer' && sourceSeq > 0 && Number(message.message_seq || 0) > sourceSeq
}

function isNearLatestMessage() {
  const pane = messageList.value
  return !pane || pane.scrollHeight - pane.clientHeight - pane.scrollTop <= 96
}

function updateLatestMessageFollow() {
  followsLatestMessage.value = isNearLatestMessage()
}

async function scrollToLatest(force = false) {
  await nextTick()
  const pane = messageList.value
  if (!pane || (!force && !followsLatestMessage.value)) return
  pane.scrollTop = pane.scrollHeight
  followsLatestMessage.value = true
}

function renderAgentMarkdown(value) {
  const escape = (text) => String(text ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#39;')
  let html = escape(value)
  html = html.replace(/^&gt;\s?(.+)$/gm, '<div class="agent-callout">$1</div>')
  html = html.replace(/^\s*[-*]\s+(.+)$/gm, '<li>$1</li>')
  html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>')
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
  return html.replace(/\n/g, '<br>')
}

const sortedSessions = computed(() => {
  const items = [...sessions.value]
  if (sortMode.value === 'urgency') {
    const rank = { P0: 0, P1: 1, P2: 2, P3: 3 }
    return items.sort((a, b) => (rank[a.urgency] ?? 9) - (rank[b.urgency] ?? 9))
  }
  return items.sort((a, b) => String(b.updated_at || '').localeCompare(String(a.updated_at || '')))
})

async function selectSession(convId) {
  busy.value = true
  errorMessage.value = ''
  try {
    selectedSession.value = { ...(await getServiceSession(convId, props.token)), conv_id: convId }
    followsLatestMessage.value = true
    await scrollToLatest(true)
    ticketStatus.value = selectedSession.value.case_state?.ticket_draft?.status === 'submitted' ? '已提交' : '待审核'
    editingTicket.value = false
    ticketError.value = ''
    if (selectedSession.value.handoff) {
      if (handoffStreamConvId !== convId) startHandoffStream(convId, selectedSession.value.handoff_event_cursor)
    } else {
      stopHandoffStream()
    }
  } catch (error) {
    errorMessage.value = error.message || '会话读取失败。'
  } finally {
    busy.value = false
  }
}

async function loadSessions() {
  try {
    const response = await getServiceSessions(props.token)
    sessions.value = response.sessions || []
    if (sessions.value.length) await selectSession(sessions.value[0].conv_id)
  } catch (error) {
    errorMessage.value = error.message || '用户列表读取失败。'
  }
}

async function refreshCurrentSession() {
  try {
    const response = await getServiceSessions(props.token)
    sessions.value = response.sessions || []
    if (selectedSession.value?.conv_id) {
      const convId = selectedSession.value.conv_id
      const current = sessions.value.find((item) => item.conv_id === convId)
      if (current) {
        const existingMessages = selectedSession.value.messages || []
        const afterSeq = activeHandoff.value
          ? existingMessages.reduce((maximum, message) => Math.max(maximum, Number(message.message_seq) || 0), 0)
          : null
        const detail = await getServiceSession(convId, props.token, afterSeq)
        if (selectedSession.value?.conv_id !== convId) return
        const wasNearLatest = isNearLatestMessage()
        const existingIds = new Set(existingMessages.map((message) => message.message_id))
        const delta = (detail.messages || []).filter((message) => !existingIds.has(message.message_id))
        const messages = afterSeq == null ? (detail.messages || []) : [...existingMessages, ...delta]
        messages.sort((a, b) => (Number(a.message_seq) || 0) - (Number(b.message_seq) || 0))
        selectedSession.value = { ...detail, messages, conv_id: convId }
        if (delta.length && wasNearLatest) await scrollToLatest(true)
        ticketStatus.value = detail.case_state?.ticket_draft?.status === 'submitted' ? '已提交' : '待审核'
        if (detail.handoff) {
          if (handoffStreamConvId !== convId) startHandoffStream(convId, detail.handoff_event_cursor)
        } else if (handoffStreamConvId === convId) {
          stopHandoffStream()
        }
      }
    } else if (sessions.value.length) {
      await selectSession(sessions.value[0].conv_id)
    }
  } catch { /* 自动刷新失败不打断客服正在输入的内容 */ }
}

async function handleRefresh() {
  const convId = selectedSession.value?.conv_id
  if (!convId || refreshing.value) return
  refreshing.value = true
  errorMessage.value = ''
  try {
    if (activeHandoff.value) await refreshServiceHandoff(convId, props.token)
    await refreshCurrentSession()
  } catch (error) {
    errorMessage.value = error.message || '刷新会话信息失败。'
  } finally {
    refreshing.value = false
  }
}

function stopHandoffStream() {
  if (handoffReconnectTimer) window.clearTimeout(handoffReconnectTimer)
  handoffReconnectTimer = null
  handoffStreamController?.abort()
  handoffStreamController = null
  handoffStreamConvId = ''
}

function scheduleHandoffReconnect(convId, delay = 900) {
  if (handoffReconnectTimer) window.clearTimeout(handoffReconnectTimer)
  handoffReconnectTimer = window.setTimeout(() => {
    handoffReconnectTimer = null
    if (selectedSession.value?.conv_id === convId) connectHandoffStream(convId)
  }, delay)
}

function applyHandoffEvent(eventType, payload, eventId, convId) {
  if (selectedSession.value?.conv_id !== convId) return
  if (eventType === 'handoff.block.updated' && payload?.handoff_id !== activeHandoff.value?.handoff_id) {
    void refreshCurrentSession()
    return
  }
  if (eventType === 'handoff.block.updated' && payload?.handoff_id === activeHandoff.value?.handoff_id) {
    const blockType = payload.block_type
    if (!['summary', 'emotion_urgency', 'next_steps'].includes(blockType)) return
    const handoff = activeHandoff.value
    const block = {
      ...(handoff.blocks?.[blockType] || {}),
      status: payload.status,
      state_hash: payload.state_hash,
      payload: payload.payload || {},
      error_code: payload.error_code || '',
      latency_ms: payload.latency_ms,
    }
    const updatedHandoff = {
      ...handoff,
      last_event_id: Number(eventId) || handoff.last_event_id,
      blocks: { ...(handoff.blocks || {}), [blockType]: block },
    }
    const updatedSession = { ...selectedSession.value, handoff: updatedHandoff }
    if (blockType === 'emotion_urgency') {
      const emotion = block.payload?.emotion?.value
      const urgency = block.payload?.final?.value || block.payload?.temporary?.value
      updatedSession.urgency = urgency || updatedSession.urgency
      updatedSession.case_state = {
        ...(updatedSession.case_state || {}),
        ...(emotion ? { emotion } : {}),
        ...(block.payload?.final ? { final_urgency: block.payload.final } : {}),
        ...(urgency ? { urgency } : {}),
      }
    }
    selectedSession.value = updatedSession
    return
  }
  if (eventType === 'handoff.message.incremental' || eventType === 'handoff.started') {
    void refreshCurrentSession()
  }
}

async function connectHandoffStream(convId) {
  if (selectedSession.value?.conv_id !== convId || handoffStreamController) return
  const controller = new AbortController()
  handoffStreamController = controller
  try {
    const result = await streamServiceHandoffEvents({
      convId,
      token: props.token,
      lastEventId: handoffEventCursor,
      signal: controller.signal,
      onEvent: (eventType, payload, eventId) => {
        if (eventId) handoffEventCursor = String(eventId)
        if (eventType.startsWith('handoff.')) applyHandoffEvent(eventType, payload, eventId, convId)
      },
    })
    if (result.lastEventId) handoffEventCursor = String(result.lastEventId)
  } catch {
    // SSE 断开时自动重连；2.5 秒会话轮询始终保留为数据兜底。
  } finally {
    if (handoffStreamController === controller) handoffStreamController = null
    if (!controller.signal.aborted && selectedSession.value?.conv_id === convId) {
      scheduleHandoffReconnect(convId, 900)
    }
  }
}

function startHandoffStream(convId, eventCursor = 0) {
  stopHandoffStream()
  handoffStreamConvId = convId
  handoffEventCursor = String(eventCursor || 0)
  void connectHandoffStream(convId)
}

async function sendReply() {
  if (!selectedSession.value?.conv_id || !serviceReply.value.trim()) return
  busy.value = true
  try {
    await sendServiceMessage(selectedSession.value.conv_id, serviceReply.value.trim(), props.agentId, props.token)
    serviceReply.value = ''
    await selectSession(selectedSession.value.conv_id)
  } catch (error) {
    errorMessage.value = error.message || '客服回复发送失败。'
  } finally {
    busy.value = false
  }
}

function beginTicketEdit() {
  editingFields.value = { ...(ticketDraft.value?.fields || {}) }
  editingTicket.value = true
}

async function saveTicketEdit() {
  if (!selectedSession.value?.conv_id) return
  busy.value = true
  try {
    await updateServiceTicket(selectedSession.value.conv_id, editingFields.value, props.token, selectedSession.value.case_state?.version)
    editingTicket.value = false
    await selectSession(selectedSession.value.conv_id)
  } catch (error) {
    errorMessage.value = error.message || '工单修改失败。'
  } finally { busy.value = false }
}

async function confirmTicket() {
  if (!selectedSession.value?.conv_id) return
  busy.value = true
  try {
    const result = await confirmServiceTicket(selectedSession.value.conv_id, props.token)
    ticketStatus.value = result.status === 'submitted' ? '已提交' : '待审核'
    await selectSession(selectedSession.value.conv_id)
  } catch (error) {
    ticketError.value = ticketErrorText(error)
  } finally { busy.value = false }
}

onMounted(async () => {
  await loadSessions()
  refreshTimer = window.setInterval(refreshCurrentSession, 2500)
})
onUnmounted(() => {
  if (refreshTimer) window.clearInterval(refreshTimer)
  stopHandoffStream()
})
</script>

<template>
  <section class="service-page" aria-label="客服会话">
    <aside class="service-session-list service-panel">
      <div class="service-sort-bar">
        <span>消费者会话 · {{ sessions.length }}</span>
        <select v-model="sortMode" aria-label="会话排序方式">
          <option value="latest">按最新消息</option>
          <option value="urgency">按紧急程度</option>
        </select>
      </div>
      <button v-for="session in sortedSessions" :key="session.conv_id" class="service-session-item" :class="{ active: selectedSession?.conv_id === session.conv_id }" type="button" @click="selectSession(session.conv_id)">
        <strong>{{ session.user_id }} <em :class="sessionPriorityClass(session)">{{ presentUrgency(session.urgency || 'P3').label }}</em></strong><span>{{ sessionStatusLabel(session.status) }}</span>
        <small>{{ presentUrgency(session.risk_level).label === '待识别' ? '风险待识别' : presentUrgency(session.risk_level).label }} · {{ session.updated_at ? '最近有新消息' : '等待消息' }}</small>
      </button>
      <p v-if="!sessions.length" class="empty-cell">消费者发消息后会出现在这里</p>
    </aside>

    <section class="service-conversation service-panel">
      <div class="service-panel-heading"><h1>{{ selectedSession?.user_id || '请选择会话' }} <span class="order-label">订单 …4821</span></h1><div class="conversation-actions"><button class="outline-button" type="button" :disabled="refreshing" @click="handleRefresh">刷新</button></div></div>
      <div ref="messageList" class="service-message-list" aria-label="对话记录" @scroll="updateLatestMessageFollow">
        <div v-for="message in displayMessages" :key="message.message_id" class="service-message" :class="[message.sender_type, { 'is-incremental': isPostHandoffMessage(message) }]">
          <span class="service-message-author">{{ message.sender_type === 'consumer' ? '消费者' : message.sender_type === 'ai' ? 'AI 客服' : '人工客服' }}</span>
          <span v-if="isPostHandoffMessage(message)" class="incremental-message-label">转人工后新增</span>
          <div class="service-message-content" v-html="renderAgentMarkdown(message.content)"></div>
        </div>
        <p v-if="!selectedSession?.messages?.length" class="empty-cell">暂无会话内容</p>
      </div>
      <div class="service-composer-area">
        <div v-if="suggestedReplies.length" class="suggested-reply"><span>建议回复 · {{ emotionLabel }}</span><div class="reply-option"><p>{{ suggestedReplies[0] }}</p><button class="outline-button" type="button" @click="serviceReply = suggestedReplies[0]">采用</button></div></div>
        <form class="service-reply" @submit.prevent="sendReply">
          <input v-model="serviceReply" :disabled="busy || !selectedSession" placeholder="输入客服回复" />
          <button class="primary-button" type="submit" :disabled="busy || !serviceReply.trim() || !selectedSession">发送</button>
        </form>
      </div>
    </section>

    <aside class="service-risk service-panel">
      <div class="service-panel-heading"><h1>交接摘要</h1><span>客服可见</span></div>
      <div v-if="activeHandoff" class="risk-summary handoff-blocks">
        <section class="handoff-content-block" aria-live="polite">
          <div class="handoff-block-heading"><b>摘要</b><span class="handoff-block-status" :class="`status-${handoffBlocks.summary?.status || 'pending'}`" role="status">{{ blockStatusLabel(handoffBlocks.summary) }}</span></div>
          <p v-if="handoffBlocks.summary?.payload?.text">{{ handoffBlocks.summary.payload.text }}</p>
          <p v-else class="handoff-block-placeholder">正在整理已确认的诉求、AI 已采取的动作和未解决事项。</p>
          <small v-if="handoffBlocks.summary?.payload?.statements?.length" class="handoff-source-list">
            来源：<span v-for="(source, index) in handoffBlocks.summary.payload.statements.flatMap((item) => item.sources || [])" :key="`${source.type}-${source.seq || source.id}-${index}`">{{ handoffSourceLabel(source) }}{{ index < handoffBlocks.summary.payload.statements.flatMap((item) => item.sources || []).length - 1 ? '、' : '' }}</span>
          </small>
          <small v-if="handoffBlocks.summary?.latency_ms != null" class="handoff-latency">生成用时 {{ formatHandoffLatency(handoffBlocks.summary.latency_ms) }}</small>
        </section>

        <section class="handoff-content-block summary-priority-section" aria-live="polite">
          <div class="handoff-block-heading"><b>情绪与紧急度</b><span class="handoff-block-status" :class="`status-${handoffBlocks.emotion_urgency?.status || 'pending'}`" role="status">{{ blockStatusLabel(handoffBlocks.emotion_urgency) }}</span></div>
          <div class="summary-tags"><span class="summary-tag" :class="emotionDisplay.className">{{ handoffBlocks.emotion_urgency?.status === 'pending' ? '情绪分析中' : emotionDisplay.label }}</span><span class="summary-tag" :class="priorityDisplay.className">{{ priorityDisplay.label }}</span></div>
          <small v-if="formatHandoffReasons(Object.values(handoffBlocks.emotion_urgency?.payload?.emotion?.evidence || {}))">情绪依据：{{ formatHandoffReasons(Object.values(handoffBlocks.emotion_urgency?.payload?.emotion?.evidence || {})) }}</small>
          <small v-if="handoffBlocks.emotion_urgency?.latency_ms != null" class="handoff-latency">分析用时 {{ formatHandoffLatency(handoffBlocks.emotion_urgency.latency_ms) }}</small>
        </section>

        <section class="handoff-content-block" aria-live="polite">
          <div class="handoff-block-heading"><b>人工下一步</b><span class="handoff-block-status" :class="`status-${handoffBlocks.next_steps?.status || 'pending'}`" role="status">{{ blockStatusLabel(handoffBlocks.next_steps) }}</span></div>
          <ol v-if="handoffBlocks.next_steps?.payload?.items?.length" class="summary-next-steps"><li v-for="item in handoffBlocks.next_steps.payload.items" :key="item.id || item.text">{{ item.text || item }}</li></ol>
          <p v-else class="handoff-block-placeholder">{{ handoffBlocks.next_steps?.status === 'failed' ? '人工下一步暂时无法生成，请根据对话内容核对。' : '正在根据未完成事项整理人工处理动作。' }}</p>
          <small v-if="handoffBlocks.next_steps?.latency_ms != null" class="handoff-latency">整理用时 {{ formatHandoffLatency(handoffBlocks.next_steps.latency_ms) }}</small>
        </section>
        <span class="risk-tag">{{ handoffReasonDisplay }}</span>
      </div>
      <div v-else-if="handoffSummary" class="risk-summary">
        <div class="summary-section"><b>诉求</b><div class="summary-tags"><span v-for="item in (handoffSummary.request_items || [handoffSummary.request || handoffSummary.message])" :key="item" class="summary-tag">{{ businessTextLabel(item) }}</span></div></div>
        <div class="summary-section summary-priority-section"><b>情绪与紧急程度</b><div class="summary-tags"><span class="summary-tag" :class="emotionDisplay.className">{{ emotionDisplay.label }}</span><span class="summary-tag" :class="priorityDisplay.className">{{ priorityDisplay.label }}</span></div></div>
        <div v-if="handoffSummary.known_facts?.length" class="summary-section"><b>已知信息</b><div class="summary-tags"><span v-for="item in handoffSummary.known_facts" :key="item" class="summary-tag summary-fact">{{ businessTextLabel(item) }}</span></div></div>
        <div v-if="handoffSummary.key_facts?.length" class="summary-section"><b>关键情况</b><div class="summary-tags"><span v-for="item in handoffSummary.key_facts" :key="item" class="summary-tag summary-alert">{{ businessTextLabel(item) }}</span></div></div>
        <div v-if="handoffSummary.pending_items?.length" class="summary-section"><b>待确认</b><div class="summary-tags"><span v-for="item in handoffSummary.pending_items" :key="item" class="summary-tag summary-missing">{{ businessTextLabel(item) }}</span></div></div>
        <div v-if="handoffSummary.next_steps?.length" class="summary-section"><b>人工下一步</b><ol class="summary-next-steps"><li v-for="item in handoffSummary.next_steps" :key="item">{{ businessTextLabel(item) }}</li></ol></div>
        <span class="risk-tag">{{ handoffReasonDisplay }}</span>
      </div>
      <div v-else class="risk-placeholder">触发转人工后，这里显示该消费者的交接摘要。</div>
      <div v-if="ticketDraft" class="ticket-card">
        <div class="ticket-heading"><strong>工单 · {{ ticketTypeLabel(ticketDraft.ticket_type) }}</strong><span>{{ ticketStatus }}</span></div>
        <div class="ticket-system-fields">
          <div><small>工单号</small><strong>{{ ticketDraft.system_fields?.ticket_id || '待生成' }}</strong></div>
          <div><small>会话 ID</small><strong>{{ ticketDraft.system_fields?.conv_id || selectedSession?.conv_id || '—' }}</strong></div>
          <div><small>买家</small><strong>{{ ticketDraft.system_fields?.buyer_nickname || selectedSession?.user_id || '—' }}</strong></div>
          <div><small>创建时间</small><strong>{{ ticketDraft.system_fields?.created_at || '—' }}</strong></div>
        </div>
        <p><b>风险等级：</b>{{ presentUrgency(ticketDraft.risk_level).label }}</p>
        <p><b>业务字段</b></p>
        <div v-if="editingTicket" class="ticket-edit-fields">
          <label v-for="(value, field) in editingFields" :key="field">{{ fieldLabel(field) }}<input v-model="editingFields[field]" :placeholder="`请输入${fieldLabel(field)}`" /></label>
        </div>
        <div v-else class="field-tags"><span v-for="(value, field) in ticketDraft.fields" :key="field" v-if="value">{{ fieldLabel(field) }}：{{ businessTextLabel(value) }}</span><span v-if="!Object.values(ticketDraft.fields || {}).some(Boolean)">暂无</span></div>
        <p v-if="ticketDraft.sub_tasks?.length"><b>部门子任务</b></p>
        <div v-if="ticketDraft.sub_tasks?.length" class="field-tags"><span v-for="task in ticketDraft.sub_tasks" :key="task.department">{{ departmentLabel(task.department) }}：{{ businessTextLabel(task.task) }}（{{ ticketSubtaskStatusLabel(task.status) }}）</span></div>
        <p><b>待补信息</b></p><div class="field-tags missing"><span v-for="field in ticketDraft.missing_blocking_fields" :key="field">{{ fieldLabel(field) }}</span><span v-if="!ticketDraft.missing_blocking_fields?.length">已完整</span></div>
        <p v-if="ticketError" class="ticket-error" role="alert">{{ ticketError }}</p>
        <div class="ticket-actions"><template v-if="editingTicket"><button class="primary-button" type="button" @click="saveTicketEdit">保存</button><button class="outline-button" type="button" @click="editingTicket = false">取消</button></template><template v-else><button class="primary-button" type="button" @click="confirmTicket">确认提交</button><button class="outline-button" type="button" @click="beginTicketEdit">修改字段</button></template><select v-model="assignee" aria-label="分派处理组"><option>质量</option><option>仓储</option><option>财务</option><option>物流</option></select></div>
      </div>
      <div v-else class="ticket-card ticket-empty">当前会话无需创建人工工单。</div>
    </aside>
    <p v-if="errorMessage" class="error-copy" role="alert">{{ errorMessage }}</p>
  </section>
</template>
