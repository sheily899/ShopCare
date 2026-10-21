import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import {
  formatHandoffLatency,
  emotionPresentation,
  urgencyPresentation,
  handoffReasonLabel,
  formatHandoffReasons,
  ticketTypeLabel,
  sessionStatusLabel,
  ticketSubtaskStatusLabel,
  handoffSourceLabel,
  businessTextLabel,
  departmentLabel,
} from '../src/lib/handoff-presentation.js'

test('handoff latencies are rendered in seconds with exactly two decimals', () => {
  assert.equal(formatHandoffLatency(987), '0.99 秒')
  assert.equal(formatHandoffLatency(2712), '2.71 秒')
  assert.equal(formatHandoffLatency(null), '—')
})

test('each supported emotion and urgency has its own semantic tone', () => {
  assert.deepEqual(emotionPresentation('平静'), { label: '平静', className: 'emotion-calm' })
  assert.deepEqual(emotionPresentation('焦虑'), { label: '焦虑', className: 'emotion-anxious' })
  assert.deepEqual(emotionPresentation('不满'), { label: '不满', className: 'emotion-upset' })
  assert.deepEqual(emotionPresentation('愤怒'), { label: '愤怒', className: 'emotion-angry' })
  assert.deepEqual(emotionPresentation('威胁升级'), { label: '威胁升级', className: 'emotion-threatening' })
  assert.deepEqual(urgencyPresentation('P3'), { label: 'P3', className: 'urgency-P3' })
  assert.deepEqual(urgencyPresentation('P0'), { label: 'P0', className: 'urgency-P0' })
})

test('unknown backend enums are never printed as customer-facing labels', () => {
  assert.equal(handoffReasonLabel('consumer_requested_human'), '消费者主动要求人工客服接手')
  assert.equal(handoffReasonLabel('unmapped_internal_reason'), '人工客服继续处理')
  assert.equal(ticketTypeLabel('adverse_reaction'), '不良反应')
  assert.equal(ticketTypeLabel('unmapped_ticket_type'), '其他工单')
  assert.equal(sessionStatusLabel('waiting_human'), '人工处理中')
  assert.equal(sessionStatusLabel('unknown_backend_status'), '处理中')
  assert.equal(ticketSubtaskStatusLabel('in_progress'), '处理中')
  assert.equal(ticketSubtaskStatusLabel('unknown_task_status'), '处理中')
  assert.equal(handoffSourceLabel({ type: 'action', id: 'adverse.confirm_medical_visit' }), '业务处理记录')
  assert.equal(businessTextLabel('adverse_reaction'), '不良反应与使用不适')
  assert.equal(businessTextLabel('unmapped_backend_enum'), '相关业务信息')
  assert.equal(businessTextLabel('SC-DEMO-4821'), 'SC-DEMO-4821')
  assert.equal(departmentLabel('quality'), '质量')
  assert.equal(departmentLabel('some_internal_group'), '相关部门')
})

test('urgency reasons render readable evidence, never object coercion or rule IDs', () => {
  assert.equal(
    formatHandoffReasons([{ rule_id: 'safety.breathing_or_emergency', text: '出现需要立即关注的安全风险', evidence: '脸红肿' }]),
    '出现需要立即关注的安全风险（“脸红肿”）',
  )
  assert.equal(formatHandoffReasons([{ rule_id: 'internal_enum' }]), '')
})

test('service workbench hides judgment reasons while keeping emotion, urgency and latency', async () => {
  const source = await readFile(new URL('../src/views/ServiceWorkbenchView.vue', import.meta.url), 'utf8')

  assert.doesNotMatch(source, /判断依据：/)
  assert.match(source, /情绪与紧急度/)
  assert.match(source, /分析用时/)
})

test('ticket submission offers a truthful progress reply for agent adoption instead of auto-sending', async () => {
  const source = await readFile(new URL('../src/views/ServiceWorkbenchView.vue', import.meta.url), 'utf8')
  const confirmFunction = source.match(/async function confirmTicket\(\) \{([\s\S]*?)\n\}/)?.[1]
  const sendFunction = source.match(/async function sendReply\(\) \{([\s\S]*?)\n\}/)?.[1]

  assert.ok(confirmFunction, '客服确认工单函数应存在')
  assert.ok(sendFunction, '客服发送消息函数应存在')
  assert.match(confirmFunction, /ticketSubmissionSuggestion\.value = \{\s*convId,\s*text: '您的工单已提交'/)
  assert.doesNotMatch(confirmFunction, /相关专员会核对后在这里回复您/)
  assert.doesNotMatch(confirmFunction, /sendServiceMessage|退款申请已通过|退款已批准/)
  assert.match(source, /progressSuggestion\?\.convId === selectedSession\.value\?\.conv_id/)
  assert.match(source, /ticketSubmissionSuggestion\.value = \{ \.\.\.ticketSubmissionSuggestion\.value, sent: true \}/)
  assert.match(source, /@click="serviceReply = suggestedReplies\[0\]"/)
  assert.match(sendFunction, /sendServiceMessage\(convId, serviceReply\.value\.trim\(\)/)
})
