import test from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { latestQuestionAssistant, unansweredSlotQuestions } from '../src/lib/slot-questions.js'

const css = await readFile(new URL('../src/styles.css', import.meta.url), 'utf8')
const app = await readFile(new URL('../src/App.vue', import.meta.url), 'utf8')
const chat = await readFile(new URL('../src/views/ChatView.vue', import.meta.url), 'utf8')
const service = await readFile(new URL('../src/views/ServiceWorkbenchView.vue', import.meta.url), 'utf8')

test('consumer chat uses the full app viewport instead of a capped card', () => {
  assert.match(app, /app-main-chat/)
  assert.match(css, /\.chat-workspace\s*\{[^}]*height:\s*100%/s)
  assert.match(css, /\.chat-vessel\s*\{[^}]*height:\s*100%[^}]*min-height:\s*0/s)
  assert.doesNotMatch(css, /\.chat-vessel\s*\{[^}]*height:\s*min\(75vh,\s*720px\)/s)
})

test('service chat keeps the composer in the viewport and follows latest messages on session entry', () => {
  assert.match(app, /app-main-service/)
  assert.match(css, /\.service-page\s*\{[^}]*height:\s*100%[^}]*min-height:\s*0/s)
  assert.match(css, /\.service-message-list\s*\{[^}]*min-height:\s*0[^}]*overflow-y:\s*auto/s)
  assert.match(service, /ref="messageList"/)
  assert.match(service, /scrollToLatest\(true\)/)
  assert.match(service, /service-composer-area/)
})

test('service refresh updates the selected handoff and hides local-only takeover toggles', () => {
  assert.match(service, /@click="handleRefresh"/)
  assert.match(service, /refreshServiceHandoff\(/)
  assert.doesNotMatch(service, /交还 AI|>接管</)
})

test('consumer slot choices only show for the latest unanswered question', () => {
  assert.match(chat, /latestAssistantMessage = computed/)
  assert.match(chat, /unansweredSlotQuestions\(message, messages\)/)
})

test('choosing one slot keeps the remaining choices under the original question', () => {
  const question = {
    role: 'assistant',
    metadata: { pending_questions: [
      { slot_id: 'opened_or_used', text: '商品是否拆封？', options: [] },
      { slot_id: 'return_scope', text: '退款范围？', options: [] },
    ] },
  }
  const messages = [
    question,
    { role: 'user', content: '已拆封', slotAnswer: true, slotId: 'opened_or_used' },
    { role: 'assistant', content: '已记录：已拆封', metadata: { slotAnswer: true } },
  ]

  assert.equal(latestQuestionAssistant(messages), question)
  assert.deepEqual(unansweredSlotQuestions(question, messages).map((item) => item.slot_id), ['return_scope'])
})

test('all selected slots disappear without hiding the recorded selections', () => {
  const question = {
    role: 'assistant',
    metadata: { pending_questions: [{ slot_id: 'return_scope', text: '退款范围？', options: [] }] },
  }
  const messages = [question, { role: 'user', content: '整单退款', slotAnswer: true, slotId: 'return_scope' }]

  assert.deepEqual(unansweredSlotQuestions(question, messages), [])
})

test('a later real assistant answer replaces the previous active slot question', () => {
  const originalQuestion = {
    role: 'assistant',
    metadata: { pending_questions: [{ slot_id: 'return_scope', text: '退款范围？', options: [] }] },
  }
  const laterAnswer = { role: 'assistant', content: '我已按整单退款记录。', metadata: {} }

  assert.equal(latestQuestionAssistant([
    originalQuestion,
    { role: 'user', content: '整单退款', slotAnswer: true, slotId: 'return_scope' },
    { role: 'assistant', content: '已记录：整单退款', metadata: { slotAnswer: true } },
    laterAnswer,
  ]), laterAnswer)
})
