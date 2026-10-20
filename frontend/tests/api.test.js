import assert from 'node:assert/strict'
import test from 'node:test'

import { confirmServiceTicket, getConsumerMessages, getServiceSession, getTraces, refreshServiceHandoff, searchKnowledge, streamServiceHandoffEvents, uploadKnowledge } from '../src/api/index.js'
import { mergeCustomerServiceMessages } from '../src/lib/consumer-message-sync.js'

test('ticket validation error preserves the backend message and missing fields', async (context) => {
  const originalFetch = globalThis.fetch
  context.after(() => { globalThis.fetch = originalFetch })

  globalThis.fetch = async () => new Response(JSON.stringify({
    detail: { message: '请先补充必填字段', missing_fields: ['order_id', 'symptom'] },
  }), {
    status: 409,
    headers: { 'Content-Type': 'application/json' },
  })

  await assert.rejects(
    confirmServiceTicket('conversation-1', 'agent-token'),
    (error) => {
      assert.equal(error.message, '请先补充必填字段')
      assert.deepEqual(error.details, { message: '请先补充必填字段', missing_fields: ['order_id', 'symptom'] })
      return true
    },
  )
})

test('handoff SSE resumes from the last event id and parses chunked updates', async (context) => {
  const originalFetch = globalThis.fetch
  context.after(() => { globalThis.fetch = originalFetch })
  let requestHeaders

  globalThis.fetch = async (_url, options) => {
    requestHeaders = new Headers(options.headers)
    const encoder = new TextEncoder()
    const chunks = [
      'id: 41\nevent: handoff.block.updated\ndata: {"block_type":"summary",',
      '"status":"ready","payload":{"text":"摘要已更新"}}\n\n',
    ]
    return new Response(new ReadableStream({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(encoder.encode(chunk))
        controller.close()
      },
    }), { status: 200, headers: { 'Content-Type': 'text/event-stream' } })
  }

  const events = []
  const result = await streamServiceHandoffEvents({
    convId: 'conv-1', token: 'agent-token', lastEventId: '40',
    onEvent: (eventType, payload, eventId) => events.push({ eventType, payload, eventId }),
  })

  assert.equal(requestHeaders.get('Last-Event-ID'), '40')
  assert.equal(requestHeaders.get('Authorization'), 'Bearer agent-token')
  assert.equal(result.lastEventId, '41')
  assert.deepEqual(events, [{
    eventType: 'handoff.block.updated',
    payload: { block_type: 'summary', status: 'ready', payload: { text: '摘要已更新' } },
    eventId: '41',
  }])
})

test('service session polling accepts an incremental message cursor', async (context) => {
  const originalFetch = globalThis.fetch
  context.after(() => { globalThis.fetch = originalFetch })
  let requestUrl
  let requestHeaders

  globalThis.fetch = async (url, options) => {
    requestUrl = String(url)
    requestHeaders = new Headers(options.headers)
    return new Response(JSON.stringify({ messages: [{ message_id: 'm2', message_seq: 2, content: '补充信息' }] }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    })
  }

  const result = await getServiceSession('conv-1', 'agent-token', 1)
  assert.match(requestUrl, /\/service\/sessions\/conv-1\?after_seq=1$/)
  assert.equal(requestHeaders.get('Authorization'), 'Bearer agent-token')
  assert.equal(result.messages[0].message_seq, 2)
})

test('manual handoff refresh sends an authenticated POST for the selected conversation', async (context) => {
  const originalFetch = globalThis.fetch
  context.after(() => { globalThis.fetch = originalFetch })
  let requestUrl
  let requestMethod
  let requestHeaders

  globalThis.fetch = async (url, options) => {
    requestUrl = String(url)
    requestMethod = options.method
    requestHeaders = new Headers(options.headers)
    return new Response(JSON.stringify({ success: true, refreshed: true }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    })
  }

  const result = await refreshServiceHandoff('selected-conversation', 'agent-token')
  assert.match(requestUrl, /\/service\/sessions\/selected-conversation\/refresh$/)
  assert.equal(requestMethod, 'POST')
  assert.equal(requestHeaders.get('Authorization'), 'Bearer agent-token')
  assert.equal(result.refreshed, true)
})

test('consumer message polling sends an owner token and sequence cursor', async (context) => {
  const originalFetch = globalThis.fetch
  context.after(() => { globalThis.fetch = originalFetch })
  let requestUrl
  let requestHeaders

  globalThis.fetch = async (url, options) => {
    requestUrl = String(url)
    requestHeaders = new Headers(options.headers)
    return new Response(JSON.stringify({ conv_id: 'conv-1', messages: [] }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    })
  }

  await getConsumerMessages('conv-1', 'consumer-token', 12)
  assert.match(requestUrl, /\/chat\/conv-1\/messages\?after_seq=12$/)
  assert.equal(requestHeaders.get('Authorization'), 'Bearer consumer-token')
})

test('knowledge and trace management requests send the authenticated agent token', async (context) => {
  const originalFetch = globalThis.fetch
  context.after(() => { globalThis.fetch = originalFetch })
  const captured = []
  globalThis.fetch = async (url, options) => {
    captured.push({ url: String(url), headers: new Headers(options.headers) })
    return new Response(JSON.stringify({ results: [], traces: [] }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    })
  }

  await searchKnowledge('成分', 5, 'agent-token')
  await uploadKnowledge(new Blob(['正文'], { type: 'text/plain' }), 'agent-token')
  await getTraces('agent-token')

  assert.deepEqual(captured.map((item) => new URL(item.url).pathname), [
    '/search', '/knowledge/upload', '/trace/tools',
  ])
  assert.ok(captured.every((item) => item.headers.get('Authorization') === 'Bearer agent-token'))
})

test('consumer sync appends public agent replies once and advances the global cursor', () => {
  const existing = [{ role: 'user', content: '申请退款', createdAt: '2026-10-19T00:00:00Z' }]
  const result = mergeCustomerServiceMessages(existing, 3, [
    { message_id: 'ai-4', message_seq: 4, sender_type: 'ai', content: '已登记', created_at: '2026-10-19T00:00:01Z' },
    { message_id: 'agent-5', message_seq: 5, sender_type: 'customer_service', content: '请补充退款原因', created_at: '2026-10-19T00:00:02Z' },
    { message_id: 'system-6', message_seq: 6, sender_type: 'system', content: '内部字段', created_at: '2026-10-19T00:00:03Z' },
  ])

  assert.equal(result.lastMessageSeq, 6)
  assert.equal(result.messages.length, 2)
  assert.equal(result.messages[1].role, 'customer_service')
  assert.equal(result.messages[1].content, '请补充退款原因')
  assert.equal(result.messages[1].messageId, 'agent-5')
  assert.equal(mergeCustomerServiceMessages(result.messages, 6, [
    { message_id: 'agent-5', message_seq: 5, sender_type: 'customer_service', content: '请补充退款原因' },
  ]).messages.length, 2)
})
