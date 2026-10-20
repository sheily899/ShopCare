export function mergeCustomerServiceMessages(existingMessages, currentCursor, incomingMessages, serverCursor = null) {
  const messages = [...(existingMessages || [])]
  const knownIds = new Set(messages.map((message) => message.messageId).filter(Boolean))
  let lastMessageSeq = Math.max(0, Number(currentCursor) || 0)

  for (const message of incomingMessages || []) {
    const sequence = Number(message?.message_seq) || 0
    lastMessageSeq = Math.max(lastMessageSeq, sequence)
    if (message?.sender_type !== 'customer_service') continue
    const messageId = String(message.message_id || '')
    if (messageId && knownIds.has(messageId)) continue
    if (messageId) knownIds.add(messageId)
    messages.push({
      role: 'customer_service',
      content: String(message.content || ''),
      createdAt: message.created_at || new Date().toISOString(),
      messageId,
      messageSeq: sequence,
    })
  }
  lastMessageSeq = Math.max(lastMessageSeq, Number(serverCursor) || 0)

  return { messages, lastMessageSeq }
}
