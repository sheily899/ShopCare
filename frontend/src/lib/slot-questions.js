export function latestQuestionAssistant(messages) {
  return [...(messages || [])].reverse().find((item) => (
    item.role === 'assistant' && !item.progress && !item.metadata?.slotAnswer
  ))
}

export function userMessageAnswersSlot(slotId, content) {
  const text = String(content || '')
  const patterns = {
    stopped_use: /已停用|停用了|停止使用|目前没有在使用|现在没有在使用|不再使用|还在使用|仍在使用|继续使用|还在用|仍在用/,
    medical_visit: /已就医|看过医生|去过医院|没就医|未就医|没有就医|没去医院|没有去医院/,
    opened_or_used: /未拆封|没拆封|没有拆封|已拆封未使用|拆封未用|开封未用|已使用|使用过|用过|用了.{0,8}(?:天|周|星期|月)|已开封|已经拆封|开封了/,
    gift_opened_or_used: /小样.{0,8}(?:未拆封|没拆|没有拆|已拆|开封|用过|使用)|赠品.{0,8}(?:未拆封|没拆|没有拆|已拆|开封|用过|使用)/,
    return_scope: /整单|全单|只退正装|仅退正装|只退小样|只退赠品|分开退|赠品一起退|小样一起退/,
    return_reason: /买错|买重|拍错|下错单|不喜欢|不想要|不需要|不合适|不适合|使用后不适|用后不适|过敏|红肿|发红|起疹|刺痛|瘙痒|脱皮/,
    order_id: /(?:订单号|订单编号)?\s*[A-Z]{1,5}[-_]?[A-Z0-9-]{5,}/i,
  }
  return Boolean(patterns[slotId]?.test(text))
}

export function unansweredSlotQuestions(questionMessage, messages) {
  const questions = questionMessage?.metadata?.pending_questions || []
  const questionIndex = (messages || []).indexOf(questionMessage)
  if (!questions.length || questionIndex < 0) return []

  const laterMessages = messages.slice(questionIndex + 1)
  return questions.filter((question) => !laterMessages.some((item) => (
    item.role === 'user' && (
      (item.slotAnswer && item.slotId === question.slot_id)
      || userMessageAnswersSlot(question.slot_id, item.content)
    )
  )))
}
