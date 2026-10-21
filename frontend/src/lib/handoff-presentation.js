const EMOTIONS = {
  calm: ['平静', 'emotion-calm'],
  neutral: ['平静', 'emotion-calm'],
  平静: ['平静', 'emotion-calm'],
  anxious: ['焦虑', 'emotion-anxious'],
  worried: ['焦虑', 'emotion-anxious'],
  焦虑: ['焦虑', 'emotion-anxious'],
  upset: ['不满', 'emotion-upset'],
  disappointed: ['不满', 'emotion-upset'],
  dissatisfied: ['不满', 'emotion-upset'],
  不满: ['不满', 'emotion-upset'],
  angry: ['愤怒', 'emotion-angry'],
  愤怒: ['愤怒', 'emotion-angry'],
  threatening: ['威胁升级', 'emotion-threatening'],
  威胁升级: ['威胁升级', 'emotion-threatening'],
}

const HANDOFF_REASON_LABELS = {
  consumer_requested_human: '消费者主动要求人工客服接手',
}

const BUSINESS_TEXT_LABELS = {
  ingredient_fit: '成分与肤质适配',
  shade_match: '色号与搭配推荐',
  usage_howto: '商品使用方法',
  product_info: '商品信息咨询',
  promo_rules: '活动规则咨询',
  pregnancy_safety: '特殊人群适用咨询',
  order_status: '订单进度查询',
  logistics_stalled: '物流停滞与催查',
  signed_not_received: '显示签收但未收到',
  intercept_change_address: '拦截或修改地址',
  missing_item: '漏发与补发',
  wrong_item: '错发与换货',
  damaged_on_arrival: '到手破损',
  return_refund: '退货退款申请',
  adverse_reaction: '不良反应与使用不适',
  suspected_abnormal_refund: '疑似异常退款',
  price_protection: '价保补差',
  refund_not_received: '退款到账异常',
  shipping_fee_reimbursement: '运费报销',
  consumer_requested_human: '消费者主动要求人工客服接手',
  pending: '待处理',
  in_progress: '处理中',
  submitted: '已提交',
  resolved: '已解决',
  blocked: '暂时受阻',
  failed: '未完成',
  success: '已完成',
}

const DEPARTMENT_LABELS = {
  quality: '质量',
  warehouse: '仓储',
  finance: '财务',
  logistics: '物流',
}

const TICKET_TYPE_LABELS = {
  logistics: '物流问题',
  reissue_exchange: '补发/换货',
  return_refund: '退货退款',
  adverse_reaction: '不良反应',
  refund_risk: '退款异常',
  price_protection: '价保',
  refund_payment: '退款到账',
}

const SESSION_STATUS_LABELS = {
  ai_processing: 'AI处理中',
  waiting_human: '人工处理中',
  resolved: '已解决',
  closed: '已结束',
}

const SUBTASK_STATUS_LABELS = {
  pending: '待处理',
  in_progress: '处理中',
  submitted: '已提交',
  completed: '已完成',
  success: '已完成',
  resolved: '已解决',
  blocked: '暂时受阻',
  failed: '未完成',
}

function isBackendEnum(value) {
  return /^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$/i.test(value)
}

export function formatHandoffLatency(latencyMs) {
  if (latencyMs == null || latencyMs === '') return '—'
  const value = Number(latencyMs)
  if (!Number.isFinite(value)) return '—'
  return `${(Math.max(0, value) / 1000).toFixed(2)} 秒`
}

export function emotionPresentation(value) {
  const normalized = typeof value === 'string' ? value.trim() : ''
  const [label, className] = EMOTIONS[normalized] || ['待识别', 'emotion-unknown']
  return { label, className }
}

export function urgencyPresentation(value) {
  const normalized = typeof value === 'string' ? value.trim().toUpperCase() : ''
  if (!['P0', 'P1', 'P2', 'P3'].includes(normalized)) {
    return { label: '待识别', className: 'urgency-unknown' }
  }
  return { label: normalized, className: `urgency-${normalized}` }
}

export function handoffReasonLabel(value) {
  if (typeof value !== 'string' || !value.trim()) return '人工客服继续处理'
  const normalized = value.trim()
  if (HANDOFF_REASON_LABELS[normalized]) return HANDOFF_REASON_LABELS[normalized]
  return isBackendEnum(normalized) ? '人工客服继续处理' : normalized
}

export function formatHandoffReasons(reasons) {
  if (!Array.isArray(reasons)) return ''
  const readable = reasons.map((item) => {
    if (typeof item === 'string') return isBackendEnum(item.trim()) ? '' : item.trim()
    if (!item || typeof item !== 'object') return ''
    const text = [item.text, item.reason]
      .find((candidate) => typeof candidate === 'string' && candidate.trim() && !isBackendEnum(candidate.trim()))
    if (!text) return ''
    const evidence = typeof item.evidence === 'string' ? item.evidence.trim() : ''
    return evidence && !text.includes(evidence) ? `${text.trim()}（“${evidence}”）` : text.trim()
  }).filter(Boolean)
  return [...new Set(readable)].join('；')
}

export function ticketTypeLabel(value) {
  if (typeof value !== 'string' || !value.trim()) return '其他工单'
  const normalized = value.trim()
  return TICKET_TYPE_LABELS[normalized] || (isBackendEnum(normalized) ? '其他工单' : normalized)
}

export function sessionStatusLabel(value) {
  if (typeof value !== 'string' || !value.trim()) return '处理中'
  const normalized = value.trim()
  return SESSION_STATUS_LABELS[normalized] || (isBackendEnum(normalized) ? '处理中' : normalized)
}

export function ticketSubtaskStatusLabel(value) {
  if (typeof value !== 'string' || !value.trim()) return '待处理'
  const normalized = value.trim()
  return SUBTASK_STATUS_LABELS[normalized] || (isBackendEnum(normalized) ? '处理中' : normalized)
}

export function handoffSourceLabel(source) {
  if (!source || typeof source !== 'object') return ''
  if (source.type === 'message' && Number.isFinite(Number(source.seq))) return `消息 #${source.seq}`
  return '业务处理记录'
}

export function businessTextLabel(value) {
  if (typeof value !== 'string' || !value.trim()) return ''
  const normalized = value.trim()
  if (BUSINESS_TEXT_LABELS[normalized]) return BUSINESS_TEXT_LABELS[normalized]
  return /^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)+$/.test(normalized) ? '相关业务信息' : normalized
}

export function departmentLabel(value) {
  if (typeof value !== 'string' || !value.trim()) return '相关部门'
  const normalized = value.trim()
  return DEPARTMENT_LABELS[normalized] || (/^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)+$/.test(normalized) ? '相关部门' : normalized)
}
