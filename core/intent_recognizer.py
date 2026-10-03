from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
import re
from typing import Any, Dict, List, Optional, Tuple


class IntentCategory(str, Enum):
    QUERY = "query"
    COMPLAINT = "complaint"
    REQUEST = "request"
    GREETING = "greeting"
    TECHNICAL = "technical"
    BILLING = "billing"
    ACCOUNT = "account"
    FEEDBACK = "feedback"
    ORDER_STATUS = "order_status"
    LOGISTICS = "logistics"
    REFUND = "refund"
    INVOICE = "invoice"
    PAYMENT_ISSUE = "payment_issue"
    ACCOUNT_SECURITY = "account_security"
    TECHNICAL_LOGIN = "technical_login"
    TECHNICAL_CRASH = "technical_crash"
    HUMAN_HANDOFF = "human_handoff"
    ESCALATION = HUMAN_HANDOFF
    OTHER = "other"


class UrgencyLevel(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class IntentResult:
    intent: IntentCategory
    confidence: float
    urgency: UrgencyLevel
    intent_group: str
    entities: Dict[str, List[str]]
    reasoning: str
    latency_ms: float
    source_scores: Dict[str, float] = field(default_factory=dict)
    source_intents: Dict[str, str] = field(default_factory=dict)
    secondary_intents: List[IntentCategory] = field(default_factory=list)
    compound_trigger: str = ""


# Specific patterns take precedence over generic domain labels when scores tie.
# Values are regular-expression fragments; ASCII terms use explicit word
# boundaries where a substring match would create false positives.
SPECIFIC_PATTERNS: Dict[IntentCategory, Tuple[str, ...]] = {
    IntentCategory.ORDER_STATUS: (
        "订单状态",
        "订单进度",
        "订单处理到哪",
        "发货了吗",
        r"\border\s+status\b",
        r"\border\s+progress\b",
    ),
    IntentCategory.LOGISTICS: (
        "物流",
        "快递",
        "配送",
        "运单",
        r"\bdelivery\b",
        r"\bshipping\b",
    ),
    IntentCategory.REFUND: (
        "退款",
        "退货",
        r"\brefund\b",
        r"\breturn\b",
    ),
    IntentCategory.INVOICE: (
        "发票",
        "发票抬头",
        "税号",
        r"\binvoice\b",
    ),
    IntentCategory.PAYMENT_ISSUE: (
        "重复扣款",
        "多扣",
        "支付失败",
        "扣费",
        r"\bpayment\s+failed\b",
        r"\bpayment\s+declined\b",
    ),
    IntentCategory.ACCOUNT_SECURITY: (
        "账户被盗",
        "账号被盗",
        "异常登录",
        "重置密码",
        "两步验证",
        "账户安全",
    ),
    IntentCategory.TECHNICAL_LOGIN: (
        "无法登录",
        "登录失败",
        "登录不了",
        r"\b401\b",
        "验证码",
        r"\blogin\s+failed\b",
    ),
    IntentCategory.TECHNICAL_CRASH: (
        "崩溃",
        "闪退",
        r"\b500\b",
        "报错",
        r"\bcrash\b",
        r"\bserver\s+error\b",
    ),
    IntentCategory.HUMAN_HANDOFF: (
        "转人工",
        "人工客服",
        "人工处理",
        "人工介入",
        "升级人工",
        "找人工",
        r"\bsupervisor\b",
    ),
}

GENERIC_PATTERNS: Dict[IntentCategory, Tuple[str, ...]] = {
    IntentCategory.QUERY: (
        "查询",
        "查一下",
        "怎么",
        "什么",
        r"\?",
        "？",
        r"\bstatus\b",
    ),
    IntentCategory.COMPLAINT: (
        "投诉",
        "太差",
        "糟糕",
        "不满意",
        "等了很久",
        r"\bhorrible\b",
    ),
    IntentCategory.REQUEST: (
        "帮我",
        "需要",
        "请协助",
        "想要",
        r"\bplease\b",
        r"\bhelp\b",
    ),
    IntentCategory.GREETING: (
        "你好",
        "嗨",
        r"\bhello\b",
        r"\bhi\b",
        "早上好",
        "晚上好",
    ),
    IntentCategory.TECHNICAL: (
        "技术问题",
        "页面打不开",
        "页面无法打开",
        "网络请求失败",
        "网络失败",
        r"\berror\b",
    ),
    IntentCategory.BILLING: (
        "账单",
        "金额核对",
        "账单金额",
        "费用",
    ),
    IntentCategory.ACCOUNT: (
        "账户",
        "账号",
        "邮箱",
        "密码",
        r"\bpassword\b",
    ),
    IntentCategory.FEEDBACK: (
        "反馈",
        "建议",
        "好评",
        "满意",
        "服务很棒",
    ),
    IntentCategory.OTHER: (
        "其他",
        "不清楚",
        "不知道",
    ),
}

_SPECIFIC_PATTERNS = SPECIFIC_PATTERNS
_GENERIC_PATTERNS = GENERIC_PATTERNS
_SPECIFIC_INTENTS = frozenset(SPECIFIC_PATTERNS)
_GENERIC_INTENTS = frozenset(GENERIC_PATTERNS)

_URGENCY_KEYWORDS: Dict[UrgencyLevel, Tuple[str, ...]] = {
    UrgencyLevel.CRITICAL: ("紧急", "emergency", "urgent", "asap", "立刻"),
    UrgencyLevel.HIGH: ("今天", "马上", "尽快", "hurry", "now"),
    UrgencyLevel.MEDIUM: ("这周", "soon", "快点"),
}

_ENTITY_KEYS = ("order_id", "product", "date", "amount", "error_code")
_ORDER_ID_RE = re.compile(
    r"(?:订单号|订单编号|order(?:[_\s-]?id)?|order\s+(?:number|no)|#)"
    r"\s*[:：#]?\s*([A-Za-z0-9][A-Za-z0-9_-]{3,31})",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"(?:\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?|"
    r"今天|明天|昨天|近一周|最近一周|本周|这周|上周|下周)"
)
_AMOUNT_RE = re.compile(
    r"(?:[¥￥]\s*\d+(?:\.\d{1,2})?|"
    r"\d+(?:\.\d{1,2})?\s*(?:元|块|人民币|rmb|cny|usd|美元)|"
    r"(?:rmb|cny|usd)\s*\d+(?:\.\d{1,2})?)",
    re.IGNORECASE,
)
_LABELED_ERROR_CODE_RE = re.compile(
    r"(?:error(?:[_\s-]?code)?|错误码|状态码|http)\s*[:：#]?\s*([45]\d{2})\b",
    re.IGNORECASE,
)
_STANDALONE_ERROR_CODE_RE = re.compile(r"\b[45]\d{2}\b")


def _pattern_matches(pattern: str, message: str) -> bool:
    return re.search(pattern, message, flags=re.IGNORECASE) is not None


def _pattern_score(hits: int) -> float:
    if hits <= 0:
        return 0.0
    return min(1.0, 0.5 + 0.25 * (hits - 1))


def match_patterns(message: str) -> Dict[str, Any]:
    """Match specific and generic rules without calling an external service.

    The return shape keeps the old ``intent``/``confidence`` keys and adds the
    candidate scores needed by later fusion steps.
    """
    normalized = message or ""
    scores: Dict[IntentCategory, float] = {}
    hit_counts: Dict[IntentCategory, int] = {}
    specificity: Dict[IntentCategory, int] = {}

    for is_specific, pattern_table in (
        (True, SPECIFIC_PATTERNS),
        (False, GENERIC_PATTERNS),
    ):
        for intent, patterns in pattern_table.items():
            hits = sum(_pattern_matches(pattern, normalized) for pattern in patterns)
            if hits:
                hit_counts[intent] = hits
                scores[intent] = _pattern_score(hits)
                specificity[intent] = 1 if is_specific else 0

    ranked = sorted(
        scores,
        key=lambda intent: (
            scores[intent],
            specificity.get(intent, 0),
            -list(scores).index(intent),
        ),
        reverse=True,
    )
    best_intent = ranked[0] if ranked else IntentCategory.OTHER
    best_score = scores.get(best_intent, 0.0)
    specific_candidates = [intent for intent in scores if intent in _SPECIFIC_INTENTS]
    specific_intent = max(
        specific_candidates,
        key=lambda intent: (scores[intent], -list(scores).index(intent)),
        default=IntentCategory.OTHER,
    )
    specific_score = scores.get(specific_intent, 0.0)
    candidates = list(ranked)

    return {
        "intent": best_intent,
        "confidence": best_score,
        "candidates": candidates,
        "candidate_intents": candidates,
        "scores": {intent.value: score for intent, score in scores.items()},
        "hits": {intent.value: count for intent, count in hit_counts.items()},
        "specific_intent": specific_intent,
        "specific_score": specific_score,
    }


def _pattern_recognize(message: str) -> Dict[str, Any]:
    """Compatibility name for the original synchronous pattern strategy."""
    return match_patterns(message)


def _unique(values: List[str]) -> List[str]:
    return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))


def extract_entities(message: str) -> Dict[str, List[str]]:
    """Extract the five fact-based entity fields used by the intent contract."""
    text = message or ""
    labeled_codes = _LABELED_ERROR_CODE_RE.findall(text)
    standalone_codes = _STANDALONE_ERROR_CODE_RE.findall(text)
    return {
        "order_id": _unique(_ORDER_ID_RE.findall(text)),
        "product": [],
        "date": _unique(_DATE_RE.findall(text)),
        "amount": _unique(_AMOUNT_RE.findall(text)),
        "error_code": _unique(labeled_codes + standalone_codes),
    }


def _contains_urgency_keyword(message: str, keyword: str) -> bool:
    if keyword.isascii() and keyword.replace("_", "").replace("-", "").isalnum():
        return re.search(rf"\b{re.escape(keyword)}\b", message, re.IGNORECASE) is not None
    return keyword.casefold() in message.casefold()


def calculate_urgency(
    message: str,
    intent: Optional[IntentCategory] = None,
) -> UrgencyLevel:
    """Apply the fixed CRITICAL -> HIGH -> MEDIUM -> intent fallback order."""
    text = message or ""
    for level in (UrgencyLevel.CRITICAL, UrgencyLevel.HIGH, UrgencyLevel.MEDIUM):
        if any(_contains_urgency_keyword(text, word) for word in _URGENCY_KEYWORDS[level]):
            return level
    if intent == IntentCategory.HUMAN_HANDOFF:
        return UrgencyLevel.HIGH
    if intent == IntentCategory.COMPLAINT:
        return UrgencyLevel.MEDIUM
    return UrgencyLevel.LOW


def apply_500_override(
    message: str,
    intent: IntentCategory,
    confidence: float,
) -> Tuple[IntentCategory, float]:
    """Force a non-login 500 response into technical_crash with confidence >= .8."""
    if "500" in (message or "") and intent != IntentCategory.TECHNICAL_LOGIN:
        return IntentCategory.TECHNICAL_CRASH, max(float(confidence), 0.8)
    return intent, float(confidence)


apply_status_code_override = apply_500_override
