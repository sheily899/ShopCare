from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import hashlib
import inspect
import logging
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


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
    intent_diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMIntentResult:
    """Internal result contract for the isolated LLM intent source."""

    intent: IntentCategory
    confidence: float
    reasoning: str
    success: bool
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        """Keep dict-style access convenient for the later fusion adapter."""
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


@dataclass
class EmbeddingIntentResult:
    """Internal result contract for the embedding intent source."""

    intent: IntentCategory
    score: float
    success: bool = True
    local_fallback: bool = False
    reasoning: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        return self.score

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


_INTENT_TOOL_NAME = "classify_intent"
CLASSIFY_INTENT_TOOL: Dict[str, Any] = {
    "name": _INTENT_TOOL_NAME,
    "description": "用于对电商客户会话进行精准意图分类，并返回判定理由与置信度。",
    "input_schema": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": [category.value for category in IntentCategory],
            },
            "confidence": {
                "type": "number",
                "description": "模型对意图分类结果的置信度，范围为 0 到 1。",
            },
            "reasoning": {
                "type": "string",
                "description": "简要陈述判定该意图的理由。",
            },
        },
        "required": ["intent", "confidence", "reasoning"],
        "additionalProperties": False,
    },
}

# Keep the source-oriented name available for the later fusion implementation.
_INTENT_TOOL = CLASSIFY_INTENT_TOOL
DEFAULT_LLM_MODEL = "claude-3-5-sonnet-20241022"
_LLM_SYSTEM_PROMPT = """你是 ShopCare 的电商客服意图分析器。
请根据用户当前消息和可选的最近会话上下文，选择最具体的一个意图。
必须调用 classify_intent 工具，不要输出自由文本 JSON。
intent 必须使用工具 schema 中的枚举值，confidence 必须是数字，reasoning 说明最直接的判定依据。
退款、发票、登录故障、崩溃、物流和人工请求优先使用对应的细粒度意图，不要用宽泛类别替代。"""


def _build_llm_user_message(
    message: str,
    history: Optional[List[Any]],
) -> str:
    """Build the user-side prompt while keeping history optional and bounded."""
    current = message or ""
    if not history:
        return current

    history_lines: List[str] = []
    for item in history[-3:]:
        if isinstance(item, dict):
            role = str(item.get("role", "user"))
            content = str(item.get("content", ""))
        else:
            role = str(getattr(item, "role", "user"))
            content = str(getattr(item, "content", ""))
        if content.strip():
            history_lines.append(f"{role}: {content.strip()}")
    if not history_lines:
        return current
    return f"最近会话：\n{chr(10).join(history_lines)}\n\n当前用户消息：\n{current}"


def _extract_intent_tool_input(content: Any) -> Optional[Dict[str, Any]]:
    """Extract the strict classify_intent tool payload from an Anthropic response."""
    for block in content or []:
        block_type = getattr(block, "type", None)
        name = getattr(block, "name", None)
        input_data = getattr(block, "input", None)
        if isinstance(block, dict):
            block_type = block.get("type", block_type)
            name = block.get("name", name)
            input_data = block.get("input", input_data)
        if (
            block_type == "tool_use"
            and name == _INTENT_TOOL_NAME
            and isinstance(input_data, dict)
        ):
            return input_data
    return None


def _failed_llm_result(reason: str) -> LLMIntentResult:
    return LLMIntentResult(
        intent=IntentCategory.OTHER,
        confidence=0.0,
        reasoning=f"LLM 失败: {reason}",
        success=False,
        diagnostics={"llm_failed": True, "error": reason},
    )


def _get_default_anthropic_client() -> Any:
    """Create the SDK client lazily so pure pattern tests need no SDK import."""
    try:
        from anthropic import AsyncAnthropic
    except Exception as exc:  # pragma: no cover - depends on optional local install
        raise RuntimeError(f"Anthropic SDK 不可用: {exc}") from exc
    return AsyncAnthropic()


async def _recognize_with_llm(
    message: str,
    history: Optional[List[Any]] = None,
    client: Any = None,
    model: Optional[str] = None,
) -> LLMIntentResult:
    """Classify one message through Anthropic Tool Use with safe fallback semantics."""
    try:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("待分析消息不能为空")

        anthropic_client = client or _get_default_anthropic_client()
        selected_model = model or os.getenv("ANTHROPIC_MODEL", DEFAULT_LLM_MODEL)
        response = await anthropic_client.messages.create(
            model=selected_model,
            max_tokens=256,
            temperature=0.1,
            system=_LLM_SYSTEM_PROMPT,
            tools=[CLASSIFY_INTENT_TOOL],
            tool_choice={"type": "tool", "name": _INTENT_TOOL_NAME},
            messages=[
                {
                    "role": "user",
                    "content": _build_llm_user_message(message, history),
                }
            ],
        )

        payload = _extract_intent_tool_input(getattr(response, "content", None))
        if payload is None:
            raise ValueError(f"未收到有效的 {_INTENT_TOOL_NAME} 工具调用")

        expected_fields = {"intent", "confidence", "reasoning"}
        payload_fields = set(payload)
        missing = expected_fields - payload_fields
        extra = payload_fields - expected_fields
        if missing:
            raise ValueError(f"工具返回缺少必填字段: {sorted(missing)}")
        if extra:
            raise ValueError(f"工具返回包含未定义字段: {sorted(extra)}")

        raw_intent = payload["intent"]
        if not isinstance(raw_intent, str):
            raise ValueError("工具返回的 intent 不是字符串")
        try:
            intent = IntentCategory(raw_intent)
        except ValueError as exc:
            raise ValueError(f"工具返回了未注册的 intent: {raw_intent!r}") from exc

        raw_confidence = payload["confidence"]
        if isinstance(raw_confidence, bool) or not isinstance(
            raw_confidence, (int, float)
        ):
            raise ValueError("工具返回的 confidence 不是数字")
        numeric_confidence = float(raw_confidence)
        if not math.isfinite(numeric_confidence):
            raise ValueError("工具返回的 confidence 不是有限数字")
        confidence = min(1.0, max(0.0, numeric_confidence))

        reasoning = payload["reasoning"]
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise ValueError("工具返回的 reasoning 为空或不是字符串")

        diagnostics: Dict[str, Any] = {}
        if confidence != numeric_confidence:
            diagnostics["confidence_clamped"] = True
        return LLMIntentResult(
            intent=intent,
            confidence=confidence,
            reasoning=reasoning.strip(),
            success=True,
            diagnostics=diagnostics,
        )
    except Exception as exc:
        logger.warning("LLM 意图识别失败: %s", exc)
        return _failed_llm_result(str(exc))


HASH_EMBEDDING_DIM = 256
EMBEDDING_MODEL = "voyage-3-lite"
EMBEDDING_TEMPLATES: Dict[IntentCategory, Tuple[str, ...]] = {
    IntentCategory.QUERY: ("我的订单状态是什么？", "如何重置密码？", "快递什么时候到？"),
    IntentCategory.COMPLAINT: ("等了好几个小时！", "服务太差了！", "一直没人处理！"),
    IntentCategory.REQUEST: ("帮我取消订单", "我需要修改地址", "请协助退款"),
    IntentCategory.GREETING: ("你好", "嗨，有人吗", "早上好"),
    IntentCategory.TECHNICAL: ("应用一直崩溃", "无法登录", "出现500错误"),
    IntentCategory.BILLING: ("为什么扣了两次款？", "申请退款", "发票问题"),
    IntentCategory.ACCOUNT: ("修改邮箱", "注销账户", "更新个人信息"),
    IntentCategory.FEEDBACK: ("服务很棒！", "非常满意", "给个好评"),
    IntentCategory.ORDER_STATUS: (
        "我的订单现在是什么状态？",
        "订单有没有发货？",
        "订单处理到哪一步了？",
    ),
    IntentCategory.LOGISTICS: ("快递什么时候到？", "物流一直不更新", "配送要多久？"),
    IntentCategory.REFUND: ("我要申请退款", "退货退款怎么处理？", "退款多久到账？"),
    IntentCategory.INVOICE: ("帮我开发票", "发票抬头怎么改？", "电子发票在哪里？"),
    IntentCategory.PAYMENT_ISSUE: (
        "为什么重复扣款？",
        "支付失败怎么办？",
        "这个月多扣了钱",
    ),
    IntentCategory.ACCOUNT_SECURITY: ("账户被盗了", "发现异常登录", "我要重置密码"),
    IntentCategory.TECHNICAL_LOGIN: ("登录一直报401", "验证码收不到", "无法登录账号"),
    IntentCategory.TECHNICAL_CRASH: ("应用一直崩溃", "页面报500错误", "系统闪退"),
    IntentCategory.HUMAN_HANDOFF: ("转人工客服", "我要找人工", "请升级处理"),
    IntentCategory.OTHER: ("其他问题", "不清楚", "不知道"),
}


def _hash_ngram_vector(text: str) -> List[float]:
    """Build the fixed 256-dimension local 1/2/3-gram MD5 vector."""
    normalized = (text or "").casefold().strip()
    vector = [0.0] * HASH_EMBEDDING_DIM
    tokens = set()
    for ngram_size in (1, 2, 3):
        if len(normalized) >= ngram_size:
            tokens.update(
                normalized[index : index + ngram_size]
                for index in range(len(normalized) - ngram_size + 1)
            )
    if not tokens:
        tokens.add(normalized)
    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % HASH_EMBEDDING_DIM
        vector[bucket] += 1.0 if digest[4] % 2 == 0 else -1.0
    return vector


def _cosine_similarity(left: List[float], right: List[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if denominator == 0.0:
        return 0.0
    score = sum(a * b for a, b in zip(left, right)) / denominator
    return min(1.0, max(0.0, float(score)))


def _parse_remote_embeddings(payload: Any) -> Optional[List[List[float]]]:
    if hasattr(payload, "json") and callable(payload.json):
        payload = payload.json()
    if isinstance(payload, dict):
        payload = payload.get("data", payload.get("embeddings", payload.get("embedding")))
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return None
    if payload and isinstance(payload[0], dict):
        payload = [item.get("embedding") for item in payload]
    if not all(isinstance(vector, list) for vector in payload):
        return None
    try:
        return [[float(value) for value in vector] for vector in payload]
    except (TypeError, ValueError):
        return None


async def _fetch_remote_embeddings(
    texts: List[str],
    remote_client: Any = None,
) -> Optional[List[List[float]]]:
    """Try an optional compatible embedding endpoint; callers always have local fallback."""
    url = os.getenv("EMBEDDING_API_URL", "").strip()
    if not url:
        return None
    payload = {"model": os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL), "input": texts}
    headers = {}
    api_key = os.getenv("EMBEDDING_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        if remote_client is not None:
            if callable(remote_client):
                response = remote_client(texts)
            else:
                response = remote_client.post(
                    url, json=payload, headers=headers, timeout=5.0
                )
            if inspect.isawaitable(response):
                response = await response
            if hasattr(response, "raise_for_status"):
                response.raise_for_status()
            return _parse_remote_embeddings(response)

        try:
            import httpx
        except Exception:
            return None
        async with httpx.AsyncClient(timeout=5.0) as http_client:
            response = await http_client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return _parse_remote_embeddings(response)
    except Exception as exc:
        logger.warning("远程 Embedding 失败，使用本地哈希向量: %s", exc)
        return None


async def _recognize_with_embedding(
    message: str,
    remote_client: Any = None,
) -> EmbeddingIntentResult:
    """Return the best template match, preferring remote vectors and degrading locally."""
    template_items = list(EMBEDDING_TEMPLATES.items())
    flattened_templates = [
        (intent, text)
        for intent, texts in template_items
        for text in texts
    ]
    template_texts = [text for _, text in flattened_templates]
    all_texts = template_texts + [message or ""]
    remote_vectors = await _fetch_remote_embeddings(all_texts, remote_client)
    local_fallback = not remote_vectors or len(remote_vectors) != len(all_texts)
    vectors = (
        [_hash_ngram_vector(text) for text in all_texts]
        if local_fallback
        else remote_vectors
    )

    query_vector = vectors[-1]
    template_scores = [
        _cosine_similarity(template_vector, query_vector)
        for template_vector in vectors[:-1]
    ]
    intent_scores: Dict[IntentCategory, float] = {}
    for (intent, _), score in zip(flattened_templates, template_scores):
        intent_scores[intent] = max(score, intent_scores.get(intent, 0.0))
    if not intent_scores or max(intent_scores.values()) <= 0.0:
        best_intent = IntentCategory.OTHER
        best_score = 0.0
    else:
        best_intent, best_score = max(intent_scores.items(), key=lambda item: item[1])
    return EmbeddingIntentResult(
        intent=best_intent,
        score=min(1.0, max(0.0, float(best_score))),
        success=True,
        local_fallback=local_fallback,
        reasoning="本地 1/2/3-gram MD5 向量" if local_fallback else "远程 Embedding 向量",
        diagnostics={"embedding_model": os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL)},
    )


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


_INTENT_GROUPS: Dict[IntentCategory, str] = {
    IntentCategory.ORDER_STATUS: "query",
    IntentCategory.LOGISTICS: "query",
    IntentCategory.REFUND: "billing",
    IntentCategory.INVOICE: "billing",
    IntentCategory.PAYMENT_ISSUE: "billing",
    IntentCategory.ACCOUNT_SECURITY: "account",
    IntentCategory.TECHNICAL_LOGIN: "technical",
    IntentCategory.TECHNICAL_CRASH: "technical",
}


def _coerce_intent(value: Any) -> IntentCategory:
    if isinstance(value, IntentCategory):
        return value
    try:
        return IntentCategory(str(value))
    except (TypeError, ValueError):
        return IntentCategory.OTHER


def _bounded_score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    numeric = float(value)
    if not math.isfinite(numeric):
        return 0.0
    return min(1.0, max(0.0, numeric))


def _normalize_llm_source(value: Any) -> LLMIntentResult:
    if isinstance(value, LLMIntentResult):
        return value
    if isinstance(value, dict):
        intent = _coerce_intent(value.get("intent"))
        failed = bool(value.get("failed", False))
        success = bool(value.get("success", not failed))
        return LLMIntentResult(
            intent=intent,
            confidence=_bounded_score(value.get("confidence", 0.0)),
            reasoning=str(value.get("reasoning", "")),
            success=success,
            diagnostics=dict(value.get("diagnostics", {})),
        )
    return _failed_llm_result("LLM 来源未返回结构化结果")


def _normalize_embedding_source(value: Any) -> EmbeddingIntentResult:
    if isinstance(value, EmbeddingIntentResult):
        return value
    if isinstance(value, dict):
        intent = _coerce_intent(value.get("intent"))
        score = _bounded_score(value.get("score", value.get("confidence", 0.0)))
        explicit_success = value.get("success")
        success = (
            bool(explicit_success)
            if explicit_success is not None
            else bool(not value.get("failed", False) and (intent is not IntentCategory.OTHER or score > 0.0))
        )
        return EmbeddingIntentResult(
            intent=intent,
            score=score,
            success=success,
            local_fallback=bool(value.get("local_fallback", False)),
            reasoning=str(value.get("reasoning", "")),
            diagnostics=dict(value.get("diagnostics", {})),
        )
    return EmbeddingIntentResult(
        intent=IntentCategory.OTHER,
        score=0.0,
        success=False,
        local_fallback=False,
        reasoning="Embedding 来源未返回结构化结果",
    )


def _pattern_score_map(pattern_result: Dict[str, Any]) -> Dict[IntentCategory, float]:
    scores: Dict[IntentCategory, float] = {}
    for raw_intent, raw_score in dict(pattern_result.get("scores", {})).items():
        intent = _coerce_intent(raw_intent)
        score = _bounded_score(raw_score)
        if score > scores.get(intent, 0.0):
            scores[intent] = score
    best_intent = _coerce_intent(pattern_result.get("intent"))
    best_score = _bounded_score(pattern_result.get("confidence", 0.0))
    if best_intent not in scores and best_score > 0.0:
        scores[best_intent] = best_score
    return scores


def _intent_group(intent: IntentCategory) -> str:
    return _INTENT_GROUPS.get(intent, intent.value)


async def recognize_intent(
    message: str,
    history: Optional[List[Any]] = None,
) -> IntentResult:
    """Run LLM, embedding and pattern sources, then apply the fixed fusion state machine."""
    started_at = time.perf_counter()
    llm_task = asyncio.create_task(_recognize_with_llm(message, history))
    embedding_task = asyncio.create_task(_recognize_with_embedding(message))
    try:
        pattern_result = _pattern_recognize(message)
    except Exception as exc:
        logger.warning("Pattern 识别失败: %s", exc)
        pattern_result = {
            "intent": IntentCategory.OTHER,
            "confidence": 0.0,
            "scores": {},
            "specific_intent": IntentCategory.OTHER,
            "specific_score": 0.0,
        }

    llm_raw, embedding_raw = await asyncio.gather(
        llm_task,
        embedding_task,
        return_exceptions=True,
    )
    llm = (
        _failed_llm_result(f"来源异常: {llm_raw}")
        if isinstance(llm_raw, Exception)
        else _normalize_llm_source(llm_raw)
    )
    embedding = (
        EmbeddingIntentResult(
            intent=IntentCategory.OTHER,
            score=0.0,
            success=False,
            reasoning=f"来源异常: {embedding_raw}",
        )
        if isinstance(embedding_raw, Exception)
        else _normalize_embedding_source(embedding_raw)
    )
    pattern_intent = _coerce_intent(pattern_result.get("intent"))
    pattern_confidence = _bounded_score(pattern_result.get("confidence", 0.0))
    pattern_scores = _pattern_score_map(pattern_result)
    specific_intent = _coerce_intent(pattern_result.get("specific_intent"))
    specific_score = _bounded_score(pattern_result.get("specific_score", 0.0))

    source_intents = {
        "llm": llm.intent.value,
        "embedding": embedding.intent.value,
        "pattern": pattern_intent.value,
    }
    source_scores = {
        "llm": _bounded_score(llm.confidence),
        "embedding": _bounded_score(embedding.score),
        "pattern": pattern_confidence,
    }
    diagnostics: Dict[str, Any] = {
        "llm_success": llm.success,
        "embedding_success": embedding.success,
        "embedding_local_fallback": embedding.local_fallback,
        "pattern_specific_score": specific_score,
        "source_intents": dict(source_intents),
        "source_scores": dict(source_scores),
        "source_failures": {
            "llm": not llm.success,
            "embedding": not embedding.success,
        },
        "weights": {
            "llm": 0.0,
            "embedding": 1.0 if embedding.success else 0.0,
            "pattern": 0.0,
        },
        "fallback": "",
    }

    final_intent = IntentCategory.OTHER
    final_confidence = 0.0
    if not llm.success:
        if specific_score > 0.0 and specific_intent is not IntentCategory.OTHER:
            final_intent = specific_intent
            final_confidence = specific_score
            diagnostics["fallback"] = "specific_pattern"
        elif embedding.success and embedding.score >= 0.45:
            final_intent = embedding.intent
            final_confidence = embedding.score
            diagnostics["fallback"] = "embedding"
        else:
            diagnostics["fallback"] = "llm_and_embedding_unavailable"
    else:
        if embedding.success:
            weights = {"llm": 0.7, "embedding": 0.2, "pattern": 0.1}
        else:
            weights = {"llm": 0.85, "embedding": 0.0, "pattern": 0.15}
        diagnostics["weights"] = weights
        fused_scores: Dict[IntentCategory, float] = {}
        fused_scores[llm.intent] = fused_scores.get(llm.intent, 0.0) + (
            weights["llm"] * llm.confidence
        )
        if embedding.success:
            fused_scores[embedding.intent] = fused_scores.get(embedding.intent, 0.0) + (
                weights["embedding"] * embedding.score
            )
        for intent, score in pattern_scores.items():
            fused_scores[intent] = fused_scores.get(intent, 0.0) + (
                weights["pattern"] * score
            )

        candidate, fused_confidence = max(
            fused_scores.items(), key=lambda item: item[1], default=(IntentCategory.OTHER, 0.0)
        )
        if fused_confidence < 0.5:
            diagnostics["fallback"] = "low_fused_score"
        else:
            final_intent = candidate
            final_confidence = fused_confidence
            if (
                candidate in _GENERIC_INTENTS
                and specific_intent is not IntentCategory.OTHER
                and specific_intent is not candidate
                and specific_score >= 0.5
                and fused_confidence < 0.8
            ):
                final_intent = specific_intent
                final_confidence = max(fused_confidence, specific_score)
                diagnostics["refined_from"] = candidate.value

    final_intent, final_confidence = apply_500_override(
        message,
        final_intent,
        final_confidence,
    )
    diagnostics["status_code_override"] = final_intent is IntentCategory.TECHNICAL_CRASH and "500" in (message or "")
    diagnostics["final_intent"] = final_intent.value
    normalized_confidence = _bounded_score(round(final_confidence, 6))
    diagnostics["final_confidence"] = normalized_confidence

    if llm.success and llm.reasoning:
        reasoning = llm.reasoning
    elif diagnostics["fallback"] == "embedding":
        reasoning = "LLM 失败，采用 Embedding 结果"
    elif diagnostics["fallback"] == "specific_pattern":
        reasoning = "LLM 失败，采用特异 Pattern 结果"
    elif final_intent is IntentCategory.OTHER:
        reasoning = "未达到意图识别阈值"
    else:
        reasoning = "采用融合后的意图结果"

    return IntentResult(
        intent=final_intent,
        confidence=normalized_confidence,
        urgency=calculate_urgency(message, final_intent),
        intent_group=_intent_group(final_intent),
        entities=extract_entities(message),
        reasoning=reasoning,
        latency_ms=max(0.0, round((time.perf_counter() - started_at) * 1000, 3)),
        source_scores=source_scores,
        source_intents=source_intents,
        intent_diagnostics=diagnostics,
    )
