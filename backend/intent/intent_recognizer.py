"""ShopCare 意图识别与复合路由核心模块。

本模块负责内部意图枚举、dataclass 结果契约、Pattern 与实体规则、可选的
Anthropic/Embedding 适配器和确定性的融合状态机；不执行领域 Agent，不修改
订单，也不产生真实转人工副作用。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum, IntEnum
import hashlib
import logging
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

from intent.beauty_intent_registry import (
    BEAUTY_INTENT_REGISTRY as REGISTRY_INTENTS,
    EMBEDDING_TEMPLATES as REGISTRY_EMBEDDING_TEMPLATES,
    LLM_INTENT_SCHEMA,
    PATTERN_MAPS,
)
from handoff.config import load_handoff_urgency_config
from handoff.urgency import is_explicit_handoff_requested, score_temporary_urgency


logger = logging.getLogger(__name__)


def _usage_from_llm_response(response: Any) -> Dict[str, int]:
    """读取兼容 Anthropic 客户端返回的 token usage。"""

    usage = response.get("usage") if isinstance(response, Mapping) else getattr(response, "usage", None)
    if usage is None:
        return {}
    values: Dict[str, int] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        raw = usage.get(key) if isinstance(usage, Mapping) else getattr(usage, key, None)
        try:
            number = int(raw)
        except (TypeError, ValueError):
            continue
        if number >= 0:
            values[key] = number
    if "total_tokens" not in values and {"input_tokens", "output_tokens"}.issubset(values):
        values["total_tokens"] = values["input_tokens"] + values["output_tokens"]
    return values


class IntentCategory(str, Enum):
    """所有识别通路都必须使用的封闭意图枚举集合。"""

    INGREDIENT_FIT = "ingredient_fit"
    SHADE_MATCH = "shade_match"
    USAGE_HOWTO = "usage_howto"
    PRODUCT_INFO = "product_info"
    PROMO_RULES = "promo_rules"
    PREGNANCY_SAFETY = "pregnancy_safety"
    ORDER_STATUS_BEAUTY = "order_status"
    LOGISTICS_STALLED = "logistics_stalled"
    SIGNED_NOT_RECEIVED = "signed_not_received"
    INTERCEPT_CHANGE_ADDRESS = "intercept_change_address"
    MISSING_ITEM = "missing_item"
    WRONG_ITEM = "wrong_item"
    DAMAGED_ON_ARRIVAL = "damaged_on_arrival"
    RETURN_REFUND = "return_refund"
    ADVERSE_REACTION = "adverse_reaction"
    SUSPECTED_ABNORMAL_REFUND = "suspected_abnormal_refund"
    PRICE_PROTECTION = "price_protection"
    REFUND_NOT_RECEIVED = "refund_not_received"
    SHIPPING_FEE_REIMBURSEMENT = "shipping_fee_reimbursement"
    ORDER_STATUS = ORDER_STATUS_BEAUTY
    HUMAN_HANDOFF = "human_handoff"
    ESCALATION = HUMAN_HANDOFF
    OTHER = "other"
    # 仅用于读取历史评测数据，不进入新的 LLM Schema、Pattern 或路由候选。
    LEGACY_QUERY = "query"
    LEGACY_COMPLAINT = "complaint"
    LEGACY_REQUEST = "request"
    LEGACY_GREETING = "greeting"
    LEGACY_TECHNICAL = "technical"
    LEGACY_BILLING = "billing"
    LEGACY_ACCOUNT = "account"
    LEGACY_FEEDBACK = "feedback"
    LEGACY_LOGISTICS = "logistics"
    LEGACY_REFUND = "refund"
    LEGACY_INVOICE = "invoice"
    LEGACY_PAYMENT_ISSUE = "payment_issue"
    LEGACY_ACCOUNT_SECURITY = "account_security"
    LEGACY_TECHNICAL_LOGIN = "technical_login"
    LEGACY_TECHNICAL_CRASH = "technical_crash"
    QUERY = LEGACY_QUERY
    COMPLAINT = LEGACY_COMPLAINT
    REQUEST = LEGACY_REQUEST
    GREETING = LEGACY_GREETING
    TECHNICAL = LEGACY_TECHNICAL
    BILLING = LEGACY_BILLING
    ACCOUNT = LEGACY_ACCOUNT
    FEEDBACK = LEGACY_FEEDBACK
    LOGISTICS = LEGACY_LOGISTICS
    REFUND = LEGACY_REFUND
    INVOICE = LEGACY_INVOICE
    PAYMENT_ISSUE = LEGACY_PAYMENT_ISSUE
    ACCOUNT_SECURITY = LEGACY_ACCOUNT_SECURITY
    TECHNICAL_LOGIN = LEGACY_TECHNICAL_LOGIN
    TECHNICAL_CRASH = LEGACY_TECHNICAL_CRASH


class UrgencyLevel(IntEnum):
    """用于路由与安全仲裁的有序紧急度等级。"""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class IntentResult:
    """意图识别流水线最终产出的统一内部结果。

    Args:
        intent: 融合与覆盖后的最终主意图。
        confidence: 最终意图的归一化置信度。
        urgency: 请求的确定性紧急度。
        intent_group: 供下游路由使用的粗粒度业务分组。
        entities: 五键实体抽取结果。
        reasoning: 面向维护与诊断的判定理由。
        latency_ms: 识别耗时，单位为毫秒。
        source_scores: 各识别来源的置信度诊断。
        source_intents: 各识别来源的意图诊断。
        secondary_intents: 合法的跨域辅助意图。
        compound_trigger: 触发复合分析的连接词。
        intent_diagnostics: 详细状态机诊断信息。
    """

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
    sub_requests: List[Dict[str, Any]] = field(default_factory=list)
    emotion: str = "neutral"
    emotion_intensity: float = 0.0
    emotion_evidence: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class CompoundSubRequest:
    """一个可独立路由、检索和回答的业务子请求。"""

    id: str
    text: str
    domain: str
    intent: IntentCategory
    confidence: float
    entities: Dict[str, List[str]] = field(default_factory=dict)
    knowledge_required: bool = True


@dataclass
class LLMIntentResult:
    """隔离的 LLM 意图来源内部结果契约。

    Args:
        intent: Tool Use 返回的意图，失败时为 ``OTHER``。
        confidence: 截断到 0..1 的 Tool Use 置信度。
        reasoning: 模型解释或确定性的失败说明。
        success: LLM 来源是否产出合法结果。
        diagnostics: 该来源的校验与失败元数据。
    """

    intent: IntentCategory
    confidence: float
    reasoning: str
    success: bool
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        """按兼容旧调用方的映射方式读取结果字段。

        Args:
            key: dataclass 属性名。

        Returns:
            对应属性中的值。
        """
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        """读取结果字段，未知字段时返回默认值。

        Args:
            key: dataclass 属性名。
            default: 属性不存在时返回的值。

        Returns:
            属性值或 ``default``。
        """
        return getattr(self, key, default)


@dataclass
class EmbeddingIntentResult:
    """Embedding 意图来源的内部结果契约。

    Args:
        intent: 模板相似度最高的意图。
        score: 归一化到 0..1 的余弦相似度。
        success: 是否产出可用的 Embedding 结果。
        local_fallback: 是否使用固定本地哈希向量。
        reasoning: 向量来源说明。
        diagnostics: Embedding 来源元数据。
    """

    intent: IntentCategory
    score: float
    success: bool = True
    local_fallback: bool = False
    reasoning: str = ""
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        """通过统一的来源置信度名称暴露相似度分数。

        Returns:
            Embedding 相似度分数。
        """
        return self.score

    def __getitem__(self, key: str) -> Any:
        """按映射兼容方式读取结果字段。

        Args:
            key: dataclass 属性名。

        Returns:
            对应属性中的值。
        """
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        """读取结果字段并为未知字段提供默认值。

        Args:
            key: dataclass 属性名。
            default: 属性不存在时返回的值。

        Returns:
            属性值或 ``default``。
        """
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
                "enum": [
                    *LLM_INTENT_SCHEMA["enum"],
                    IntentCategory.HUMAN_HANDOFF.value,
                    IntentCategory.OTHER.value,
                ],
                "description": "只允许使用 ShopCare 美妆意图注册表中的 19 个意图，以及 human_handoff/other 两个控制结果。",
            },
            "confidence": {
                "type": "number",
                "description": "模型对意图分类结果的置信度，范围为 0 到 1。",
            },
            "reasoning": {
                "type": "string",
                "description": "简要陈述判定该意图的理由。",
            },
            "is_compound": {
                "type": "boolean",
                "description": "当前消息是否包含两个或以上可独立处理的业务子请求。",
            },
            "sub_requests": {
                "type": "array",
                "description": "复合消息的业务子请求，每个子请求只属于一个业务域。",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "text": {"type": "string"},
                        "domain": {"type": "string", "enum": ["consult", "order", "after_sales"]},
                        "intent": LLM_INTENT_SCHEMA,
                        "confidence": {"type": "number"},
                    },
                    "required": ["id", "text", "domain", "intent", "confidence"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["intent", "confidence", "reasoning"],
        "additionalProperties": False,
    },
}

# 保留来源导向的别名，供后续融合实现继续使用。
_INTENT_TOOL = CLASSIFY_INTENT_TOOL
DEFAULT_LLM_MODEL = "claude-3-5-sonnet-20241022"
_INTENT_BOUNDARY_GUIDE = "\n".join(
    f"- {item['intent_id']}（{item['name_zh']}）：{item['description']} 区分：{item['distinction']} 边界：{item['boundary']}"
    for item in REGISTRY_INTENTS
)

_LLM_SYSTEM_PROMPT = f"""你是 ShopCare 的美妆电商客服意图分析器。
只根据用户当前消息识别用户此刻明确表达的需求；历史消息仅用于消歧，不把历史诉求当作当前诉求。
只能选择工具 schema 中的 19 个美妆业务意图，或 human_handoff / other 控制结果，不使用其他行业的旧意图。
必须区分肯定、否定和转折：被明确否定的动作不是请求；例如“我不要退款，只换货”不能判为退货退款，也不能在没有错发事实时判为错发；换货原因不明时判为 other 并等待澄清。
如果消息包含多个可独立处理的业务问题，请返回 is_compound=true 和 sub_requests；每个子请求拆成独立文本，并只归属 consult、order、after_sales 三个业务域之一。
必须调用 classify_intent 工具，不要输出自由文本 JSON。confidence 必须是数字，reasoning 说明当前消息中支持判定的依据。
以下定义和边界均来自唯一意图注册表：
{_INTENT_BOUNDARY_GUIDE}"""


def _build_llm_user_message(
    message: str,
    history: Optional[List[Any]],
) -> str:
    """构造发送给 LLM 分类器的有界用户提示词。

    Args:
        message: 当前用户消息。
        history: 可选的历史消息对象或字典列表。

    Returns:
        最多包含最近三条历史消息的提示词。
    """
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
    """从 LLM 响应中提取一个严格的 ``classify_intent`` 工具参数。

    Args:
        content: Anthropic 内容块，可以是对象或字典。

    Returns:
        工具输入字典；没有合法工具块时返回 ``None``。
    """
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
    """创建 LLM 失败路径使用的确定性 ``OTHER`` 结果。

    Args:
        reason: 保存在诊断结果中的失败详情。

    Returns:
        置信度为零的失败 ``LLMIntentResult``。
    """
    return LLMIntentResult(
        intent=IntentCategory.OTHER,
        confidence=0.0,
        reasoning=f"LLM 失败: {reason}",
        success=False,
        diagnostics={"llm_failed": True, "error": reason},
    )


def _get_default_anthropic_client() -> Any:
    """为未注入客户端的调用方惰性创建 Anthropic 客户端。

    Returns:
        配置好的 ``AsyncAnthropic`` 客户端。

    Raises:
        RuntimeError: 无法导入可选 SDK 时抛出。
    """
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
    """通过 Anthropic Tool Use 分类单条消息，并提供安全降级语义。

    Args:
        message: 待分类的当前用户消息。
        history: 仅用于提示词的可选会话上下文。
        client: 可选的 Anthropic 兼容注入客户端。
        model: 可选模型覆盖值。

    Returns:
        校验后的 ``LLMIntentResult``；失败时返回 ``OTHER``，不向编排层冒泡。
    """
    try:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("待分析消息不能为空")

        anthropic_client = client or _get_default_anthropic_client()
        selected_model = model or os.getenv("ANTHROPIC_MODEL", DEFAULT_LLM_MODEL)
        response = await anthropic_client.messages.create(
            model=selected_model,
            max_tokens=256,
            temperature=0.1,
            # Why：部分兼容 Anthropic 的模型默认开启 Thinking，但 Thinking 与强制 Tool Use 冲突，
            # 显式关闭才能稳定返回结构化意图结果。
            extra_body={"thinking": {"type": "disabled"}},
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
        allowed_fields = expected_fields | {"is_compound", "sub_requests"}
        payload_fields = set(payload)
        missing = expected_fields - payload_fields
        extra = payload_fields - allowed_fields
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
        usage = _usage_from_llm_response(response)
        if usage:
            diagnostics["usage"] = usage
        if isinstance(payload.get("sub_requests"), list):
            diagnostics["llm_sub_requests"] = payload["sub_requests"]
            diagnostics["llm_is_compound"] = bool(payload.get("is_compound"))
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
# 意图识别与知识库召回共用同一套本地模型配置。
# 保留 ``EMBEDDING_MODEL`` 这个变量名作为兼容别名，避免外部调用方断裂。
EMBEDDING_MODEL = os.getenv(
    "CUSTOMEROPS_EMBEDDING_MODEL",
    "BAAI/bge-small-zh-v1.5",
)
EMBEDDING_TEMPLATES: Dict[IntentCategory, Tuple[str, ...]] = {
    **{IntentCategory(intent_id): examples for intent_id, examples in REGISTRY_EMBEDDING_TEMPLATES.items()},
    IntentCategory.HUMAN_HANDOFF: ("转人工客服", "我要找人工", "请升级处理"),
    IntentCategory.OTHER: ("其他问题", "不清楚", "不知道"),
}


def _hash_ngram_vector(text: str) -> List[float]:
    """构造固定 256 维的本地 1/2/3-gram MD5 向量。

    Args:
        text: 待归一化并哈希的输入文本。

    Returns:
        确定性的 256 维有符号哈希向量。
    """
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
    """计算两个等长向量的有界余弦相似度。

    Args:
        left: 第一个数值向量。
        right: 第二个数值向量。

    Returns:
        截断在 0..1 的相似度；向量非法时返回零。
    """
    if not left or not right or len(left) != len(right):
        return 0.0
    denominator = math.sqrt(sum(value * value for value in left)) * math.sqrt(
        sum(value * value for value in right)
    )
    if denominator == 0.0:
        return 0.0
    score = sum(a * b for a, b in zip(left, right)) / denominator
    return min(1.0, max(0.0, float(score)))


_LOCAL_EMBEDDING_FUNCTION: Any = None
_LOCAL_EMBEDDING_INIT_FAILED = False


def _get_local_embedding_function() -> Any:
    """懒加载与知识库相同的 BGE embedding function。

    模型只在首次意图识别时初始化一次；初始化失败交给上层使用现有哈希降级。
    """
    global _LOCAL_EMBEDDING_FUNCTION, _LOCAL_EMBEDDING_INIT_FAILED
    if _LOCAL_EMBEDDING_FUNCTION is not None or _LOCAL_EMBEDDING_INIT_FAILED:
        return _LOCAL_EMBEDDING_FUNCTION
    model_name = os.getenv("CUSTOMEROPS_EMBEDDING_MODEL", EMBEDDING_MODEL).strip()
    # 不在本地时不要触发隐式下载；保持现有哈希降级的快速、离线行为。
    model_path = Path(model_name)
    project_model = Path(__file__).resolve().parents[1] / "models" / model_name
    if not model_path.exists() and not project_model.is_dir():
        _LOCAL_EMBEDDING_INIT_FAILED = True
        logger.info("本地 BGE 模型未发现，使用哈希向量降级: %s", model_name)
        return None
    try:
        from mcp.knowledge_base import build_embedding_function

        _LOCAL_EMBEDDING_FUNCTION = build_embedding_function(model_name)
    except Exception as exc:
        _LOCAL_EMBEDDING_INIT_FAILED = True
        logger.warning("本地 BGE Embedding 初始化失败，使用哈希向量降级: %s", exc)
    return _LOCAL_EMBEDDING_FUNCTION


def _encode_local_embeddings(texts: List[str]) -> Optional[List[List[float]]]:
    """使用知识库同款 BGE 编码文本；返回非法结果时交给降级链路。"""
    embedding_function = _get_local_embedding_function()
    if embedding_function is None:
        return None
    try:
        if hasattr(embedding_function, "embed_documents"):
            vectors = embedding_function.embed_documents(texts)
        elif callable(embedding_function):
            vectors = embedding_function(texts)
        else:
            return None
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            return None
        if not all(isinstance(vector, (list, tuple)) for vector in vectors):
            return None
        return [[float(value) for value in vector] for vector in vectors]
    except Exception as exc:
        logger.warning("本地 BGE Embedding 编码失败，使用哈希向量降级: %s", exc)
        return None


async def _recognize_with_embedding(
    message: str,
) -> EmbeddingIntentResult:
    """返回最佳模板匹配，优先本地 BGE，失败时保留哈希降级。

    Args:
        message: 与意图模板库比较的用户消息。
    Returns:
        相似度最高的 ``EmbeddingIntentResult``。
    """
    template_items = list(EMBEDDING_TEMPLATES.items())
    flattened_templates = [
        (intent, text)
        for intent, texts in template_items
        for text in texts
    ]
    template_texts = [text for _, text in flattened_templates]
    all_texts = template_texts + [message or ""]
    vectors = await asyncio.to_thread(_encode_local_embeddings, all_texts)
    vector_source = "local_bge"
    local_fallback = not vectors or len(vectors) != len(all_texts)
    if local_fallback:
        vectors = [_hash_ngram_vector(text) for text in all_texts]
        vector_source = "hash_ngram"

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
        reasoning={
            "local_bge": "本地 BGE 向量",
            "hash_ngram": "本地 1/2/3-gram MD5 向量",
        }[vector_source],
        diagnostics={
            "embedding_model": os.getenv("CUSTOMEROPS_EMBEDDING_MODEL", EMBEDDING_MODEL),
            "embedding_source": vector_source,
        },
    )


# 分数相同时，特异规则优先于泛化业务域标签。
# 规则值是正则片段；ASCII 词使用显式单词边界，避免子串命中造成误判。
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
        "支付状态",
        "支付结果",
        "支付处理中",
        "处于处理中",
        "处理中",
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
    IntentCategory.HUMAN_HANDOFF: tuple(
        load_handoff_urgency_config()["handoff_policy"]["explicit_request_phrases"]
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

# 新注册表规则覆盖同名旧意图；旧规则仅保留兼容未迁移的历史输入。
for _intent_id, _rules in PATTERN_MAPS.items():
    _intent = IntentCategory(_intent_id)
    SPECIFIC_PATTERNS[_intent] = tuple(_rules["exact"])
    GENERIC_PATTERNS[_intent] = tuple(_rules["generic"])

# 旧 Pattern 定义只保留源码历史痕迹，不进入当前运行候选集合。
_ACTIVE_INTENT_IDS = set(LLM_INTENT_SCHEMA["enum"]) | {IntentCategory.HUMAN_HANDOFF.value, IntentCategory.OTHER.value}
SPECIFIC_PATTERNS = {key: value for key, value in SPECIFIC_PATTERNS.items() if key.value in _ACTIVE_INTENT_IDS}
GENERIC_PATTERNS = {key: value for key, value in GENERIC_PATTERNS.items() if key.value in _ACTIVE_INTENT_IDS}

_SPECIFIC_PATTERNS = SPECIFIC_PATTERNS
_GENERIC_PATTERNS = GENERIC_PATTERNS
_SPECIFIC_INTENTS = frozenset(SPECIFIC_PATTERNS)
_GENERIC_INTENTS = frozenset(GENERIC_PATTERNS)

_ENTITY_KEYS = (
    "order_id", "product_name", "sku", "shade", "quantity", "symptom",
    "symptom_area", "onset_time", "stopped_use", "medical_visit", "missing_item",
    "wrong_item", "damaged_area", "tracking_no", "return_reason", "opened_or_used",
    "refund_amount", "price_difference",
)
_ORDER_ID_RE = re.compile(
    # 兼容消费者常说的“订单 SC-...”以及“订单号/订单编号 SC-...”，
    # 但不把普通数字或正文中的“订单”误当成订单号。
    r"(?:订单(?:号|编号)?|order(?:[_\s-]?id)?|order\s+(?:number|no)|#)"
    r"\s*[:：#]?\s*([A-Za-z0-9][A-Za-z0-9_-]{3,31})",
    re.IGNORECASE,
)
_MONEY_RE = re.compile(
    r"(?:[¥￥]\s*\d+(?:\.\d{1,2})?|"
    r"\d+(?:\.\d{1,2})?\s*(?:元|块|人民币|rmb|cny|usd|美元)|"
    r"(?:rmb|cny|usd)\s*\d+(?:\.\d{1,2})?)",
    re.IGNORECASE,
)


def _pattern_matches(pattern: str, message: str) -> bool:
    """判断一个正则或字面规则是否命中消息。

    Args:
        pattern: 规则表中的正则片段。
        message: 归一化后的用户消息。

    Returns:
        规则存在时返回 ``True``。
    """
    matches = list(re.finditer(pattern, message, flags=re.IGNORECASE))
    if not matches:
        return False
    # 先处理局部否定，避免“我不要退款，只换货”被退款关键词误路由。
    negation = re.compile(r"(?:不|不要|无需|不是|并非|没|没有|不想)\s*(?:去|申请|办理|要|想)?$")
    for match in matches:
        prefix = message[max(0, match.start() - 8):match.start()]
        if not negation.search(prefix):
            return True
    return False


def _pattern_score(hits: int) -> float:
    """将规则命中次数转换为固定的确定性 Pattern 分数。

    Args:
        hits: 某意图的规则命中次数。

    Returns:
        从 0.5 起步并封顶为 1.0 的分数。
    """
    if hits <= 0:
        return 0.0
    # 0.5 是单条规则的最低证据分；额外独立命中增加证据，但不允许重复关键词无限抬高置信度。
    return min(1.0, 0.5 + 0.25 * (hits - 1))


def match_patterns(message: str) -> Dict[str, Any]:
    """不调用外部服务，匹配特异规则与泛化规则。

    Args:
        message: 待与本地规则表匹配的用户消息。

    Returns:
        包含最佳意图、候选、分数、命中次数和最强特异候选的兼容字典。
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
    """保留原工程同步 Pattern 识别器的入口名称。

    Args:
        message: 待分类的用户消息。

    Returns:
        与 :func:`match_patterns` 相同的结果字典。
    """
    return match_patterns(message)


def _unique(values: List[str]) -> List[str]:
    """清理空白并保持实体首次出现顺序。

    Args:
        values: 原始实体值列表。

    Returns:
        去重后的非空实体值列表。
    """
    return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))


def extract_entities(message: str) -> Dict[str, List[str]]:
    """抽取美妆意图契约规定的业务实体字段。

    Args:
        message: 可能包含订单、商品、物流、使用不适或售后信息的用户消息。

    Returns:
        严格包含当前美妆注册表使用的实体 key，不返回旧技术或账单字段。
    """
    text = message or ""
    refund_requested = bool(re.search(r"退款|退货|退掉|退钱", text))
    use_markers = r"用了|使用了|用过|使用过|使用后|用后|连续用"
    onset_match = re.search(
        r"第一天|第[二三四五六七八九十\d]+天|用后\s*[一二两三四五六七八九十\d]+\s*[天小时分钟日]",
        text,
    )
    if onset_match:
        onset_time = [onset_match.group(0)]
    else:
        onset_time = _unique(re.findall(r"(用后\s*[一二两三四五六七八九十\d]+\s*[天小时分钟日])", text))

    stopped_use: List[str] = []
    if re.search(r"目前没有在使用|现在没有在使用|已经停用|已停用|停用了|停止使用|不再使用", text):
        stopped_use = ["已停用"]
    elif re.search(r"还在使用|仍在使用|还在用|仍在用|继续使用|继续用", text):
        stopped_use = ["还在使用"]

    main_item_text = re.sub(r"[^，。；！？\n]*(?:小样|赠品)[^，。；！？\n]*", "", text)
    opened_or_used: List[str] = []
    if re.search(use_markers, main_item_text):
        opened_or_used = ["已使用"]
    elif re.search(r"(?:已拆封未使用|拆封但没用|开封但没用|开封未使用|拆封未用|(?:拆封|开封).{0,6}(?:未使用|没有使用|没使用|没用|未用|尚未使用))", main_item_text):
        opened_or_used = ["已拆封未使用"]
    elif re.search(r"(?:正装|商品|产品)?(?:未拆封|没有拆封|没拆封|未拆|还没拆)", main_item_text):
        opened_or_used = ["未拆封"]
    elif re.search(r"(?:正装|商品|产品)?(?:已经拆封|已拆封|已经开封|已开封|开封了|拆开了)|(?:正装|商品|产品)(?:已|已经)?(?:拆封|开封)", main_item_text):
        opened_or_used = ["已拆封，使用情况不确定"]

    gift_opened_or_used: List[str] = []
    gift_text = " ".join(re.findall(r"[^，。；！？\n]*(?:小样|赠品)[^，。；！？\n]*", text))
    if gift_text and re.search(use_markers, gift_text):
        gift_opened_or_used = ["已使用"]
    elif gift_text and re.search(r"(?:拆封未使用|拆封但没用|开封未用)", gift_text):
        gift_opened_or_used = ["已拆封未使用"]
    elif gift_text and re.search(r"(?:未拆封|没拆封|没有拆封|没开封|没拆|未拆|还没拆)", gift_text):
        gift_opened_or_used = ["未拆封"]
    elif gift_text and re.search(r"(?:已经拆封|已拆封|已经开封|已开封|开封了|拆开了)", gift_text):
        gift_opened_or_used = ["已拆封，使用情况不确定"]

    medical_visit: List[str] = []
    if re.search(r"已就医|看过医生|去过医院|已经去医院", text):
        medical_visit = ["已就医"]
    elif re.search(r"没就医|未就医|没有就医|没去医院|没有去医院", text):
        medical_visit = ["没就医"]

    return_scope: List[str] = []
    if re.search(r"只退(?:正装|商品)|仅退(?:正装|商品)|正装单独退|小样不退|赠品不退", text):
        return_scope = ["只退正装"]
    elif re.search(r"只退(?:小样|赠品)|仅退(?:小样|赠品)", text):
        return_scope = ["只退赠品/小样"]
    elif re.search(r"分开退|分开处理|分别退款|正装和小样分开", text):
        return_scope = ["正装和赠品分开处理"]
    elif re.search(r"整单|全单|整套|全部退|一起退|商品和赠品(?:一起|都)退|正装和小样(?:一起|都)退", text):
        return_scope = ["整单退款"]

    return_reason: List[str] = []
    if refund_requested:
        if re.search(r"买错|买重|拍错|下错单", text):
            return_reason = ["买错/重复购买"]
        elif re.search(r"红肿|脸红|泛红|发红|起疹|刺痛|发痒|瘙痒|脱皮|过敏|肿胀|不舒服|不适", text):
            return_reason = ["使用不适"]
        elif re.search(r"不喜欢|不想要|不需要|不合适|不适合", text):
            return_reason = ["不喜欢/不需要"]
        else:
            reason_match = re.search(r"(?:退款|退货)(?:的)?(?:原因|理由)(?:是|为|：|:)\s*([^，。；！？\n]+)", text)
            if reason_match:
                return_reason = [reason_match.group(1).strip()]

    intercept_or_change: List[str] = []
    if re.search(r"拦截.{0,6}(?:并|和).{0,4}(?:改地址|改址)|(?:改地址|改址).{0,6}(?:并|和).{0,4}拦截", text):
        intercept_or_change = ["拦截并改址"]
    elif re.search(r"拦截(?:一下|包裹|快递)?", text):
        intercept_or_change = ["拦截包裹"]
    elif re.search(r"改地址|修改地址|改址|换地址", text):
        intercept_or_change = ["修改地址"]

    signed_info: List[str] = []
    if re.search(r"显示签收.{0,8}(?:没收到|没有收到|未收到)|签收了.{0,8}(?:没收到|没有收到|未收到)", text):
        signed_info = ["物流显示签收，消费者未收到"]
    elif re.search(r"本人已签收|我已签收|已经签收|已签收", text):
        signed_info = ["消费者确认已签收"]

    entities = {
        "order_id": _unique(_ORDER_ID_RE.findall(text)),
        "product_name": [],
        "sku": [],
        "shade": _unique(re.findall(r"(?:色号|色號)\s*([A-Za-z0-9一二三四五六七八九十-]+)", text, re.IGNORECASE)),
        "quantity": _unique(re.findall(r"(?:缺少|漏发|漏了|少了|数量)\s*(\d+|[一二两三四五六七八九十]+)\s*(?:件|个|支|瓶)?", text)),
        "symptom": _unique(re.findall(r"(红肿|脸红|泛红|发红|起疹|刺痛|发痒|瘙痒|脱皮|过敏|肿胀|呼吸困难)", text)),
        "symptom_area": _unique(re.findall(r"(全脸|脸部|脸|眼周|眼睛|嘴唇|手部|颈部|身体)", text)),
        "onset_time": onset_time,
        "stopped_use": stopped_use,
        "medical_visit": medical_visit,
        "signed_info": signed_info,
        "intercept_or_change": intercept_or_change,
        "missing_item": _unique(re.findall(r"(?:漏发|少发|漏了|少了|缺少)\s*([^，。！？\s]{1,20})", text)),
        "wrong_item": _unique(re.findall(r"(?:错发|发错)\s*([^，。！？\s]{1,20})", text)),
        "damaged_area": _unique(re.findall(r"(瓶身|包装|外盒|泵头|瓶口|漏液|破裂|压坏)", text)),
        "tracking_no": _unique(re.findall(r"(?:物流单号|运单号|快递单号)\s*[:：]?\s*([A-Za-z0-9-]{6,32})", text, re.IGNORECASE)),
        "return_reason": return_reason,
        "opened_or_used": opened_or_used,
        "return_scope": return_scope,
        "gift_opened_or_used": gift_opened_or_used,
        "refund_amount": _unique(_MONEY_RE.findall(text)),
        "price_difference": _unique(re.findall(r"(?:差价|补差)\s*[:：]?\s*([¥￥]?\s*\d+(?:\.\d{1,2})?)", text, re.IGNORECASE)),
    }
    return entities


_EMOTION_MARKERS = {
    "threatening": ("12315", "曝光", "起诉", "平台介入", "平台投诉", "限期", "最后通牒"),
    "angry": ("气死", "什么垃圾", "骗子", "骗人", "赖账", "踢皮球", "滚", "愤怒"),
    "upset": ("投诉", "太差", "怎么还", "一直没", "第3次", "第三次", "没人理", "无语", "失望", "敷衍"),
    "anxious": ("担心", "着急", "害怕", "怎么办", "赶紧", "严重吗", "会不会", "来得及"),
}


def detect_emotion(message: str) -> Tuple[str, float, List[str]]:
    """输出五档情绪标签；模型只提供证据时也必须由此固定顺序归类。"""
    text = message or ""
    for label in ("threatening", "angry", "upset", "anxious"):
        evidence = [word for word in _EMOTION_MARKERS[label] if word in text]
        if evidence:
            return label, min(1.0, 0.45 + 0.15 * len(evidence)), evidence
    return "calm", 0.0, []


def calculate_urgency(
    message: str,
    intent: Optional[IntentCategory] = None,
) -> UrgencyLevel:
    """应用固定的 CRITICAL -> HIGH -> MEDIUM -> 意图兜底优先级。

    Args:
        message: 包含紧急表达的用户消息。
        intent: 已选定的可选意图，用于无关键词时的紧急度兜底。

    Returns:
        按优先级链选出的确定性 ``UrgencyLevel``。
    """
    config = load_handoff_urgency_config()
    intent_value = str(getattr(intent, "value", intent) or "")
    priority = score_temporary_urgency(message or "", config, intent=intent_value)["priority"]
    # 普通业务路由紧急度和客服队列优先级使用同一套风险信号，保持既有枚举契约。
    return {
        "P0": UrgencyLevel.CRITICAL,
        "P1": UrgencyLevel.HIGH,
        "P2": UrgencyLevel.MEDIUM,
        "P3": UrgencyLevel.LOW,
    }.get(str(priority), UrgencyLevel.LOW)


def apply_500_override(
    message: str,
    intent: IntentCategory,
    confidence: float,
) -> Tuple[IntentCategory, float]:
    """将非登录场景的 500 响应强制覆盖为 technical_crash，置信度至少为 0.8。

    Args:
        message: 用于检查服务端错误标记的用户消息。
        intent: 当前候选意图。
        confidence: 当前候选置信度。

    Returns:
        可能经过覆盖的意图与置信度二元组。
    """
    # 500 是高风险技术信号；0.8 防止较弱来源掩盖崩溃，同时保留登录专属例外。
    if "500" in (message or "") and intent != IntentCategory.TECHNICAL_LOGIN:
        # 0.8 置信度下限确保状态码安全覆盖具有决定性。
        return IntentCategory.TECHNICAL_CRASH, max(float(confidence), 0.8)
    return intent, float(confidence)


apply_status_code_override = apply_500_override


_INTENT_GROUPS: Dict[IntentCategory, str] = {
    IntentCategory.ORDER_STATUS: "order",
    IntentCategory.LOGISTICS: "order",
    IntentCategory.REFUND: "after_sales",
    IntentCategory.INVOICE: "after_sales",
    IntentCategory.PAYMENT_ISSUE: "after_sales",
    IntentCategory.ACCOUNT_SECURITY: "consult",
    IntentCategory.TECHNICAL_LOGIN: "consult",
    IntentCategory.TECHNICAL_CRASH: "consult",
    IntentCategory.INGREDIENT_FIT: "consult",
    IntentCategory.SHADE_MATCH: "consult",
    IntentCategory.USAGE_HOWTO: "consult",
    IntentCategory.PRODUCT_INFO: "consult",
    IntentCategory.PROMO_RULES: "consult",
    IntentCategory.PREGNANCY_SAFETY: "consult",
    IntentCategory.ORDER_STATUS_BEAUTY: "order",
    IntentCategory.LOGISTICS_STALLED: "order",
    IntentCategory.SIGNED_NOT_RECEIVED: "order",
    IntentCategory.INTERCEPT_CHANGE_ADDRESS: "order",
    IntentCategory.MISSING_ITEM: "order",
    IntentCategory.WRONG_ITEM: "order",
    IntentCategory.DAMAGED_ON_ARRIVAL: "order",
    IntentCategory.RETURN_REFUND: "after_sales",
    IntentCategory.ADVERSE_REACTION: "after_sales",
    IntentCategory.SUSPECTED_ABNORMAL_REFUND: "after_sales",
    IntentCategory.PRICE_PROTECTION: "after_sales",
    IntentCategory.REFUND_NOT_RECEIVED: "after_sales",
    IntentCategory.SHIPPING_FEE_REIMBURSEMENT: "after_sales",
}


COMPOUND_CONNECTORS: List[str] = [
    "同时",
    "另外",
    "还想",
    "还要",
    "而且",
    "并且",
    "此外",
    "分别",
    "两个问题",
    "两个诉求",
    "除此之外",
    "但是",
    "不过",
    "以及",
]
COMPOUND_CONNECTOR_WHITELIST = COMPOUND_CONNECTORS

_COMPOUND_CONNECTOR_RE = "|".join(
    re.escape(item) for item in sorted(COMPOUND_CONNECTORS, key=len, reverse=True)
)
_COMPOUND_CLAUSE_SPLIT_RE = re.compile(
    rf"({_COMPOUND_CONNECTOR_RE}|[，,；;。！？!?\n])"
)
_COMPOUND_STRONG_TRIGGER_RE = re.compile(rf"(?:{_COMPOUND_CONNECTOR_RE})")
_COMPOUND_TRIM_CHARS = " \t\r\n，,；;。！？!?：:"


def _intent_domain(intent: IntentCategory) -> str:
    """将意图映射到复合路由使用的三个业务域。

    Args:
        intent: 待映射的意图。

    Returns:
        ``technical``、``billing``、``escalation`` 或 ``general``。
    """
    if intent in {
        IntentCategory.INGREDIENT_FIT,
        IntentCategory.SHADE_MATCH,
        IntentCategory.USAGE_HOWTO,
        IntentCategory.PRODUCT_INFO,
        IntentCategory.PROMO_RULES,
        IntentCategory.PREGNANCY_SAFETY,
    }:
        return "consult"
    if intent in {
        IntentCategory.ORDER_STATUS_BEAUTY,
        IntentCategory.LOGISTICS_STALLED,
        IntentCategory.SIGNED_NOT_RECEIVED,
        IntentCategory.INTERCEPT_CHANGE_ADDRESS,
        IntentCategory.MISSING_ITEM,
        IntentCategory.WRONG_ITEM,
        IntentCategory.DAMAGED_ON_ARRIVAL,
    }:
        return "order"
    if intent in {
        IntentCategory.RETURN_REFUND,
        IntentCategory.ADVERSE_REACTION,
        IntentCategory.SUSPECTED_ABNORMAL_REFUND,
        IntentCategory.PRICE_PROTECTION,
        IntentCategory.REFUND_NOT_RECEIVED,
        IntentCategory.SHIPPING_FEE_REIMBURSEMENT,
    }:
        return "after_sales"
    if intent in {
        IntentCategory.TECHNICAL,
        IntentCategory.TECHNICAL_LOGIN,
        IntentCategory.TECHNICAL_CRASH,
    }:
        return "order"
    if intent in {
        IntentCategory.BILLING,
        IntentCategory.ACCOUNT,
        IntentCategory.ACCOUNT_SECURITY,
        IntentCategory.REFUND,
        IntentCategory.INVOICE,
        IntentCategory.PAYMENT_ISSUE,
    }:
        return "after_sales"
    if intent is IntentCategory.HUMAN_HANDOFF:
        return "escalation"
    return "consult"


# 这些词只在子句被当作独立复合诉求评估时使用；这样可以提升 scoped split
# 的实用性，同时不扩大 Step 2.1 的通用 Pattern 契约。
_COMPOUND_TECHNICAL_PATTERNS: Tuple[str, ...] = (
    "开不了机",
    "无法开机",
    "屏幕花屏",
    "花屏",
    "页面打不开",
    "网络请求失败",
    "页面返回500",
    "页面报500",
    "错误码401",
    "错误码403",
)


def _split_compound_parts(message: str) -> Tuple[List[str], List[str]]:
    """按白名单连接词和标点拆分子句并记录触发词。

    Args:
        message: 原始用户消息。

    Returns:
        子句列表与按原文顺序记录的连接词/标点列表。
    """
    text = message or ""
    parts = _COMPOUND_CLAUSE_SPLIT_RE.split(text)
    clauses: List[str] = []
    triggers: List[str] = []
    pending_trigger = ""
    for part in parts:
        if not part:
            continue
        if _COMPOUND_CLAUSE_SPLIT_RE.fullmatch(part):
            if part in COMPOUND_CONNECTORS or re.fullmatch(r"[，,；;。！？!?\n]", part):
                pending_trigger = part
            continue
        clause = part.strip(_COMPOUND_TRIM_CHARS)
        if clause:
            clauses.append(clause)
            if pending_trigger:
                triggers.append(pending_trigger)
                pending_trigger = ""
    return clauses, triggers


def split_compound_clauses(message: str) -> List[str]:
    """仅按批准的连接词和常见句末标点拆分复合消息。

    Args:
        message: 原始用户消息。

    Returns:
        过滤空白后的子句列表。
    """
    clauses, _ = _split_compound_parts(message)
    return clauses


def _compound_match_lengths(pattern: str, clause: str) -> List[int]:
    """计算 Pattern 在单个子句中的匹配长度。

    Args:
        pattern: 待匹配的正则或字面规则。
        clause: 独立复合子句。

    Returns:
        所有命中片段的字符长度；规则非法时使用安全字面匹配。
    """
    try:
        return [len(match.group(0)) for match in re.finditer(pattern, clause, re.IGNORECASE)]
    except re.error:
        return [len(pattern)] if pattern.casefold() in clause.casefold() else []


def _compound_clause_candidates(clause: str) -> List[Dict[str, Any]]:
    """为一个子句生成按匹配长度排序的意图候选。

    Args:
        clause: 待独立识别的复合子句。

    Returns:
        包含意图、匹配长度、特异性和命中文本的候选字典列表。
    """
    candidates: List[Dict[str, Any]] = []
    pattern_tables = (
        (True, SPECIFIC_PATTERNS),
        (False, GENERIC_PATTERNS),
        (True, {IntentCategory.TECHNICAL: _COMPOUND_TECHNICAL_PATTERNS}),
    )
    for is_specific, table in pattern_tables:
        for intent, patterns in table.items():
            for pattern in patterns:
                lengths = _compound_match_lengths(pattern, clause)
                if lengths:
                    candidates.append(
                        {
                            "intent": intent,
                            "length": max(lengths),
                            "specific": is_specific,
                            "match": max(
                                (match.group(0) for match in re.finditer(pattern, clause, re.IGNORECASE)),
                                key=len,
                                default=pattern,
                            ),
                        }
                    )
    candidates.sort(key=lambda item: (-item["length"], not item["specific"]))
    return candidates


def _best_compound_clause_candidate(clause: str) -> Optional[Dict[str, Any]]:
    """筛选一个子句的最长且最具体合法候选。

    Args:
        clause: 待筛选的复合子句。

    Returns:
        最佳候选字典；没有规则命中时返回 ``None``。
    """
    candidates = _compound_clause_candidates(clause)
    if not candidates:
        return None
    specific_domains = {
        _intent_domain(item["intent"])
        for item in candidates
        if item["specific"] and item["intent"] is not IntentCategory.HUMAN_HANDOFF
    }
    filtered = [
        item
        for item in candidates
        if not (
            not item["specific"]
            and _intent_domain(item["intent"]) in specific_domains
        )
    ]
    return (filtered or candidates)[0]


def detect_compound_intents(
    message: str,
    primary_intent: IntentCategory,
) -> Tuple[List[IntentCategory], str, Dict[str, Any]]:
    """为复合消息返回保守的跨域辅助意图。

    Args:
        message: 原始用户消息。
        primary_intent: 已确定的主意图。

    Returns:
        辅助意图列表、触发词和复合识别诊断字典。
    """
    text = message or ""
    if primary_intent is IntentCategory.HUMAN_HANDOFF:
        return [], "", {"terminal_handoff": True, "clauses": [], "clause_matches": []}

    clauses, triggers = _split_compound_parts(text)
    diagnostics: Dict[str, Any] = {
        "terminal_handoff": False,
        "clauses": list(clauses),
        "clause_matches": [],
        "clause_intents": [],
        "compound_detected": len(clauses) >= 2,
        "sub_requests": [],
    }
    if len(clauses) < 2:
        return [], "", diagnostics

    clause_matches: List[Dict[str, Any]] = []
    for clause in clauses:
        candidate = _best_compound_clause_candidate(clause)
        if candidate is None:
            continue
        record = {
            "clause": clause,
            "intent": candidate["intent"].value,
            "domain": _intent_domain(candidate["intent"]),
            "match": candidate["match"],
            "specific": bool(candidate["specific"]),
        }
        clause_matches.append(record)

    diagnostics["clause_matches"] = clause_matches
    diagnostics["clause_intents"] = [item["intent"] for item in clause_matches]
    diagnostics["sub_requests"] = [
        {
            "id": f"sub_{index}",
            "text": item["clause"],
            "domain": item["domain"],
            "intent": item["intent"],
            "confidence": 0.75 if item["specific"] else 0.55,
            "knowledge_required": item["domain"] != "consult",
        }
        for index, item in enumerate(clause_matches, start=1)
    ]
    primary_domain = _intent_domain(primary_intent)
    secondary: List[IntentCategory] = []
    for item in clause_matches:
        intent = IntentCategory(item["intent"])
        if intent is IntentCategory.HUMAN_HANDOFF:
            continue
        if intent is primary_intent or _intent_domain(intent) == primary_domain:
            continue
        if intent not in secondary:
            secondary.append(intent)

    secondary = secondary[:3]
    trigger = triggers[0] if triggers else ""
    if not secondary:
        diagnostics["compound_detected"] = False
    diagnostics["secondary_intents"] = [intent.value for intent in secondary]
    if diagnostics["sub_requests"]:
        diagnostics["primary_sub_request_id"] = diagnostics["sub_requests"][0]["id"]
    return secondary, trigger, diagnostics


def build_scoped_retrieval_queries(
    message: str,
    secondary_intents: List[IntentCategory],
    clause_matches: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, str]:
    """构造领域检索 query；没有合法辅助意图时保留完整原文。

    Args:
        message: 原始用户消息。
        secondary_intents: 已过滤的跨域辅助意图。
        clause_matches: 子句命中诊断，用于提取领域范围文本。

    Returns:
        以业务域为 key 的 scoped retrieval query 字典。
    """
    text = message or ""
    if not secondary_intents and not clause_matches:
        return {"default": text}
    matches = clause_matches or []
    queries: Dict[str, List[str]] = {}
    for item in matches:
        intent = _coerce_intent(item.get("intent"))
        domain = _intent_domain(intent)
        queries.setdefault(domain, []).append(item["clause"])
    return {
        domain: "；".join(dict.fromkeys(clauses))
        for domain, clauses in queries.items()
    } or {"default": text}


def _normalize_structured_sub_requests(value: Any) -> List[Dict[str, Any]]:
    """校验 LLM 返回的复合子请求，非法项整体丢弃而不影响主链路。"""

    if not isinstance(value, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            continue
        text = str(item.get("text") or "").strip()
        domain = str(item.get("domain") or "").strip().lower()
        intent = _coerce_intent(item.get("intent"))
        if not text or domain not in {"consult", "order", "after_sales"}:
            continue
        if _intent_domain(intent) != domain:
            continue
        try:
            confidence = _bounded_score(float(item.get("confidence", 0.0)))
        except (TypeError, ValueError):
            continue
        normalized.append(
            {
                "id": str(item.get("id") or f"llm_sub_{index}"),
                "text": text,
                "domain": domain,
                "intent": intent.value,
                "confidence": confidence,
                "knowledge_required": domain != "consult",
            }
        )
    return normalized


def _resolve_structured_compound(
    message: str,
    primary_intent: IntentCategory,
    sub_requests: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """把经过校验的 LLM 子请求转换为统一的复合意图诊断结构。"""

    primary_domain = _intent_domain(primary_intent)
    secondary: List[IntentCategory] = []
    clause_matches: List[Dict[str, Any]] = []
    for item in sub_requests:
        intent = _coerce_intent(item["intent"])
        clause_matches.append(
            {
                "clause": item["text"],
                "intent": intent.value,
                "domain": item["domain"],
                "match": "llm_structured",
                "specific": True,
            }
        )
        if intent is IntentCategory.HUMAN_HANDOFF:
            continue
        if intent is not primary_intent and item["domain"] != primary_domain and intent not in secondary:
            secondary.append(intent)
    diagnostics = {
        "terminal_handoff": False,
        "clauses": [item["text"] for item in sub_requests],
        "clause_matches": clause_matches,
        "clause_intents": [item["intent"] for item in sub_requests],
        "compound_detected": len(sub_requests) >= 2,
        "secondary_intents": [intent.value for intent in secondary[:3]],
        "sub_requests": sub_requests,
        "primary_sub_request_id": sub_requests[0]["id"],
    }
    diagnostics["scoped_retrieval_queries"] = build_scoped_retrieval_queries(
        message, secondary, clause_matches
    )
    return {
        "secondary_intents": secondary[:3],
        "compound_trigger": "llm_structured",
        "scoped_retrieval_queries": diagnostics["scoped_retrieval_queries"],
        "diagnostics": diagnostics,
    }


def resolve_compound_intents(
    message: str,
    primary_intent: IntentCategory,
    structured_sub_requests: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """完成复合意图识别并装配 scoped retrieval 诊断。

    Args:
        message: 原始用户消息。
        primary_intent: 已确定的主意图。

    Returns:
        包含辅助意图、触发词、scoped query 与完整诊断的字典。
    """
    if structured_sub_requests and len(structured_sub_requests) >= 2:
        return _resolve_structured_compound(
            message, primary_intent, structured_sub_requests
        )
    secondary, trigger, diagnostics = detect_compound_intents(message, primary_intent)
    diagnostics["scoped_retrieval_queries"] = build_scoped_retrieval_queries(
        message, secondary, diagnostics.get("clause_matches")
    )
    return {
        "secondary_intents": secondary,
        "compound_trigger": trigger,
        "scoped_retrieval_queries": diagnostics["scoped_retrieval_queries"],
        "diagnostics": diagnostics,
    }


def _coerce_intent(value: Any) -> IntentCategory:
    """将外部来源值安全转换为 ``IntentCategory``。

    Args:
        value: 可能是枚举、字符串或非法外部值。

    Returns:
        合法意图；无法转换时返回 ``OTHER``。
    """
    if isinstance(value, IntentCategory):
        return value
    try:
        return IntentCategory(str(value))
    except (TypeError, ValueError):
        return IntentCategory.OTHER


def _bounded_score(value: Any) -> float:
    """将来源分数安全归一化到 0..1。

    Args:
        value: 外部来源给出的任意分数值。

    Returns:
        有限且位于 0..1 的浮点分数。
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    numeric = float(value)
    if not math.isfinite(numeric):
        return 0.0
    return min(1.0, max(0.0, numeric))


def _normalize_llm_source(value: Any) -> LLMIntentResult:
    """把 LLM 来源的字典或对象归一化为内部 dataclass。

    Args:
        value: LLM 来源返回值。

    Returns:
        归一化后的 LLM 结果，异常结构转为失败结果。
    """
    if isinstance(value, LLMIntentResult):
        if value.intent.value not in _ACTIVE_INTENT_IDS:
            return _failed_llm_result(f"LLM 返回了未注册的美妆意图: {value.intent.value}")
        return value
    if isinstance(value, dict):
        raw_intent = str(value.get("intent", ""))
        if raw_intent not in _ACTIVE_INTENT_IDS:
            return _failed_llm_result(f"LLM 返回了未注册的美妆意图: {raw_intent or 'empty'}")
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
    """把 Embedding 来源的字典或对象归一化为内部 dataclass。

    Args:
        value: Embedding 来源返回值。

    Returns:
        归一化后的 Embedding 结果，异常结构转为失败结果。
    """
    if isinstance(value, EmbeddingIntentResult):
        if value.intent.value not in _ACTIVE_INTENT_IDS:
            return EmbeddingIntentResult(
                intent=IntentCategory.OTHER,
                score=0.0,
                success=False,
                reasoning=f"Embedding 返回了未注册的美妆意图: {value.intent.value}",
            )
        return value
    if isinstance(value, dict):
        raw_intent = str(value.get("intent", ""))
        if raw_intent not in _ACTIVE_INTENT_IDS:
            return EmbeddingIntentResult(
                intent=IntentCategory.OTHER,
                score=0.0,
                success=False,
                reasoning=f"Embedding 返回了未注册的美妆意图: {raw_intent or 'empty'}",
            )
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
    """将 Pattern 结果中的字符串意图分数转换为枚举映射。

    Args:
        pattern_result: Pattern 识别器的兼容结果字典。

    Returns:
        以 ``IntentCategory`` 为 key 的安全分数映射。
    """
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
    """返回意图对应的粗粒度业务分组。

    Args:
        intent: 已归一化的意图。

    Returns:
        下游路由使用的分组字符串。
    """
    return _INTENT_GROUPS.get(intent, intent.value)


async def recognize_intent(
    message: str,
    history: Optional[List[Any]] = None,
) -> IntentResult:
    """并发运行 LLM、Embedding、Pattern，并应用固定融合状态机。

    Args:
        message: 待识别的用户消息。
        history: 可选的近期会话上下文。

    Returns:
        装配了意图、置信度、实体、紧急度和诊断信息的 ``IntentResult``。
    """
    started_at = time.perf_counter()
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

    # 常规且证据明确的问题走本地快路径，不等待远程 LLM 或 BGE。
    fast_intents = {
        IntentCategory.INGREDIENT_FIT, IntentCategory.SHADE_MATCH,
        IntentCategory.USAGE_HOWTO, IntentCategory.PRODUCT_INFO,
        IntentCategory.PROMO_RULES, IntentCategory.ORDER_STATUS_BEAUTY,
        IntentCategory.LOGISTICS_STALLED, IntentCategory.MISSING_ITEM,
        IntentCategory.WRONG_ITEM, IntentCategory.DAMAGED_ON_ARRIVAL,
        IntentCategory.RETURN_REFUND, IntentCategory.REFUND_NOT_RECEIVED,
    }
    fast_intent = _coerce_intent(pattern_result.get("specific_intent"))
    fast_score = _bounded_score(pattern_result.get("specific_score", 0.0))
    # 特异 Pattern 本身就是业务词组的精确命中；单条命中即允许快路径，
    # 泛化 Pattern 仍然不会绕过模型，避免把模糊表达误判成业务意图。
    if fast_intent in fast_intents and fast_score >= 0.5:
        # 快路径只跳过 LLM/Embedding，不跳过复合请求拆分；否则主意图虽然快速命中，
        # 其他业务域诉求会从 Agent Fan-out 和工单子任务中消失。
        compound = resolve_compound_intents(message, fast_intent)
        emotion, emotion_intensity, emotion_evidence = detect_emotion(message)
        diagnostics = {
            "fast_path": True,
            "fast_path_reason": "specific_pattern",
            "final_intent": fast_intent.value,
            "final_confidence": fast_score,
            "source_intents": {"pattern": fast_intent.value},
            "source_scores": {"pattern": fast_score},
            "knowledge_required": fast_intent not in {IntentCategory.ORDER_STATUS_BEAUTY, IntentCategory.LOGISTICS_STALLED},
        }
        diagnostics.update(compound["diagnostics"])
        diagnostics["scoped_retrieval_queries"] = compound["scoped_retrieval_queries"]
        return IntentResult(
            intent=fast_intent,
            confidence=fast_score,
            urgency=calculate_urgency(message, fast_intent),
            intent_group=_intent_group(fast_intent),
            entities=extract_entities(message),
            reasoning="采用明确的本地业务规则，无需等待模型",
            latency_ms=max(0.0, round((time.perf_counter() - started_at) * 1000, 3)),
            source_scores={"pattern": fast_score},
            source_intents={"pattern": fast_intent.value},
            secondary_intents=list(compound["secondary_intents"]),
            compound_trigger=str(compound["compound_trigger"] or ""),
            intent_diagnostics=diagnostics,
            sub_requests=list(compound["diagnostics"].get("sub_requests") or []),
            emotion=emotion,
            emotion_intensity=emotion_intensity,
            emotion_evidence=emotion_evidence,
        )

    llm_task = asyncio.create_task(_recognize_with_llm(message, history))
    embedding_task = asyncio.create_task(_recognize_with_embedding(message))

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
    if llm.diagnostics:
        diagnostics.update(llm.diagnostics)

    final_intent = IntentCategory.OTHER
    final_confidence = 0.0
    if not llm.success:
        if specific_score > 0.0 and specific_intent is not IntentCategory.OTHER:
            final_intent = specific_intent
            final_confidence = specific_score
            diagnostics["fallback"] = "specific_pattern"
        # 0.45 是 LLM 失败时 Embedding 作为兜底的最低相似度，避免弱匹配误路由。
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
        # 融合分低于 0.5 时证据不足，强制 OTHER 以降低错误分流风险。
        if fused_confidence < 0.5:
            diagnostics["fallback"] = "low_fused_score"
        else:
            final_intent = candidate
            final_confidence = fused_confidence
            if (
                candidate in _GENERIC_INTENTS
                and specific_intent is not IntentCategory.OTHER
                and specific_intent is not candidate
                # 0.5 是特异规则最低证据，0.8 是避免覆盖高置信泛化结果的保护线。
                and specific_score >= 0.5
                and fused_confidence < 0.8
            ):
                final_intent = specific_intent
                final_confidence = max(fused_confidence, specific_score)
                diagnostics["refined_from"] = candidate.value

    # 500/technical_crash 属于旧通用客服域，不在当前美妆 19 意图注册表内；
    # 不允许旧覆盖规则绕过已确认的美妆意图边界。
    diagnostics["status_code_override"] = False
    diagnostics["final_intent"] = final_intent.value
    normalized_confidence = _bounded_score(round(final_confidence, 6))
    diagnostics["final_confidence"] = normalized_confidence

    structured_sub_requests = _normalize_structured_sub_requests(
        llm.diagnostics.get("llm_sub_requests")
    ) if llm.success and llm.diagnostics.get("llm_is_compound") else []
    compound = resolve_compound_intents(message, final_intent, structured_sub_requests)
    explicit_handoff = is_explicit_handoff_requested(
        message, load_handoff_urgency_config()
    )
    diagnostics["explicit_handoff_requested"] = explicit_handoff
    # 复合请求已经成功拆出结构化子请求时，优先使用第一个子请求作为主路由锚点。
    # 顶层 human_handoff 只有在用户明确表达人工诉求时才允许覆盖业务子请求；
    # CRITICAL 则交给后续路由安全红线处理。
    if structured_sub_requests and len(structured_sub_requests) >= 2:
        if not explicit_handoff:
            candidate_intent = _coerce_intent(structured_sub_requests[0].get("intent"))
            if candidate_intent not in {
                IntentCategory.OTHER,
                IntentCategory.HUMAN_HANDOFF,
            }:
                if final_intent is IntentCategory.HUMAN_HANDOFF:
                    diagnostics["compound_top_level_handoff_ignored"] = True
                final_intent = candidate_intent
                final_confidence = max(
                    normalized_confidence,
                    float(structured_sub_requests[0].get("confidence") or 0.55),
                )
                normalized_confidence = _bounded_score(round(final_confidence, 6))
                diagnostics["final_intent"] = final_intent.value
                diagnostics["final_confidence"] = normalized_confidence
                diagnostics["compound_primary_subrequest_override"] = candidate_intent.value
                compound = resolve_compound_intents(
                    message,
                    final_intent,
                    structured_sub_requests,
                )
    if final_intent is IntentCategory.HUMAN_HANDOFF and not explicit_handoff:
        # Why：模型可能把强烈情绪或风险误分类成“转人工”；实际转接只由明确请求触发。
        candidates = [specific_intent]
        candidates.extend(
            _coerce_intent(item.get("intent"))
            for item in structured_sub_requests
            if isinstance(item, Mapping)
        )
        candidate_intent = next(
            (
                item for item in candidates
                if item not in {IntentCategory.OTHER, IntentCategory.HUMAN_HANDOFF}
                and item.value in {entry["intent_id"] for entry in REGISTRY_INTENTS}
            ),
            IntentCategory.OTHER,
        )
        final_intent = candidate_intent
        final_confidence = specific_score if candidate_intent is specific_intent else 0.0
        normalized_confidence = _bounded_score(round(final_confidence, 6))
        diagnostics["unrequested_handoff_prediction_demoted"] = True
        diagnostics["final_intent"] = final_intent.value
        diagnostics["final_confidence"] = normalized_confidence
        compound = resolve_compound_intents(message, final_intent, structured_sub_requests)
    # 复合消息若融合结果退化为 OTHER，优先使用第一个明确业务子请求作为主路由，
    # 再按该主域重新计算 secondary，避免整条请求直接进入澄清分支。
    if not structured_sub_requests and final_intent in {
        IntentCategory.OTHER,
        IntentCategory.QUERY,
        IntentCategory.REQUEST,
    } and compound["diagnostics"].get("compound_detected"):
        sub_requests = compound["diagnostics"].get("sub_requests") or []
        if sub_requests:
            candidate_intent = _coerce_intent(sub_requests[0].get("intent"))
            if candidate_intent not in {
                IntentCategory.OTHER,
                IntentCategory.HUMAN_HANDOFF,
            }:
                final_intent = candidate_intent
                final_confidence = max(
                    normalized_confidence,
                    float(sub_requests[0].get("confidence") or 0.55),
                )
                normalized_confidence = _bounded_score(round(final_confidence, 6))
                diagnostics["final_intent"] = final_intent.value
                diagnostics["final_confidence"] = normalized_confidence
                diagnostics["compound_primary_fallback"] = candidate_intent.value
                compound = resolve_compound_intents(
                    message,
                    final_intent,
                    structured_sub_requests,
                )
    diagnostics.update(compound["diagnostics"])
    diagnostics["scoped_retrieval_queries"] = compound["scoped_retrieval_queries"]

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

    emotion, emotion_intensity, emotion_evidence = detect_emotion(message)
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
        secondary_intents=compound["secondary_intents"],
        compound_trigger=compound["compound_trigger"],
        intent_diagnostics=diagnostics,
        sub_requests=list(compound["diagnostics"].get("sub_requests") or []),
        emotion=emotion,
        emotion_intensity=emotion_intensity,
        emotion_evidence=emotion_evidence,
    )
