"""ShopCare 领域 Agent 契约、白名单执行边界与确定性升级 Agent。

本模块属于 Agent 能力层：定义内部 dataclass 契约、四类领域 Profile，以及
受工具白名单约束的 Anthropic Tool Use 执行基类。它不实现多 Agent 路由、结果
合成或任何真实订单、支付、工单和人工坐席接口；这些能力由后续编排步骤负责。
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple

from core.intent_recognizer import (
    IntentCategory,
    IntentResult,
    UrgencyLevel,
    _intent_domain,
)
from agents.tools import (
    AgentToolSpec,
    CREATE_HANDOFF_SUMMARY_TOOL,
    billing_tools,
    general_tools,
    make_tool,
    technical_tools,
)


logger = logging.getLogger(__name__)


class AgentType(str, Enum):
    GENERAL = "general"
    TECHNICAL = "technical"
    BILLING = "billing"
    ESCALATION = "escalation"


@dataclass(frozen=True)
class AgentProfile:
    role: str
    mission: str
    workflow: Tuple[str, ...]
    input_contract: Tuple[str, ...]
    output_contract: Tuple[str, ...]
    handoff_conditions: Tuple[str, ...] = ()
    tool_scope: Tuple[str, ...] = ()
    model: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 1024


@dataclass
class AgentResponse:
    agent_type: AgentType
    content: str
    success: bool
    confidence: float = 1.0
    latency_ms: float = 0.0
    escalate: bool = False
    tools_used: List[str] = field(default_factory=list)
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Request:
    message: str
    user_id: str = "anonymous"
    conv_id: Optional[str] = None
    context: str = ""
    history: Optional[List[Any]] = None
    entities: Dict[str, List[str]] = field(default_factory=dict)
    intent: Optional[IntentCategory] = None
    intent_group: Optional[str] = None
    urgency: Optional[UrgencyLevel] = None
    intent_confidence: float = 1.0
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    intent_diagnostics: Dict[str, Any] = field(default_factory=dict)
    secondary_intents: List[IntentCategory] = field(default_factory=list)
    knowledge_required: bool = False


@dataclass
class RoutingDecision:
    # 澄清分支没有可执行 Agent，因此允许显式使用 None 表示不派发。
    primary_agent: Optional[AgentType]
    supporting_agents: List[AgentType] = field(default_factory=list)
    reason: str = ""
    confidence: float = 0.0


@dataclass
class AgentStats:
    """记录单个 Agent 的调用次数、成功次数和耗时统计。

    Returns:
        统计对象本身；`success_rate`、`avg_ms` 和 `routing_score` 通过属性读取。
    """

    total: int = 0
    success: int = 0
    total_ms: float = 0.0
    monitor_penalty: float = 0.0

    @property
    def success_rate(self) -> float:
        """返回成功率；尚未调用时返回 0，避免伪造成功数据。"""

        return self.success / self.total if self.total else 0.0

    @property
    def avg_ms(self) -> float:
        """返回平均耗时毫秒数。"""

        return self.total_ms / self.total if self.total else 0.0

    @property
    def routing_score(self) -> float:
        """按设计基线计算扣除监控惩罚后的路由评分。"""

        score = (self.success_rate * 0.7 + (1 / (1 + self.avg_ms / 1000)) * 0.3)
        return score * max(0.0, 1.0 - self.monitor_penalty)


def _mapping_value(value: Any, key: str, default: Any = None) -> Any:
    """兼容 Anthropic 对象块和测试字典块的字段读取。"""

    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _normalize_content_block(block: Any) -> Dict[str, Any]:
    """把 SDK 响应块转换为可安全放入下一轮 messages 的字典。"""

    block_type = _mapping_value(block, "type")
    if block_type == "tool_use":
        return {
            "type": "tool_use",
            "id": _mapping_value(block, "id", "tool-use"),
            "name": _mapping_value(block, "name", ""),
            "input": _mapping_value(block, "input", {}),
        }
    if block_type == "text":
        return {"type": "text", "text": str(_mapping_value(block, "text", ""))}
    if isinstance(block, Mapping):
        return dict(block)
    return {"type": str(block_type or "text"), "text": str(block)}


def _response_blocks(response: Any) -> List[Any]:
    """提取 Anthropic response 的 content block 列表。"""

    content = _mapping_value(response, "content", [])
    if content is None:
        return []
    return list(content) if isinstance(content, list) else [content]


async def _search_knowledge_base_placeholder(params: Any) -> Dict[str, Any]:
    """提供待 Step 6 注入的共享检索工具占位，不连接外部知识库。"""

    if not isinstance(params, Mapping) or not isinstance(params.get("query"), str):
        return {"success": False, "error": "query 必须是字符串", "available": False}
    return {
        "success": False,
        "error": "知识库检索工具尚未由编排层注入",
        "available": False,
        "read_only": True,
    }


SEARCH_KNOWLEDGE_BASE_TOOL = make_tool(
    "search_knowledge_base",
    "检索只读客服知识库；实际 KnowledgeBase handler 由后续编排层注入。",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    _search_knowledge_base_placeholder,
)


class BaseAgent(ABC):
    """受工具白名单约束的 Anthropic Tool Use Agent 基类。

    Args:
        agent_type: Agent 的领域类型。
        profile: 该领域固定的职责、工具范围与模型超参。
        tools: 允许模型看到和调用的 `AgentToolSpec` 列表。
        client: 可选的 Anthropic 异步客户端；缺失时安全失败，不自动联网创建。
        stats: 可注入的统计对象，未提供时创建独立实例。

    Returns:
        `handle()` 返回 `AgentResponse`，所有运行时异常均在 Agent 边界内隔离。
    """

    MAX_TOOL_ROUNDS = 3

    def __init__(
        self,
        agent_type: AgentType,
        profile: AgentProfile,
        tools: Optional[List[AgentToolSpec]] = None,
        client: Optional[Any] = None,
        stats: Optional[AgentStats] = None,
    ) -> None:
        self.agent_type = agent_type
        self.profile = profile
        self.tools: List[AgentToolSpec] = list(tools or [])
        self.client = client
        self.stats = stats or AgentStats()

    def _tool_context(self, request: Request) -> Dict[str, Any]:
        """构造只读请求快照，供白名单工具执行时使用。"""

        intent = request.intent.value if isinstance(request.intent, IntentCategory) else request.intent
        return {
            "request_id": request.request_id,
            "user_id": request.user_id,
            "intent": intent or "other",
            "timestamp": None,
        }

    async def _run_llm(
        self,
        request: Request,
        tools_used: List[str],
        tool_traces: List[Dict[str, Any]],
    ) -> str:
        """执行最多三轮 Tool Use，并只暴露当前 Agent 的工具白名单。"""

        if self.client is None:
            raise RuntimeError("Anthropic client 未配置")
        messages: List[Dict[str, Any]] = [{"role": "user", "content": request.message}]
        declarations = [tool.to_anthropic_tool() for tool in self.tools]
        tool_map = {tool.name: tool for tool in self.tools}
        model = self.profile.model or "claude-3-5-sonnet-20241022"

        for round_index in range(self.MAX_TOOL_ROUNDS):
            response = await self.client.messages.create(
                model=model,
                max_tokens=self.profile.max_tokens,
                temperature=self.profile.temperature,
                system=self.profile.mission,
                tools=declarations,
                messages=messages,
            )
            blocks = _response_blocks(response)
            text_parts = [
                str(_mapping_value(block, "text", ""))
                for block in blocks
                if _mapping_value(block, "type") == "text"
            ]
            tool_blocks = [
                block
                for block in blocks
                if _mapping_value(block, "type") == "tool_use"
            ]
            if not tool_blocks:
                text = "\n".join(part for part in text_parts if part).strip()
                if not text:
                    raise ValueError("模型未返回文本或工具调用")
                return text

            messages.append(
                {
                    "role": "assistant",
                    "content": [_normalize_content_block(block) for block in blocks],
                }
            )
            tool_results: List[Dict[str, Any]] = []
            for block in tool_blocks:
                name = _mapping_value(block, "name", "")
                tool = tool_map.get(name)
                if tool is None:
                    raise PermissionError(f"Agent 不允许调用工具: {name}")
                tool_input = _mapping_value(block, "input", {})
                tool_result = await tool.execute(tool_input, context=self._tool_context(request))
                tools_used.append(name)
                tool_traces.append(
                    {
                        "round": round_index + 1,
                        "tool": name,
                        "success": bool(
                            tool_result.get("success", False)
                            if isinstance(tool_result, Mapping)
                            else True
                        ),
                    }
                )
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": _mapping_value(block, "id", f"tool-{round_index + 1}"),
                        "content": json.dumps(tool_result, ensure_ascii=False, default=str),
                    }
                )
            messages.append({"role": "user", "content": tool_results})

            # Why：三轮上限控制 Token 与工具副作用边界，防止模型在 ReAct 循环中无限调用。
            if round_index == self.MAX_TOOL_ROUNDS - 1:
                return "\n".join(part for part in text_parts if part).strip() or "已完成必要的工具处理。"

        raise RuntimeError("Agent Tool Use 循环异常结束")

    async def handle(self, request: Request) -> AgentResponse:
        """处理一次领域请求并隔离异常、更新统计。

        Args:
            request: 内部 dataclass 请求，包含用户消息、意图和会话上下文。

        Returns:
            成功时返回模型文本与工具轨迹；失败时返回安全错误文本和非负耗时，
            不向上游抛出运行时异常。
        """

        started = time.perf_counter()
        self.stats.total += 1
        tools_used: List[str] = []
        tool_traces: List[Dict[str, Any]] = []
        try:
            content = await self._run_llm(request, tools_used, tool_traces)
            latency_ms = max(0.0, (time.perf_counter() - started) * 1000)
            self.stats.success += 1
            self.stats.total_ms += latency_ms
            return AgentResponse(
                agent_type=self.agent_type,
                content=content,
                success=True,
                latency_ms=latency_ms,
                tools_used=tools_used,
                tool_traces=tool_traces,
            )
        except Exception as exc:
            latency_ms = max(0.0, (time.perf_counter() - started) * 1000)
            self.stats.total_ms += latency_ms
            return AgentResponse(
                agent_type=self.agent_type,
                content=f"Agent 执行失败: {exc}",
                success=False,
                confidence=0.0,
                latency_ms=latency_ms,
                tools_used=tools_used,
                tool_traces=tool_traces,
            )


GENERAL_PROFILE = AgentProfile(
    role="general",
    mission="负责首轮接待、澄清问题并提供通用客服知识回答。",
    workflow=("inspect_context", "search_knowledge", "clarify", "respond"),
    input_contract=("message", "history", "intent", "urgency", "context"),
    output_contract=("response", "required_fields", "knowledge_context"),
    tool_scope=("search_knowledge_base", "inspect_request_context", "suggest_required_fields"),
    temperature=0.3,
    max_tokens=900,
)

TECHNICAL_PROFILE = AgentProfile(
    role="technical",
    mission="负责错误码解释、环境排障和低风险诊断建议。",
    workflow=("inspect_error", "search_knowledge", "diagnose", "respond"),
    input_contract=("message", "entities", "context", "urgency"),
    output_contract=("response", "diagnostic_plan", "knowledge_context"),
    tool_scope=("search_knowledge_base", "lookup_error_code", "build_diagnostic_plan"),
    temperature=0.1,
    max_tokens=1200,
)

BILLING_PROFILE = AgentProfile(
    role="billing",
    mission="负责账单、退款和发票所需字段的只读核验。",
    workflow=("inspect_fields", "search_knowledge", "compare", "respond"),
    input_contract=("message", "entities", "context", "urgency"),
    output_contract=("response", "missing_fields", "amount_comparison", "knowledge_context"),
    tool_scope=("search_knowledge_base", "check_billing_fields", "compare_amounts"),
    temperature=0.0,
    max_tokens=1100,
)

ESCALATION_PROFILE = AgentProfile(
    role="escalation",
    mission="记录升级原因、意图、紧急度和实体，生成会话内交接建议。",
    workflow=("summarize", "recommend_handoff"),
    input_contract=("message", "intent", "urgency", "entities", "context"),
    output_contract=("handoff_summary", "recommendation"),
    handoff_conditions=("critical", "human_handoff", "agent_recommended_handoff"),
    tool_scope=("search_knowledge_base", "create_handoff_summary"),
    temperature=0.0,
    max_tokens=500,
)


class GeneralAgent(BaseAgent):
    """通用客服 Agent，提供上下文检查、补字段和共享知识检索。"""

    def __init__(
        self,
        client: Optional[Any] = None,
        tools: Optional[List[AgentToolSpec]] = None,
        stats: Optional[AgentStats] = None,
    ) -> None:
        default_tools = list(general_tools().values())
        default_tools.insert(0, SEARCH_KNOWLEDGE_BASE_TOOL)
        super().__init__(
            AgentType.GENERAL,
            GENERAL_PROFILE,
            default_tools if tools is None else tools,
            client,
            stats,
        )


class TechnicalAgent(BaseAgent):
    """Technical Agent，隔离错误码查询与低风险诊断工具。"""

    def __init__(
        self,
        client: Optional[Any] = None,
        tools: Optional[List[AgentToolSpec]] = None,
        stats: Optional[AgentStats] = None,
    ) -> None:
        default_tools = [SEARCH_KNOWLEDGE_BASE_TOOL, *technical_tools().values()]
        super().__init__(
            AgentType.TECHNICAL,
            TECHNICAL_PROFILE,
            default_tools if tools is None else tools,
            client,
            stats,
        )


class BillingAgent(BaseAgent):
    """Billing Agent，隔离账单字段核验与金额算术工具。"""

    def __init__(
        self,
        client: Optional[Any] = None,
        tools: Optional[List[AgentToolSpec]] = None,
        stats: Optional[AgentStats] = None,
    ) -> None:
        default_tools = [SEARCH_KNOWLEDGE_BASE_TOOL, *billing_tools().values()]
        super().__init__(
            AgentType.BILLING,
            BILLING_PROFILE,
            default_tools if tools is None else tools,
            client,
            stats,
        )


class EscalationAgent(BaseAgent):
    """Escalation Agent，以确定性摘要表达升级建议而不调用 LLM。"""

    def __init__(
        self,
        client: Optional[Any] = None,
        tools: Optional[List[AgentToolSpec]] = None,
        stats: Optional[AgentStats] = None,
    ) -> None:
        default_tools = [SEARCH_KNOWLEDGE_BASE_TOOL, CREATE_HANDOFF_SUMMARY_TOOL]
        super().__init__(
            AgentType.ESCALATION,
            ESCALATION_PROFILE,
            default_tools if tools is None else tools,
            client,
            stats,
        )

    async def handle(self, request: Request) -> AgentResponse:
        """生成会话内交接建议，不创建工单、不调用 LLM、不声明人工已接入。

        Args:
            request: 包含 request_id、消息、意图、紧急度和业务实体的内部请求。

        Returns:
            成功的 `AgentResponse`；`escalate=True` 仅表示建议升级，不代表真实转接。
        """

        started = time.perf_counter()
        self.stats.total += 1
        try:
            intent = request.intent.value if isinstance(request.intent, IntentCategory) else str(request.intent or "other")
            urgency = request.urgency.value if isinstance(request.urgency, UrgencyLevel) else request.urgency or 1
            content = (
                f"已生成交接摘要与升级建议（request_id={request.request_id}，"
                f"原因：{request.message[:120]}，意图：{intent}，紧急度：{urgency}）。"
                "当前仅记录会话上下文，不会创建外部工单。"
            )
            latency_ms = max(0.0, (time.perf_counter() - started) * 1000)
            self.stats.success += 1
            self.stats.total_ms += latency_ms
            return AgentResponse(
                agent_type=self.agent_type,
                content=content,
                success=True,
                latency_ms=latency_ms,
                escalate=True,
            )
        except Exception as exc:
            latency_ms = max(0.0, (time.perf_counter() - started) * 1000)
            self.stats.total_ms += latency_ms
            return AgentResponse(
                agent_type=self.agent_type,
                content=f"Escalation 摘要生成失败: {exc}",
                success=False,
                confidence=0.0,
                latency_ms=latency_ms,
            )


# 初始池每类只放一个 Agent；Step 6.2 再负责并发编排和实例选择。
AGENT_POOL: Dict[AgentType, BaseAgent] = {
    AgentType.GENERAL: GeneralAgent(),
    AgentType.TECHNICAL: TechnicalAgent(),
    AgentType.BILLING: BillingAgent(),
    AgentType.ESCALATION: EscalationAgent(),
}


_GENERAL_INTENTS = frozenset(
    {
        IntentCategory.QUERY,
        IntentCategory.ORDER_STATUS,
        IntentCategory.LOGISTICS,
        IntentCategory.REQUEST,
        IntentCategory.COMPLAINT,
        IntentCategory.GREETING,
        IntentCategory.FEEDBACK,
        IntentCategory.OTHER,
    }
)
_TECHNICAL_INTENTS = frozenset(
    {
        IntentCategory.TECHNICAL,
        IntentCategory.TECHNICAL_LOGIN,
        IntentCategory.TECHNICAL_CRASH,
    }
)
_BILLING_INTENTS = frozenset(
    {
        IntentCategory.BILLING,
        IntentCategory.ACCOUNT,
        IntentCategory.ACCOUNT_SECURITY,
        IntentCategory.REFUND,
        IntentCategory.INVOICE,
        IntentCategory.PAYMENT_ISSUE,
    }
)
_TECHNICAL_KEYWORDS = (
    "崩溃",
    "报错",
    "error",
    "crash",
    "无法登录",
    "登录失败",
    "500",
    "401",
    "403",
    "404",
    "验证码",
    "网络",
)
_BILLING_KEYWORDS = (
    "退款",
    "退货",
    "扣款",
    "发票",
    "账单",
    "支付",
    "订阅",
    "refund",
    "invoice",
    "多扣",
)
_GENERAL_KEYWORDS = ("订单", "物流", "快递", "配送", "会员", "积分", "咨询", "帮助")
CLARIFICATION_PROMPT = "我还不能确定您要处理的是哪类问题。请补充一下是订单物流、退款账单、账户资料，还是技术故障？"


class _DomainScores(dict):
    """保存三域分数，并兼容 `AgentType` 与字符串两种读取方式。

    旧版编排器使用 `AgentType` 作为字典键，而当前模块的测试和诊断输出
    更适合使用稳定的字符串域名；统一内部键为枚举并兼容两种读取方式，
    可以避免迁移期间出现隐蔽的键类型不一致。
    """

    @staticmethod
    def _normalize_key(key: Any) -> Any:
        if isinstance(key, AgentType):
            return key
        try:
            return AgentType(str(key))
        except (TypeError, ValueError):
            return key

    def __getitem__(self, key: Any) -> float:
        return super().__getitem__(self._normalize_key(key))

    def get(self, key: Any, default: Any = None) -> Any:
        return super().get(self._normalize_key(key), default)

    def __contains__(self, key: object) -> bool:
        return super().__contains__(self._normalize_key(key))


def _coerce_intent(value: Any) -> IntentCategory:
    """将识别结果中的字符串或枚举安全归一为 IntentCategory。"""

    if isinstance(value, IntentCategory):
        return value
    try:
        return IntentCategory(str(value))
    except (TypeError, ValueError):
        return IntentCategory.OTHER


def _intent_result_value(result: Any, key: str, default: Any = None) -> Any:
    """兼容 IntentResult dataclass 与测试字典。"""

    if result is None:
        return default
    if isinstance(result, Mapping):
        return result.get(key, default)
    return getattr(result, key, default)


def _domain_scores(
    request: Request,
    intent_result: Optional[IntentResult] = None,
) -> Dict[AgentType, float]:
    """按主意图、关键词和实体为 General/Technical/Billing 计算领域分数。

    Args:
        request: 当前内部请求，提供原始文本、意图和实体。
        intent_result: 可选的识别结果；存在时优先使用其意图与实体。

    Returns:
        以 `AgentType.GENERAL`、`AgentType.TECHNICAL`、`AgentType.BILLING` 为
        key 的三域分数；也兼容用对应字符串读取。
        分数仅用于相对路由排序，不代表概率。
    """

    message = (request.message or "").lower()
    intent = _coerce_intent(
        _intent_result_value(intent_result, "intent", request.intent)
    )
    result_entities = _intent_result_value(intent_result, "entities", None)
    # 请求上的实体是识别器外部补充信息，不能因 IntentResult 使用默认空字典而丢失。
    entities: Dict[str, Any] = dict(request.entities or {})
    if isinstance(result_entities, Mapping):
        entities.update(result_entities)
    scores = _DomainScores(
        {
            AgentType.GENERAL: 0.1,
            AgentType.TECHNICAL: 0.0,
            AgentType.BILLING: 0.0,
        }
    )

    if intent in _GENERAL_INTENTS:
        scores[AgentType.GENERAL] += 0.55
    if intent in _TECHNICAL_INTENTS:
        scores[AgentType.TECHNICAL] += 0.75
    if intent in _BILLING_INTENTS:
        scores[AgentType.BILLING] += 0.75

    # Why：关键词加分上限防止单条长消息重复命中后压过明确的主意图分流。
    technical_hits = sum(1 for keyword in _TECHNICAL_KEYWORDS if keyword in message)
    billing_hits = sum(1 for keyword in _BILLING_KEYWORDS if keyword in message)
    general_hits = sum(1 for keyword in _GENERAL_KEYWORDS if keyword in message)
    scores[AgentType.TECHNICAL] += min(0.45, technical_hits * 0.18)
    scores[AgentType.BILLING] += min(0.45, billing_hits * 0.18)
    scores[AgentType.GENERAL] += min(0.35, general_hits * 0.12)

    if entities.get("error_code"):
        scores[AgentType.TECHNICAL] += 0.2
    if entities.get("amount"):
        scores[AgentType.BILLING] += 0.15
    if entities.get("order_id"):
        scores[AgentType.GENERAL] += 0.1
    return _DomainScores({domain: round(score, 3) for domain, score in scores.items()})


def _agent_for_intent(intent: IntentCategory) -> AgentType:
    """将已确认意图映射到其领域 Agent。"""

    domain = _intent_domain(intent)
    return {
        "general": AgentType.GENERAL,
        "technical": AgentType.TECHNICAL,
        "billing": AgentType.BILLING,
        "escalation": AgentType.ESCALATION,
    }[domain]


def _route_decision(
    request: Request,
    intent_result: Optional[IntentResult] = None,
) -> RoutingDecision:
    """执行安全优先的主 Agent 选择、澄清和显式辅助 Agent 映射。

    Args:
        request: 当前内部请求。
        intent_result: 可选的识别结果；用于置信度、紧急度和 secondary intents。

    Returns:
        `RoutingDecision`。低置信度 OTHER 返回 ``primary_agent=None``，表示只能
        输出澄清问题，绝不应进入任何 Agent 执行任务。
    """

    request_intent = _coerce_intent(request.intent)
    result_intent = _coerce_intent(
        _intent_result_value(intent_result, "intent", request_intent)
    )
    intent = result_intent
    confidence_value = _intent_result_value(
        intent_result, "confidence", request.intent_confidence
    )
    try:
        confidence = float(confidence_value)
    except (TypeError, ValueError):
        confidence = 0.0
    request_urgency = request.urgency
    urgency = _intent_result_value(intent_result, "urgency", request_urgency)
    if not isinstance(urgency, UrgencyLevel):
        try:
            urgency = UrgencyLevel(int(urgency)) if urgency is not None else None
        except (TypeError, ValueError):
            urgency = None

    # Why：CRITICAL 需要先进入升级建议路径，避免领域 Agent 继续执行普通问答。
    # Why：任一层标记为 CRITICAL 都必须升级，防止识别结果覆盖请求侧的安全信号。
    if urgency == UrgencyLevel.CRITICAL or request_urgency == UrgencyLevel.CRITICAL:
        return RoutingDecision(
            primary_agent=AgentType.ESCALATION,
            reason="紧急度为 CRITICAL，触发升级路由",
            confidence=1.0,
        )
    # Why：HUMAN_HANDOFF 是终止型意图，最低 0.8 才能稳定表达升级决策。
    # Why：任一层确认人工意图都不能被普通域路由覆盖，避免越过人机协同边界。
    if intent is IntentCategory.HUMAN_HANDOFF or request_intent is IntentCategory.HUMAN_HANDOFF:
        return RoutingDecision(
            primary_agent=AgentType.ESCALATION,
            reason="意图为 human_handoff，触发升级路由",
            confidence=max(confidence, 0.8),
        )
    # Why：低置信度 OTHER 不应误派发，先澄清可避免错误的业务域执行。
    if intent is IntentCategory.OTHER and len((request.message or "").strip()) > 2 and confidence < 0.5:
        return RoutingDecision(
            primary_agent=None,
            reason=CLARIFICATION_PROMPT,
            confidence=confidence,
        )

    scores = _domain_scores(request, intent_result)
    available = {
        agent_type: score
        for agent_type, score in scores.items()
        if AGENT_POOL.get(agent_type)
    }
    if not available:
        return RoutingDecision(
            primary_agent=AgentType.GENERAL,
            reason="无可用领域 Agent，降级到 GeneralAgent",
            confidence=0.1,
        )
    primary_agent, primary_score = max(available.items(), key=lambda item: item[1])

    if intent_result is not None:
        # 识别器明确返回空列表时也必须尊重它，不能从请求文本再次猜测辅助域。
        secondary_values = _intent_result_value(intent_result, "secondary_intents", [])
        secondary_intents = list(secondary_values or [])
    else:
        secondary_intents = list(request.secondary_intents or [])
    supporting: List[AgentType] = []
    for secondary in secondary_intents:
        target = _agent_for_intent(_coerce_intent(secondary))
        if target in (primary_agent, AgentType.ESCALATION) or target in supporting:
            continue
        if AGENT_POOL.get(target):
            supporting.append(target)
    score_text = ", ".join(
        f"{agent_type.value}={score:.2f}"
        for agent_type, score in sorted(
            available.items(), key=lambda item: item[1], reverse=True
        )
    )
    return RoutingDecision(
        primary_agent=primary_agent,
        supporting_agents=supporting,
        reason=f"intent={intent.value}, primary={primary_agent.value}, scores=[{score_text}]",
        confidence=round(min(primary_score, 1.0), 3),
    )


def _friendly_failure(agent_type: AgentType, latency_ms: float = 0.0) -> AgentResponse:
    """构造不暴露内部异常细节的友好失败响应。"""

    return AgentResponse(
        agent_type=agent_type,
        content="当前暂时无法完成该请求，请稍后重试。",
        success=False,
        confidence=0.0,
        latency_ms=max(0.0, latency_ms),
    )


async def _execute(request: Request, agent_type: AgentType) -> AgentResponse:
    """执行指定 Agent，并对技术/账单失败自动降级到 General。

    Args:
        request: 当前内部请求。
        agent_type: 路由选定的 Agent 类型。

    Returns:
        目标 Agent 或 General fallback 的响应；异常被转换为友好失败结果。
    """

    agent = AGENT_POOL.get(agent_type)
    if agent is None and agent_type in (AgentType.TECHNICAL, AgentType.BILLING):
        logger.warning("%s Agent 不可用，直接尝试 General fallback", agent_type.value)
        agent = AGENT_POOL.get(AgentType.GENERAL)
        agent_type = AgentType.GENERAL
    if agent is None:
        return _friendly_failure(agent_type)

    started = time.perf_counter()
    try:
        response = await agent.handle(request)
    except Exception as exc:
        logger.warning("%s Agent 执行异常：%s", agent_type.value, exc)
        response = _friendly_failure(
            agent_type, (time.perf_counter() - started) * 1000
        )

    if response.success or agent_type in (AgentType.GENERAL, AgentType.ESCALATION):
        return response if response.success else _friendly_failure(agent_type, response.latency_ms)

    # Why：Technical/Billing 的局部失败必须给用户一个通用兜底，避免专属 Agent 故障扩大为全局失败。
    fallback = AGENT_POOL.get(AgentType.GENERAL)
    if fallback is None or fallback is agent:
        return _friendly_failure(agent_type, response.latency_ms)
    try:
        fallback_response = await fallback.handle(request)
        if fallback_response.success:
            return fallback_response
        return _friendly_failure(AgentType.GENERAL, fallback_response.latency_ms)
    except Exception as exc:
        logger.warning("General fallback 执行异常：%s", exc)
        return _friendly_failure(AgentType.GENERAL, response.latency_ms)


@dataclass
class OrchestratorResult:
    request_id: str
    response: str
    agent_type: AgentType
    intent: Optional[IntentCategory]
    escalated: bool = False
    agent_recommended_handoff: bool = False
    latency_ms: float = 0.0
    agent_types: List[AgentType] = field(default_factory=list)
    primary_agent: Optional[AgentType] = None
    supporting_agents: List[AgentType] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    tool_traces: List[Dict[str, Any]] = field(default_factory=list)
    routing_reason: str = ""
    routing_confidence: float = 0.0
    intent_diagnostics: Dict[str, Any] = field(default_factory=dict)
    secondary_intents: List[IntentCategory] = field(default_factory=list)
    knowledge_required: bool = False
