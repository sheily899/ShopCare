"""ShopCare 领域 Agent 契约、白名单执行边界与确定性升级 Agent。

本模块属于 Agent 能力层：定义内部 dataclass 契约、四类领域 Profile，以及
受工具白名单约束的 Anthropic Tool Use 执行基类。它不实现多 Agent 路由、结果
合成或任何真实订单、支付、工单和人工坐席接口；这些能力由后续编排步骤负责。
"""

from __future__ import annotations

import json
import time
import uuid
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple

from core.intent_recognizer import IntentCategory, UrgencyLevel
from agents.tools import (
    AgentToolSpec,
    CREATE_HANDOFF_SUMMARY_TOOL,
    billing_tools,
    general_tools,
    make_tool,
    technical_tools,
)


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
    primary_agent: AgentType
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
