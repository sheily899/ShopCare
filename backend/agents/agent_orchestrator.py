"""ShopCare 领域 Agent 契约、白名单执行边界与确定性升级 Agent。

本模块属于 Agent 能力层：定义内部 dataclass 契约、四类领域 Profile，以及
受工具白名单约束的 Anthropic Tool Use 执行基类。它不实现多 Agent 路由、结果
合成或任何真实订单、支付、工单和人工坐席接口；这些能力由后续编排步骤负责。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import time
import uuid
from abc import ABC
from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Tuple

from intent.intent_recognizer import (
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
        skill_manager: Optional[Any] = None,
    ) -> None:
        self.agent_type = agent_type
        self.profile = profile
        self.tools: List[AgentToolSpec] = list(tools or [])
        self.client = client
        self.stats = stats or AgentStats()
        self.skill_manager = skill_manager

    def _system_prompt(self) -> str:
        """合并领域 Profile 与当前 Agent 可见的本地 Skill Prompt。

        Returns:
            供 Anthropic 请求使用的系统提示词；Skill 管理器不可用时仅返回
            固定的领域 Profile，不影响 Agent 的基础执行。
        """

        if self.skill_manager is None:
            return self.profile.mission
        try:
            skill_prompt = self.skill_manager.prompt_for_agent(self.agent_type.value)
        except Exception as exc:  # pragma: no cover - 仅在可选 Skill 管理器异常时触发
            logger.warning("Skill Prompt 注入失败，将使用 Agent Profile：%s", exc)
            return self.profile.mission
        if not skill_prompt:
            return self.profile.mission
        return f"{self.profile.mission}\n\n{skill_prompt}"

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
        tools_override: Optional[List[AgentToolSpec]] = None,
    ) -> str:
        """执行最多三轮 Tool Use，并只暴露当前 Agent 的工具白名单。"""

        if self.client is None:
            raise RuntimeError("Anthropic client 未配置")
        user_content = request.message
        if request.context:
            user_content = f"[请求上下文]\n{request.context}\n\n用户问题：{request.message}"
        messages: List[Dict[str, Any]] = [{"role": "user", "content": user_content}]
        visible_tools = self.tools if tools_override is None else tools_override
        declarations = [tool.to_anthropic_tool() for tool in visible_tools]
        tool_map = {tool.name: tool for tool in visible_tools}
        model = self.profile.model or "claude-3-5-sonnet-20241022"

        for round_index in range(self.MAX_TOOL_ROUNDS):
            response = await self.client.messages.create(
                model=model,
                max_tokens=self.profile.max_tokens,
                temperature=self.profile.temperature,
                # Why：领域 Agent 使用 Tool Use 时关闭 Thinking，避免兼容模型拒绝工具调用。
                extra_body={"thinking": {"type": "disabled"}},
                system=self._system_prompt(),
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
                tool_started = time.perf_counter()
                tool_result = await tool.execute(tool_input, context=self._tool_context(request))
                tools_used.append(name)
                tool_success = bool(
                    tool_result.get("success", False)
                    if isinstance(tool_result, Mapping)
                    else True
                )
                tool_traces.append(
                    {
                        "round": round_index + 1,
                        "tool": name,
                        "tool_name": name,
                        "input": tool_input,
                        "latency_ms": round(
                            max(0.0, (time.perf_counter() - tool_started) * 1000),
                            3,
                        ),
                        "status": "success" if tool_success else "failed",
                        "success": tool_success,
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

    async def handle(
        self,
        request: Request,
        *,
        tools_override: Optional[List[AgentToolSpec]] = None,
    ) -> AgentResponse:
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
            content = await self._run_llm(
                request,
                tools_used,
                tool_traces,
                tools_override=tools_override,
            )
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
        skill_manager: Optional[Any] = None,
    ) -> None:
        default_tools = list(general_tools().values())
        default_tools.insert(0, SEARCH_KNOWLEDGE_BASE_TOOL)
        super().__init__(
            AgentType.GENERAL,
            GENERAL_PROFILE,
            default_tools if tools is None else tools,
            client,
            stats,
            skill_manager,
        )


class TechnicalAgent(BaseAgent):
    """Technical Agent，隔离错误码查询与低风险诊断工具。"""

    def __init__(
        self,
        client: Optional[Any] = None,
        tools: Optional[List[AgentToolSpec]] = None,
        stats: Optional[AgentStats] = None,
        skill_manager: Optional[Any] = None,
    ) -> None:
        default_tools = [SEARCH_KNOWLEDGE_BASE_TOOL, *technical_tools().values()]
        super().__init__(
            AgentType.TECHNICAL,
            TECHNICAL_PROFILE,
            default_tools if tools is None else tools,
            client,
            stats,
            skill_manager,
        )


class BillingAgent(BaseAgent):
    """Billing Agent，隔离账单字段核验与金额算术工具。"""

    def __init__(
        self,
        client: Optional[Any] = None,
        tools: Optional[List[AgentToolSpec]] = None,
        stats: Optional[AgentStats] = None,
        skill_manager: Optional[Any] = None,
    ) -> None:
        default_tools = [SEARCH_KNOWLEDGE_BASE_TOOL, *billing_tools().values()]
        super().__init__(
            AgentType.BILLING,
            BILLING_PROFILE,
            default_tools if tools is None else tools,
            client,
            stats,
            skill_manager,
        )


class EscalationAgent(BaseAgent):
    """Escalation Agent，以确定性摘要表达升级建议而不调用 LLM。"""

    def __init__(
        self,
        client: Optional[Any] = None,
        tools: Optional[List[AgentToolSpec]] = None,
        stats: Optional[AgentStats] = None,
        skill_manager: Optional[Any] = None,
    ) -> None:
        default_tools = [SEARCH_KNOWLEDGE_BASE_TOOL, CREATE_HANDOFF_SUMMARY_TOOL]
        super().__init__(
            AgentType.ESCALATION,
            ESCALATION_PROFILE,
            default_tools if tools is None else tools,
            client,
            stats,
            skill_manager,
        )

    async def handle(
        self,
        request: Request,
        *,
        tools_override: Optional[List[AgentToolSpec]] = None,
    ) -> AgentResponse:
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


KnowledgeSearch = Callable[[str, int], Awaitable[Any] | Any]


def _dedupe_preserving_order(values: List[str]) -> List[str]:
    """按首次出现顺序合并字符串列表。"""

    return list(dict.fromkeys(value for value in values if value))


def _extract_retrieval_payload(raw: Any) -> Tuple[bool, List[Dict[str, Any]], Optional[str]]:
    """兼容列表、ToolResult 和结构化字典形式的检索返回值。

    Args:
        raw: 知识库 callback 或 Tool handler 返回的对象。

    Returns:
        `(success, chunks, error)`，其中 chunks 只保留字典切片，不伪造缺失结果。
    """

    if isinstance(raw, Mapping):
        success = bool(raw.get("success", True))
        error = raw.get("error")
        data = raw.get("data", raw.get("results", []))
    else:
        success = bool(getattr(raw, "success", True))
        error = getattr(raw, "error", None)
        data = getattr(raw, "data", raw)
    if not success:
        return False, [], str(error or "知识库检索失败")
    if isinstance(data, Mapping):
        data = data.get("results", data.get("chunks", []))
    chunks = [dict(item) for item in data if isinstance(item, Mapping)] if isinstance(data, list) else []
    return True, chunks, None


def _format_retrieval_context(chunks: List[Dict[str, Any]]) -> str:
    """把知识库切片格式化成只读 Prompt 上下文。"""

    if not chunks:
        return "知识库本次未召回可用切片，请基于已有信息谨慎回答。"
    lines = ["知识库只读检索结果（仅作为参考证据，不代表执行了外部业务操作）："]
    for index, chunk in enumerate(chunks, start=1):
        title = str(chunk.get("title") or "未命名切片")
        content = str(chunk.get("content") or "").strip()
        score = chunk.get("score")
        score_text = f"，score={score}" if score is not None else ""
        lines.append(f"[{index}] {title}{score_text}\n{content}")
    return "\n\n".join(lines)


def _retrieval_summary(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """生成 Trace 中的轻量切片摘要，避免复制无关大字段。"""

    return [
        {
            "title": str(chunk.get("title") or ""),
            "source_chunk_id": str(chunk.get("source_chunk_id") or ""),
            "score": chunk.get("score"),
            "content_preview": str(chunk.get("content") or "")[:160],
        }
        for chunk in chunks
    ]


async def _call_knowledge_search(
    searcher: Optional[Any],
    query: str,
    top_k: int,
    request: Request,
) -> Any:
    """执行一次可注入的知识搜索，并将同步实现移出事件循环。"""

    params = {"query": query, "top_k": top_k}
    if searcher is None:
        return await SEARCH_KNOWLEDGE_BASE_TOOL.execute(params)
    if isinstance(searcher, AgentToolSpec):
        return await searcher.execute(params, context={"request_id": request.request_id})
    if callable(searcher):
        call = getattr(searcher, "__call__", searcher)
        if inspect.iscoroutinefunction(searcher) or inspect.iscoroutinefunction(call):
            result = searcher(query, top_k=top_k)
        else:
            result = await asyncio.to_thread(searcher, query, top_k=top_k)
    elif hasattr(searcher, "search"):
        # KnowledgeBase.search 是同步方法，在线程中执行，避免阻塞其他 Agent。
        result = await asyncio.to_thread(searcher.search, query, top_k=top_k)
    else:
        raise TypeError("knowledge_search 必须是可调用对象或提供 search 方法")
    if inspect.isawaitable(result):
        return await result
    return result


async def _prepare_agent_request(
    request: Request,
    agent_type: AgentType,
    searcher: Optional[Any],
) -> Tuple[Request, Optional[Dict[str, Any]], Optional[List[AgentToolSpec]]]:
    """为单个 Agent 完成一次知识前置检索和 search 工具隔离。"""

    if not request.knowledge_required:
        return request, None, None
    scoped_queries = request.intent_diagnostics.get("scoped_retrieval_queries")
    scoped_query_value = request.intent_diagnostics.get("scoped_retrieval_query")
    if isinstance(scoped_queries, Mapping):
        scoped_query_value = scoped_queries.get(agent_type.value, scoped_query_value)
    scoped_query = str(scoped_query_value or request.message or "")
    # Why：每个实际 Agent 固定只预检索一次且使用 top_k=5，避免重复召回和上下文膨胀。
    recall_k = 5
    started = time.perf_counter()
    try:
        raw = await _call_knowledge_search(searcher, scoped_query, recall_k, request)
        success, chunks, error = _extract_retrieval_payload(raw)
    except Exception as exc:
        success, chunks, error = False, [], str(exc)
    latency_ms = max(0.0, (time.perf_counter() - started) * 1000)
    trace: Dict[str, Any] = {
        "tool_name": "search_knowledge_base",
        "input": {"query": scoped_query, "top_k": recall_k},
        "latency_ms": round(latency_ms, 3),
        "status": "success" if success else "failed",
        "retrieval_stage": "knowledge_prefetch",
        "agent_type": agent_type.value,
        "retrieved_chunks": chunks,
        "chunk_summary": _retrieval_summary(chunks),
    }
    if error:
        trace["error"] = error
    context_text = _format_retrieval_context(chunks)
    merged_context = (
        f"{request.context}\n\n{context_text}" if request.context else context_text
    )
    prepared_request = replace(request, context=merged_context)
    return prepared_request, trace, None


def _agent_tools_without_search(agent: Any) -> Optional[List[AgentToolSpec]]:
    """返回移除共享知识检索工具后的 Agent 工具白名单。"""

    tools = getattr(agent, "tools", None)
    if tools is None:
        return None
    return [tool for tool in tools if tool.name != "search_knowledge_base"]


async def _invoke_agent(
    agent: Any,
    request: Request,
    tools_override: Optional[List[AgentToolSpec]],
) -> AgentResponse:
    """调用真实 BaseAgent 或兼容旧测试替身的 `handle` 方法。"""

    if tools_override is not None and isinstance(agent, BaseAgent):
        parameters = inspect.signature(agent.handle).parameters
        if "tools_override" in parameters:
            return await agent.handle(request, tools_override=tools_override)
    return await agent.handle(request)


def _attach_trace(
    response: AgentResponse,
    retrieval_trace: Optional[Dict[str, Any]],
) -> AgentResponse:
    """把检索 Trace 与 Agent 自身 Trace 合并，并保持工具首次出现顺序。"""

    if retrieval_trace is None:
        return response
    tools_used = _dedupe_preserving_order(
        ["search_knowledge_base", *response.tools_used]
    )
    traces = [retrieval_trace, *response.tool_traces]
    return replace(response, tools_used=tools_used, tool_traces=traces)


async def _execute_once(
    request: Request,
    agent_type: AgentType,
    agent: Any,
    searcher: Optional[Any],
) -> AgentResponse:
    """执行单 Agent 一次，并在其响应中附加知识检索 Trace。"""

    prepared_request, retrieval_trace, _ = await _prepare_agent_request(
        request, agent_type, searcher
    )
    tools_override = _agent_tools_without_search(agent) if retrieval_trace else None
    try:
        response = await _invoke_agent(agent, prepared_request, tools_override)
    except Exception as exc:
        logger.warning("%s Agent 执行异常：%s", agent_type.value, exc)
        response = _friendly_failure(
            agent_type,
            retrieval_trace["latency_ms"] if retrieval_trace else 0.0,
        )
    return _attach_trace(response, retrieval_trace)


def _resolve_agent(pool: Mapping[AgentType, Any], agent_type: AgentType) -> Any:
    """从单实例或旧版同类 Agent 列表中取出一个可执行实例。"""

    value = pool.get(agent_type)
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


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
    *,
    agent_pool: Optional[Mapping[AgentType, Any]] = None,
) -> RoutingDecision:
    """执行安全优先的主 Agent 选择、澄清和显式辅助 Agent 映射。

    Args:
        request: 当前内部请求。
        intent_result: 可选的识别结果；用于置信度、紧急度和 secondary intents。
        agent_pool: 可选的可用 Agent 池；未传入时使用模块级默认池。

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

    pool = AGENT_POOL if agent_pool is None else agent_pool
    scores = _domain_scores(request, intent_result)
    available = {
        agent_type: score
        for agent_type, score in scores.items()
        if pool.get(agent_type)
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
        if pool.get(target):
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


class ResponseComposer:
    """把多个领域 Agent 的成功结果合成为面向客户的中文回复。

    Args:
        client: 可选的 Anthropic 异步客户端；未注入时多结果合成安全降级。
        model: 合成模型名称；未提供时使用项目默认模型。

    Returns:
        `compose()` 返回纯客户可见文本；LLM 失败时返回确定性拼接文本。

    该类只负责语义合并，不执行知识检索、订单写操作、退款确认或人工坐席转接。
    """

    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
    DEFAULT_TEMPERATURE = 0.1
    DEFAULT_MAX_TOKENS = 1000

    def __init__(self, client: Optional[Any] = None, model: Optional[str] = None) -> None:
        self.client = client
        self.model = model or self.DEFAULT_MODEL
        self.temperature = self._read_float_env(
            "CUSTOMEROPS_COMPOSER_TEMPERATURE", self.DEFAULT_TEMPERATURE
        )
        self.max_tokens = self._read_int_env(
            "CUSTOMEROPS_COMPOSER_MAX_TOKENS", self.DEFAULT_MAX_TOKENS
        )

    @staticmethod
    def _read_float_env(name: str, default: float) -> float:
        """读取合成温度配置，非法值回退到锁定默认值。"""

        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            value = float(raw)
            return value if 0.0 <= value <= 1.0 else default
        except (TypeError, ValueError):
            logger.warning("ResponseComposer 配置 %s 无法解析，回退默认值 %s", name, default)
            return default

    @staticmethod
    def _read_int_env(name: str, default: int) -> int:
        """读取最大输出 Token 配置，非法值回退到锁定默认值。"""

        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            value = int(raw)
            return value if value > 0 else default
        except (TypeError, ValueError):
            logger.warning("ResponseComposer 配置 %s 无法解析，回退默认值 %s", name, default)
            return default

    @staticmethod
    def _successful(responses: List[AgentResponse]) -> List[AgentResponse]:
        """过滤成功且有可见文本的分支，避免把失败详情交给合成模型。"""

        return [
            response
            for response in responses
            if isinstance(response, AgentResponse)
            and response.success
            and bool((response.content or "").strip())
        ]

    @staticmethod
    def _fallback(
        successful: List[AgentResponse],
        primary_agent: Optional[AgentType] = None,
    ) -> str:
        """按固定主次顺序拼接合成失败时的安全回复。"""

        if not successful:
            return "抱歉，所有 Agent 均处理失败。"
        primary = next(
            (item for item in successful if item.agent_type is primary_agent),
            successful[0],
        )
        supporting = [item for item in successful if item is not primary]
        output = primary.content
        for support in supporting:
            output += f"\n\n补充说明：\n{support.content}"
        return output

    @staticmethod
    def _extract_text(response: Any) -> str:
        """从 Anthropic 响应中提取文本块。"""

        text_parts = [
            str(_mapping_value(block, "text", ""))
            for block in _response_blocks(response)
            if _mapping_value(block, "type") == "text"
        ]
        return "\n".join(part for part in text_parts if part).strip()

    def _build_prompt(
        self,
        request: Request,
        primary: AgentResponse,
        supporting: List[AgentResponse],
    ) -> str:
        """构造只包含当前请求与分支结论的合成提示。"""

        support_text = "\n\n".join(
            f"补充分支 {index}（内部类型：{response.agent_type.value}，不要在输出中提及该类型）：\n{response.content}"
            for index, response in enumerate(supporting, start=1)
        )
        return (
            "你是 ShopCare 的客户回复合成器。请以主处理分支的结论为主体，"
            "吸收补充分支中不重复且不冲突的信息，删除重复开场白并平滑处理冲突。\n"
            "严禁无中生有编造订单发货、退款成功或后台查询结果；只输出面向客户的纯净中文段落，"
            "不得暴露内部 Agent 专有名词。\n\n"
            f"用户问题：{request.message}\n\n"
            f"主处理分支类型（仅供内部排序，不要在输出中提及）：{primary.agent_type.value}\n"
            f"主处理分支结论：\n{primary.content}\n\n"
            f"补充信息：\n{support_text}"
        )

    async def compose(
        self,
        request: Request,
        responses: List[AgentResponse],
        primary_agent: Optional[AgentType] = None,
    ) -> str:
        """按成功分支数量合并客户回复，并在失败时确定性降级。

        Args:
            request: 当前用户请求，仅用于合成 Prompt 的问题上下文。
            responses: Fan-out 返回的 Agent 响应列表。
            primary_agent: 路由选定的主 Agent，用于确定主次顺序。

        Returns:
            单结果直接文本、多结果合成文本或固定降级拼接文本。
        """

        successful = self._successful(responses)
        if not successful:
            return self._fallback(successful, primary_agent)
        if len(successful) == 1:
            # Why：单分支无需再次请求模型，避免额外 Token 与合成延迟。
            return successful[0].content

        primary = next(
            (item for item in successful if item.agent_type is primary_agent),
            successful[0],
        )
        supporting = [item for item in successful if item is not primary]
        if self.client is None:
            logger.warning("ResponseComposer 未配置 Anthropic client，使用确定性拼接降级")
            return self._fallback(successful, primary_agent)
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=(
                    "只输出面向客户的中文回复。不要输出分析过程、内部 Agent 名称或未被证实的外部业务结果。"
                ),
                messages=[
                    {"role": "user", "content": self._build_prompt(request, primary, supporting)}
                ],
            )
            text = self._extract_text(response)
            if not text:
                raise ValueError("合成模型未返回文本")
            return text
        except Exception as exc:
            # Why：合成器故障不能让已成功的领域答案白屏，确定性拼接保留事实信息。
            logger.warning("ResponseComposer 合成失败，降级拼接：%s", exc)
            return self._fallback(successful, primary_agent)


async def _execute(
    request: Request,
    agent_type: AgentType,
    *,
    knowledge_search: Optional[Any] = None,
    agent_pool: Optional[Mapping[AgentType, Any]] = None,
) -> AgentResponse:
    """执行指定 Agent，并对技术/账单失败自动降级到 General。

    Args:
        request: 当前内部请求。
        agent_type: 路由选定的 Agent 类型。
        knowledge_search: 可注入的知识检索函数或提供 `search` 方法的对象。
        agent_pool: 可选 Agent 池；未传入时使用模块级默认池。

    Returns:
        目标 Agent 或 General fallback 的响应；异常被转换为友好失败结果。
    """

    pool = AGENT_POOL if agent_pool is None else agent_pool
    agent = _resolve_agent(pool, agent_type)
    if agent is None and agent_type in (AgentType.TECHNICAL, AgentType.BILLING):
        logger.warning("%s Agent 不可用，直接尝试 General fallback", agent_type.value)
        agent = _resolve_agent(pool, AgentType.GENERAL)
        agent_type = AgentType.GENERAL
    if agent is None:
        return _friendly_failure(agent_type)

    response = await _execute_once(request, agent_type, agent, knowledge_search)

    if response.success or agent_type in (AgentType.GENERAL, AgentType.ESCALATION):
        return response if response.success else _friendly_failure(agent_type, response.latency_ms)

    # Why：Technical/Billing 的局部失败必须给用户一个通用兜底，避免专属 Agent 故障扩大为全局失败。
    fallback = _resolve_agent(pool, AgentType.GENERAL)
    if fallback is None or fallback is agent:
        return _friendly_failure(agent_type, response.latency_ms)
    fallback_response = await _execute_once(
        request, AgentType.GENERAL, fallback, knowledge_search
    )
    if fallback_response.success:
        return replace(
            fallback_response,
            tools_used=_dedupe_preserving_order(
                [*response.tools_used, *fallback_response.tools_used]
            ),
            tool_traces=[*response.tool_traces, *fallback_response.tool_traces],
        )
    failed = _friendly_failure(AgentType.GENERAL, fallback_response.latency_ms)
    return replace(
        failed,
        tools_used=_dedupe_preserving_order(
            [*response.tools_used, *fallback_response.tools_used]
        ),
        tool_traces=[*response.tool_traces, *fallback_response.tool_traces],
    )


@dataclass(frozen=True)
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


async def _fan_out(
    request: Request,
    decision: RoutingDecision,
    *,
    knowledge_search: Optional[Any] = None,
    agent_pool: Optional[Mapping[AgentType, Any]] = None,
) -> List[AgentResponse]:
    """按路由决策并发执行主/辅助 Agent，并隔离单分支异常。

    Args:
        request: 当前内部请求。
        decision: Step 6.1 产出的主 Agent 与辅助 Agent 决策。
        knowledge_search: 可选知识检索 callback；业务请求会为每个 Agent 调用一次。
        agent_pool: 可选 Agent 池。

    Returns:
        仅包含成功的 `AgentResponse`；单个协程异常会被记录并跳过。
    """

    if decision.primary_agent is None:
        return []
    agent_types = [decision.primary_agent, *decision.supporting_agents]
    execute_parameters = inspect.signature(_execute).parameters
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in execute_parameters.values()
    )
    execute_kwargs: Dict[str, Any] = {}
    if accepts_kwargs or "knowledge_search" in execute_parameters:
        execute_kwargs["knowledge_search"] = knowledge_search
    if accepts_kwargs or "agent_pool" in execute_parameters:
        execute_kwargs["agent_pool"] = agent_pool
    # Why：Fan-out 使用 return_exceptions=True，保证辅助 Agent 崩溃不丢失主结果。
    tasks = [
        _execute(request, agent_type, **execute_kwargs)
        for agent_type in agent_types
    ]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    successful: List[AgentResponse] = []
    for agent_type, response in zip(agent_types, responses):
        if isinstance(response, Exception):
            logger.error("%s Agent Fan-out 异常：%s", agent_type.value, response)
            continue
        if isinstance(response, AgentResponse) and response.success:
            successful.append(response)
    return successful


class AgentOrchestrator:
    """负责 Fan-out、知识预检索、结果合成和 Trace 聚合的轻量编排入口。

    Args:
        agent_pool: 可注入的 Agent 映射，默认使用模块级四类 Agent 池。
        knowledge_search: 可注入的知识检索函数或提供 `search` 方法的对象。
        knowledge_base: `knowledge_search` 的兼容别名。
        composer: 可注入的结果合成器；未提供时使用无网络的 `ResponseComposer`。

    Returns:
        `run()` 和 `run_parallel()` 返回 `OrchestratorResult`。本阶段不包含
        订单写操作、退款确认或真实人工坐席转接。
    """

    def __init__(
        self,
        agent_pool: Optional[Mapping[AgentType, Any]] = None,
        knowledge_search: Optional[Any] = None,
        knowledge_base: Optional[Any] = None,
        composer: Optional[ResponseComposer] = None,
    ) -> None:
        self.agent_pool: Dict[AgentType, Any] = dict(
            AGENT_POOL if agent_pool is None else agent_pool
        )
        self.knowledge_search = knowledge_search or knowledge_base
        self.composer = composer or ResponseComposer()
        self._recent_traces: deque[Dict[str, Any]] = deque(maxlen=200)

    def _decision(self, request: Request) -> RoutingDecision:
        """使用当前注入 Agent 池计算路由决策。"""

        return _route_decision(request, agent_pool=self.agent_pool)

    async def _compose_responses(
        self,
        request: Request,
        responses: List[AgentResponse],
        primary_agent: Optional[AgentType],
    ) -> str:
        """调用注入合成器，并兼容只接受两个位置参数的测试替身。"""

        compose = getattr(self.composer, "compose", None)
        if not callable(compose):
            logger.warning("ResponseComposer 未提供 compose 方法，使用确定性拼接降级")
            return ResponseComposer._fallback(responses, primary_agent)
        try:
            parameters = inspect.signature(compose).parameters
            accepts_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
            if accepts_kwargs or "primary_agent" in parameters:
                return await compose(request, responses, primary_agent=primary_agent)
            return await compose(request, responses)
        except Exception as exc:
            # Why：合成器属于可选增强层，异常必须回退到事实保真的主次拼接。
            logger.warning("编排层调用 ResponseComposer 失败，降级拼接：%s", exc)
            return ResponseComposer._fallback(responses, primary_agent)

    async def run_parallel(
        self,
        request: Request,
        decision: RoutingDecision,
    ) -> OrchestratorResult:
        """并发执行路由中的 Agent，合成客户回复并聚合工具 Trace。

        Args:
            request: 当前内部请求。
            decision: 主/辅助 Agent 决策。

        Returns:
            包含最终客户文本、成功 Agent 列表、去重工具清单和检索阶段 Trace 的结果。
        """

        started = time.perf_counter()
        valid = await _fan_out(
            request,
            decision,
            knowledge_search=self.knowledge_search,
            agent_pool=self.agent_pool,
        )
        # Fan-out 只接受结构完整的响应；成功筛选再次放在合成入口，避免测试替身或未来扩展泄漏失败文案。
        successful = ResponseComposer._successful(valid)
        tools_used = _dedupe_preserving_order(
            [tool_name for response in successful for tool_name in response.tools_used]
        )
        tool_traces = [trace for response in successful for trace in response.tool_traces]
        retrieval_stage = [
            trace
            for trace in tool_traces
            if isinstance(trace, Mapping) and trace.get("retrieval_stage")
        ]
        diagnostics = dict(request.intent_diagnostics or {})
        diagnostics["retrieval_stage"] = retrieval_stage
        primary_response = next(
            (response for response in successful if response.agent_type is decision.primary_agent),
            successful[0] if successful else None,
        )
        if primary_response is None:
            # Why：所有分支失败时返回固定安全提示，不把异常详情暴露给用户。
            response_text = "抱歉，所有 Agent 均处理失败。"
            result_agent_type = decision.primary_agent or AgentType.GENERAL
            agent_types: List[AgentType] = []
        else:
            response_text = await self._compose_responses(
                request,
                successful,
                decision.primary_agent,
            )
            result_agent_type = primary_response.agent_type
            agent_types = [response.agent_type for response in successful]
        result = OrchestratorResult(
            request_id=request.request_id,
            response=response_text,
            agent_type=result_agent_type,
            intent=request.intent,
            latency_ms=max(0.0, (time.perf_counter() - started) * 1000),
            agent_types=agent_types,
            primary_agent=decision.primary_agent,
            supporting_agents=list(decision.supporting_agents),
            tools_used=tools_used,
            tool_traces=tool_traces,
            routing_reason=decision.reason,
            routing_confidence=decision.confidence,
            intent_diagnostics=diagnostics,
            secondary_intents=list(request.secondary_intents),
            knowledge_required=request.knowledge_required,
            escalated=(
                request.urgency is UrgencyLevel.CRITICAL
                or request.intent is IntentCategory.HUMAN_HANDOFF
            ),
            agent_recommended_handoff=any(response.escalate for response in successful),
        )
        self._recent_traces.append(
            {
                "request_id": result.request_id,
                "retrieval_stage": retrieval_stage,
                "tools_used": list(result.tools_used),
                "tool_traces": list(result.tool_traces),
                "executed_agents": [agent.value for agent in result.agent_types],
            }
        )
        return result

    async def run(self, request: Request) -> OrchestratorResult:
        """计算路由并进入 Step 6.2 Fan-out 主链路。"""

        decision = self._decision(request)
        if decision.primary_agent is None:
            return OrchestratorResult(
                request_id=request.request_id,
                response=decision.reason,
                agent_type=AgentType.GENERAL,
                intent=request.intent,
                primary_agent=None,
                routing_reason=decision.reason,
                routing_confidence=decision.confidence,
                intent_diagnostics=dict(request.intent_diagnostics),
                secondary_intents=list(request.secondary_intents),
                knowledge_required=request.knowledge_required,
            )
        return await self.run_parallel(request, decision)

    def get_recent_traces(self, limit: int = 20) -> List[Dict[str, Any]]:
        """返回最近的内存 Trace 快照，不持久化任何敏感凭证。"""

        bounded = max(1, min(int(limit or 20), len(self._recent_traces)))
        return list(reversed(list(self._recent_traces)[-bounded:]))
