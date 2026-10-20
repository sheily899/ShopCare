"""ShopCare 领域 Agent 契约、白名单执行边界与确定性升级 Agent。

本模块属于 Agent 能力层：定义内部 dataclass 契约、三个美妆业务 Profile，以及
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
    beauty_tools,
    make_tool,
)
from agents.handoff import build_handoff_payload
from agents.customer_tone import customer_friendly_response
from agents.response_guard import (
    filter_redundant_field_questions,
    filter_repeated_answer_content,
    guard_response,
)
from agents.case_facts import CaseFacts
from handoff.config import load_handoff_urgency_config
from handoff.urgency import score_temporary_urgency
from intent.beauty_intent_ticket_registry import (
    BEAUTY_INTENT_TICKET_BY_ID,
    policy_for_intents,
    should_prefetch_knowledge,
)
from intent.slot_registry import get_slot


logger = logging.getLogger(__name__)

_KNOWLEDGE_UNAVAILABLE_RESPONSE = (
    "有一部分问题暂时没有检索到可确认的资料，我不凭猜测给您结论。"
    "您可以稍后再试，或联系人工客服继续核对。"
)

_CUSTOMER_CONTINUITY_RULES = (
    "持续对话时，先读取最近对话和已知事实；只回答消费者本轮新增或尚未解决的问题，不重复已解释的内容或已回答的问题。"
    "安全提醒仅在首次出现、风险新发或加重、或消费者再次询问安全处理时完整说明；此前已说明且风险未变时，不重复整段提醒。"
)

_UNANSWERED_FIELD_VALUES = frozenset({
    "", "unknown", "null", "none", "未知", "不清楚", "不确定", "不知道", "未提供", "待确认"
})


def build_business_field_state(request: "Request") -> Dict[str, Any]:
    """按本轮全部业务意图汇总已知事实及注册表要求的缺失字段。"""

    intent_ids = _all_request_intent_ids(request)
    contracts = [BEAUTY_INTENT_TICKET_BY_ID[item] for item in intent_ids]
    blocking_fields = list(dict.fromkeys(
        field for contract in contracts for field in contract.get("blocking_fields", [])
    ))
    facts = CaseFacts.from_dict(request.case_facts)
    effective_values = facts.effective_values(request.entities)

    def has_value(values: Any) -> bool:
        items = values if isinstance(values, (list, tuple, set)) else [values]
        return any(
            value is not None
            and str(value).strip().casefold() not in _UNANSWERED_FIELD_VALUES
            for value in items
        )

    relevant_fields = list(dict.fromkeys(
        [*blocking_fields]
        + [field for contract in contracts for field in contract.get("ticket_fields", [])]
        + list(request.entities or {})
        + list(facts.facts)
    ))
    known_fields = {
        field: list(effective_values.get(field, []))
        for field in relevant_fields
        if has_value(effective_values.get(field, []))
    }
    missing_fields = [field for field in blocking_fields if field not in known_fields]
    missing_text_fields: list[str] = []
    missing_enum_fields: list[str] = []
    for field in missing_fields:
        slot = get_slot(field)
        if slot and slot.get("type") == "enum":
            missing_enum_fields.append(field)
        else:
            missing_text_fields.append(field)
    return {
        "intent_ids": intent_ids,
        "known_fields": known_fields,
        "blocking_fields": blocking_fields,
        "missing_fields": missing_fields,
        "missing_text_fields": missing_text_fields,
        "askable_text_fields": missing_text_fields[:1],
        "missing_enum_fields": missing_enum_fields,
    }


def _customer_safe_text(text: str, request: Request) -> str:
    """兼容旧入口；不重写模型正文，安全约束由 Prompt 与 response_guard 负责。"""
    del request
    return str(text or "").strip()


class AgentType(str, Enum):
    INQUIRY = "consult"
    FULFILLMENT = "order"
    REFUND_RISK = "after_sales"
    # 兼容旧代码引用；对外序列化值已统一为美妆业务名称。
    GENERAL = "consult"
    TECHNICAL = "order"
    BILLING = "after_sales"
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
    usage: Dict[str, int] = field(default_factory=dict)
    case_facts: Dict[str, Any] = field(default_factory=dict)


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
    sub_requests: List[Dict[str, Any]] = field(default_factory=list)
    case_facts: Dict[str, Any] = field(default_factory=dict)
    requires_tool_fact_composition: bool = False
    source_message_ref: str = ""
    case_action_logger: Optional[Callable[..., Awaitable[Any]]] = field(
        default=None, repr=False, compare=False
    )

    async def log_case_action(
        self,
        action_type: str,
        description: str,
        source_ref: str,
        idempotency_key: str,
        status: str,
    ) -> None:
        """在交接开关开启时记录动作；普通流程不附加持久化副作用。"""
        if self.case_action_logger is not None:
            await self.case_action_logger(
                action_type, description, source_ref, idempotency_key, status
            )


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


def _usage_from_response(response: Any) -> Dict[str, int]:
    """读取 Anthropic/兼容客户端的 usage，并补齐 total_tokens。"""

    usage = _mapping_value(response, "usage", None)
    if usage is None:
        return {}
    values: Dict[str, int] = {}
    for key in (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "total_tokens",
    ):
        raw = _mapping_value(usage, key, None)
        if isinstance(raw, bool):
            continue
        try:
            number = int(raw)
        except (TypeError, ValueError):
            continue
        if number >= 0:
            values[key] = number
    if "total_tokens" not in values and {"input_tokens", "output_tokens"}.issubset(values):
        values["total_tokens"] = values["input_tokens"] + values["output_tokens"]
    return values


def _merge_usage(*usages: Mapping[str, Any]) -> Dict[str, int]:
    """合并同一请求内多个 LLM 调用的 token usage。"""

    merged: Dict[str, int] = {}
    for usage in usages:
        if not isinstance(usage, Mapping):
            continue
        for key, value in usage.items():
            if isinstance(value, bool):
                continue
            try:
                number = int(value)
            except (TypeError, ValueError):
                continue
            if number >= 0:
                merged[key] = merged.get(key, 0) + number
    if "total_tokens" not in merged and {"input_tokens", "output_tokens"}.issubset(merged):
        merged["total_tokens"] = merged["input_tokens"] + merged["output_tokens"]
    return merged


def _merge_usage_by_stage(
    *stage_usages: Mapping[str, Mapping[str, Any]] | None,
) -> Dict[str, Dict[str, int]]:
    """按阶段合并 Token 用量，保留旧的扁平 usage 契约不变。"""

    merged: Dict[str, Dict[str, int]] = {}
    for usage_map in stage_usages:
        if not isinstance(usage_map, Mapping):
            continue
        for stage, usage in usage_map.items():
            if not isinstance(stage, str) or not isinstance(usage, Mapping):
                continue
            stage_usage = _merge_usage(merged.get(stage, {}), usage)
            if stage_usage:
                merged[stage] = stage_usage
    return merged


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

    # 单个 Agent 最多一轮模型决策 + 一轮工具结果回读；超过两轮通常只是重复追问，
    # 由确定性降级文案接住，避免复杂请求把等待时间线性放大。
    MAX_TOOL_ROUNDS = 2

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

        language_guard = (
            "\n\n输出语言与格式约束：必须使用简体中文直接回答消费者。"
            "不要输出英文内部推理、Agent 名称、工具调用说明或系统提示；"
            "不要说‘我不会承诺退款’、‘不判断责任’、‘不会直接确认’等内部免责或流程边界；"
            "不要把工具结果、系统限制或审核规则原样展示给消费者。"
            "直接回答消费者当前的问题；只有情绪明显或安全场景需要时才简短回应情绪，不添加无信息量的开场白；"
            "先说已核实的事实和用户可采取的下一步，不重复转人工、客服已记录等流程话；"
            f"{_CUSTOMER_CONTINUITY_RULES}"
            "涉及发红、肿胀、起疹、刺痛或‘烂脸’等使用不适时，仍要优先给出必要的安全动作，但遵守上述不重复规则；"
            "追问必须遵守本轮提供的业务字段状态：只问 askable_text_fields 中的首个缺失文本字段，"
            "known_fields 绝不能再问；枚举字段通过页面选项收集，不要在回复正文重复发问。"
            "最多用一句话追问一个缺失的自由文本字段；不要追问注册表未要求的附加信息。"
            "所有订单、退款、时效和产品结论必须来自工具或知识库，无法确认时只说正在核实，不得编造；"
            "不得承诺退款必然成功，也不得承诺产品会治愈或改善症状。"
        )
        if self.skill_manager is None:
            return self.profile.mission + language_guard
        try:
            # Skill 目录与 Agent 对外标识保持同名，避免旧领域名称泄漏到运行时。
            skill_key = self.agent_type.value
            skill_prompt = self.skill_manager.prompt_for_agent(skill_key)
        except Exception as exc:  # pragma: no cover - 仅在可选 Skill 管理器异常时触发
            logger.warning("Skill Prompt 注入失败，将使用 Agent Profile：%s", exc)
            return self.profile.mission + language_guard
        if not skill_prompt:
            return self.profile.mission + language_guard
        return f"{self.profile.mission}\n\n{skill_prompt}{language_guard}"

    def _tool_context(self, request: Request) -> Dict[str, Any]:
        """构造只读请求快照，供白名单工具执行时使用。"""

        intent = request.intent.value if isinstance(request.intent, IntentCategory) else request.intent
        facts = CaseFacts.from_dict(request.case_facts)
        effective_values = facts.effective_values(request.entities)
        entities = {
            key: [str(value) for value in values]
            for key, values in effective_values.items()
            if values
        }
        return {
            "request_id": request.request_id,
            "user_id": request.user_id,
            "intent": intent or "other",
            "intent_ids": _all_request_intent_ids(request),
            "message": request.message,
            "entities": entities,
            "case_facts": facts.to_dict(),
            "timestamp": None,
        }

    async def _final_answer_after_tools(
        self,
        request: Request,
        messages: List[Dict[str, Any]],
        model: str,
    ) -> Tuple[str, Dict[str, int]]:
        """工具阶段后要求模型仅基于工具结果和共享事实生成消费者答复。"""
        facts = CaseFacts.from_dict(request.case_facts).to_prompt_dict()
        field_state = build_business_field_state(request)
        response = await self.client.messages.create(
            model=model,
            max_tokens=self.profile.max_tokens,
            temperature=self.profile.temperature,
            extra_body={"thinking": {"type": "disabled"}},
            system=(
                f"{self._system_prompt()}\n\n"
                "你现在处于最终答复阶段。工具已返回业务事实；以下 CaseFacts 是字段归一后的引用视图。"
                "请直接回答消费者的问题，只陈述工具或知识证据支持的内容。"
                "未返回的状态要明确说尚未查到。追问只允许针对 askable_text_fields，"
                "最多一句、一个字段；枚举字段由页面选项收集，不能重复追问；known_fields 和其他非必需字段不得询问。"
                "不要写占位话术、内部工具名、执行过程或编造状态。"
            ),
            messages=[
                *messages,
                {
                    "role": "user",
                    "content": (
                        "本轮可引用的 CaseFacts："
                        + json.dumps(facts, ensure_ascii=False, default=str)
                        + "\n当前业务字段状态（追问以此为准）："
                        + json.dumps(field_state, ensure_ascii=False, default=str)
                    ),
                },
            ],
        )
        text = "\n".join(
            str(_mapping_value(block, "text", ""))
            for block in _response_blocks(response)
            if _mapping_value(block, "type") == "text"
        ).strip()
        placeholders = (
            "我已了解您的问题", "已完成必要的工具处理", "正在核对相关信息",
        )
        if not text or any(text.startswith(item) for item in placeholders) or "DSML" in text:
            raise ValueError("工具已成功返回，但最终答复模型仍返回占位内容")
        return text, _usage_from_response(response)

    async def _run_llm(
        self,
        request: Request,
        tools_used: List[str],
        tool_traces: List[Dict[str, Any]],
        tools_override: Optional[List[AgentToolSpec]] = None,
    ) -> Tuple[str, Dict[str, int]]:
        """执行最多两轮 Tool Use，并只暴露当前 Agent 的工具白名单。"""

        if self.client is None:
            raise RuntimeError("Anthropic client 未配置")
        user_content = request.message
        if request.context:
            user_content = f"[请求上下文]\n{request.context}\n\n用户问题：{request.message}"
        user_content += (
            "\n\n[当前业务字段状态；追问只能来自 askable_text_fields，"
            "枚举字段由页面选项收集，known_fields 与非必需字段不得再问]\n"
            + json.dumps(build_business_field_state(request), ensure_ascii=False, default=str)
        )
        messages: List[Dict[str, Any]] = [{"role": "user", "content": user_content}]
        visible_tools = self.tools if tools_override is None else tools_override
        declarations = [tool.to_anthropic_tool() for tool in visible_tools]
        tool_map = {tool.name: tool for tool in visible_tools}
        model = self.profile.model or "claude-3-5-sonnet-20241022"
        usage: Dict[str, int] = {}

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
            usage = _merge_usage(usage, _usage_from_response(response))
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
                placeholders = ("我已了解您的问题", "已完成必要的工具处理", "正在核对相关信息")
                if tools_used and (any(text.startswith(item) for item in placeholders) or "DSML" in text):
                    final_text, final_usage = await self._final_answer_after_tools(request, messages, model)
                    text = final_text
                    usage = _merge_usage(usage, final_usage)
                return text, usage

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
                tool_use_id = str(_mapping_value(block, "id", f"tool-{round_index + 1}"))
                action_key = f"tool:{request.request_id}:{tool_use_id}"
                action_source = request.source_message_ref or request.request_id
                await request.log_case_action(
                    "tool", f"开始调用工具 {name}", action_source, action_key, "pending"
                )
                try:
                    tool_result = await tool.execute(tool_input, context=self._tool_context(request))
                except Exception:
                    await request.log_case_action(
                        "tool", f"工具 {name} 调用失败", action_source, action_key, "failed"
                    )
                    raise
                tools_used.append(name)
                if isinstance(tool_result, Mapping):
                    facts = CaseFacts.from_dict(request.case_facts)
                    facts.record_tool_result(name, tool_result, source_ref=request.request_id)
                    request.case_facts = facts.to_dict()
                retrieval_tool = name == "search_knowledge_base"
                tool_success = bool(
                    tool_result.get("success", False)
                    if isinstance(tool_result, Mapping)
                    else True
                )
                retrieval_operation_succeeded = tool_success
                retrieval_available = bool(
                    retrieval_tool
                    and isinstance(tool_result, Mapping)
                    and tool_result.get("success")
                    and tool_result.get("available")
                )
                if retrieval_tool:
                    tool_success = retrieval_available
                await request.log_case_action(
                    "tool",
                    f"工具 {name} {'返回成功' if tool_success else '返回失败'}",
                    action_source,
                    action_key,
                    "success" if tool_success else "failed",
                )
                tool_trace = {
                    "round": round_index + 1,
                    "tool": name,
                    "tool_name": name,
                    "input": tool_input,
                    "latency_ms": round(
                        max(0.0, (time.perf_counter() - tool_started) * 1000),
                        3,
                    ),
                    "status": (
                        "success" if retrieval_available else
                        "failed" if retrieval_tool and not retrieval_operation_succeeded else
                        "empty" if retrieval_tool else
                        "success" if tool_success else "failed"
                    ),
                    "success": tool_success,
                }
                if retrieval_tool:
                    tool_trace.update({
                        "retrieval_stage": "knowledge_tool",
                        "knowledge_required": True,
                        "retrieved_chunks": list(tool_result.get("chunks") or [])
                        if isinstance(tool_result, Mapping) else [],
                    })
                tool_traces.append(tool_trace)
                if retrieval_tool and not retrieval_available:
                    # Why：Agent 一旦主动请求条件知识检索，该证据就成为本次答案的依据；
                    # 空召回或异常时立即终止，不允许再请求模型“补全”事实。
                    raise RuntimeError("必需知识检索未返回可用资料，已停止模型回答")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": _mapping_value(block, "id", f"tool-{round_index + 1}"),
                        "content": json.dumps(tool_result, ensure_ascii=False, default=str),
                    }
                )
            messages.append({"role": "user", "content": tool_results})

            # Why：两轮上限控制 Token 与工具副作用边界，防止模型在 ReAct 循环中无限调用。
            if round_index == self.MAX_TOOL_ROUNDS - 1:
                final_text, final_usage = await self._final_answer_after_tools(request, messages, model)
                return final_text, _merge_usage(usage, final_usage)

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
            # 订单/物流的常规“查不到订单”场景不值得等待一次模型调用：
            # 本地规则已确认意图，且没有订单号可供工具查询时，直接给出下一步。
            # 有订单号时仍走白名单工具，避免把真实订单状态写死。
            fast_intent = request.intent in {
                IntentCategory.ORDER_STATUS_BEAUTY,
                IntentCategory.LOGISTICS_STALLED,
            }
            has_order_id = bool(request.entities.get("order_id"))
            if request.intent_diagnostics.get("fast_path") and fast_intent and not has_order_id:
                content = customer_friendly_response(
                    "目前还缺少订单号，我暂时无法定位具体订单。请把订单号发给我，我就能继续帮您核对发货和物流进度。",
                    intent=request.intent,
                    emotion=str(request.intent_diagnostics.get("emotion", "neutral")),
                )
                latency_ms = max(0.0, (time.perf_counter() - started) * 1000)
                self.stats.success += 1
                self.stats.total_ms += latency_ms
                return AgentResponse(
                    agent_type=self.agent_type,
                    content=content,
                    success=True,
                    latency_ms=latency_ms,
                    confidence=request.intent_confidence,
                    case_facts=dict(request.case_facts or {}),
                )
            content, usage = await self._run_llm(
                request,
                tools_used,
                tool_traces,
                tools_override=tools_override,
            )
            content = _customer_safe_text(content, request)
            content = customer_friendly_response(
                content,
                intent=request.intent,
                emotion=str(request.intent_diagnostics.get("emotion", "neutral")),
            )
            field_state = build_business_field_state(request)
            content, repeated_fields = filter_redundant_field_questions(
                content,
                field_state["known_fields"],
                allowed_fields=(
                    set(field_state["askable_text_fields"])
                    if field_state["blocking_fields"]
                    else None
                ),
            )
            if repeated_fields:
                logger.warning(
                    "Agent 回复中的重复/越界字段追问已拦截：agent=%s fields=%s",
                    self.agent_type.value,
                    repeated_fields,
                )
            if not content and field_state["askable_text_fields"]:
                missing_field = field_state["askable_text_fields"][0]
                slot = get_slot(missing_field)
                display_name = str((slot or {}).get("display_name") or missing_field)
                content = f"还需要您补充一下{display_name}，我才能继续处理。"
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
                usage=usage,
                case_facts=dict(request.case_facts or {}),
            )
        except Exception as exc:
            latency_ms = max(0.0, (time.perf_counter() - started) * 1000)
            self.stats.total_ms += latency_ms
            logger.exception("%s Agent 处理失败: %s", self.agent_type.value, exc)
            return AgentResponse(
                agent_type=self.agent_type,
                content=f"Agent 执行失败: {exc}",
                success=False,
                confidence=0.0,
                latency_ms=latency_ms,
                tools_used=tools_used,
                tool_traces=[
                    *tool_traces,
                    {
                        "stage": "agent_execution",
                        "agent_type": self.agent_type.value,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                ],
                usage={},
                case_facts=dict(request.case_facts or {}),
            )


GENERAL_PROFILE = AgentProfile(
    role="consult",
    mission="负责美妆产品、成分、色号、用法、活动和特殊人群适用咨询。",
    workflow=("inspect_context", "search_beauty_knowledge", "clarify", "respond"),
    input_contract=("message", "history", "intent", "entities", "urgency", "context"),
    output_contract=("response", "customer_emotion_ack", "next_steps", "required_fields", "knowledge_context"),
    tool_scope=("search_knowledge_base", "inspect_request_context", "suggest_required_fields", "create_ticket_draft"),
    temperature=0.3,
    max_tokens=900,
)

TECHNICAL_PROFILE = AgentProfile(
    role="order",
    mission="负责美妆订单、物流、漏发、错发、破损和履约异常处理。",
    workflow=("inspect_order", "search_beauty_knowledge", "clarify", "draft_ticket", "respond"),
    input_contract=("message", "entities", "context", "urgency"),
    output_contract=("response", "customer_emotion_ack", "next_steps", "required_fields", "ticket_draft", "knowledge_context"),
    tool_scope=("search_knowledge_base", "inspect_request_context", "suggest_required_fields", "lookup_order", "lookup_logistics", "create_ticket_draft"),
    temperature=0.1,
    max_tokens=1200,
)

BILLING_PROFILE = AgentProfile(
    role="after_sales",
    mission="负责美妆退货退款、价保、退款异常和不良反应风险处理。",
    workflow=("inspect_case", "search_beauty_knowledge", "clarify", "draft_ticket", "respond"),
    input_contract=("message", "entities", "context", "urgency"),
    output_contract=("response", "customer_emotion_ack", "next_steps", "missing_fields", "ticket_draft", "risk_event", "knowledge_context"),
    tool_scope=("search_knowledge_base", "inspect_request_context", "suggest_required_fields", "lookup_order", "check_refund_status", "calculate_price_difference", "create_ticket_draft", "record_risk_event"),
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
        default_tools = list(beauty_tools().values())
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
        default_tools = [SEARCH_KNOWLEDGE_BASE_TOOL, *beauty_tools().values()]
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
        default_tools = [SEARCH_KNOWLEDGE_BASE_TOOL, *beauty_tools().values()]
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
            # Why：request_id、意图和紧急度属于内部追踪字段，不应混入用户可见的客服回复正文。
            content = "您的问题已记录下来，当前建议联系人工客服进一步核实和处理。"
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
    domain: Optional[str] = None,
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
        search_kwargs = {"top_k": top_k}
        if domain:
            search_kwargs["domain"] = domain
        try:
            result = await asyncio.to_thread(searcher.search, query, **search_kwargs)
        except TypeError:
            # 兼容旧版 KnowledgeBase 或测试替身的二参数 search 签名。
            search_kwargs.pop("domain", None)
            result = await asyncio.to_thread(searcher.search, query, **search_kwargs)
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
    """按当前 Agent 涉及意图的策略决定必需/条件知识检索。"""
    intent_ids = _intent_ids_for_agent(request, agent_type)
    if not intent_ids and not request.knowledge_required:
        return request, None, None
    policy = policy_for_intents(intent_ids)
    should_prefetch = bool(request.knowledge_required or policy["knowledge_mode"] == "required") or any(
        should_prefetch_knowledge(intent_id, request.message)
        for intent_id in intent_ids
    )
    if not should_prefetch:
        policy_context = (
            "[Agent 执行边界] "
            + json.dumps(
                {
                    "intents": intent_ids,
                    "knowledge_mode": policy["knowledge_mode"],
                    "allowed_tools": policy["allowed_tools"],
                    "blocking_fields": policy["blocking_fields"],
                    "missing_fields_action": policy["missing_fields_action"],
                    "departments": policy["departments"],
                },
                ensure_ascii=False,
            )
        )
        context = f"{request.context}\n\n{policy_context}" if request.context else policy_context
        return replace(request, context=context), None, None
    scoped_queries = request.intent_diagnostics.get("scoped_retrieval_queries")
    scoped_query_value = request.intent_diagnostics.get("scoped_retrieval_query")
    if isinstance(scoped_queries, Mapping):
        scoped_query_value = scoped_queries.get(agent_type.value, scoped_query_value)
    scoped_query = str(scoped_query_value or request.message or "")
    domain = agent_type.value if agent_type in {
        AgentType.GENERAL,
        AgentType.TECHNICAL,
        AgentType.BILLING,
    } else None
    # Why：每个实际 Agent 固定只预检索一次且使用 top_k=5，避免重复召回和上下文膨胀。
    recall_k = 5
    started = time.perf_counter()
    try:
        raw = await _call_knowledge_search(
            searcher, scoped_query, recall_k, request, domain=domain
        )
        success, chunks, error = _extract_retrieval_payload(raw)
    except Exception as exc:
        success, chunks, error = False, [], str(exc)
    latency_ms = max(0.0, (time.perf_counter() - started) * 1000)
    trace: Dict[str, Any] = {
        "tool_name": "search_knowledge_base",
        "input": {
            "query": scoped_query,
            "top_k": recall_k,
            **({"domain": domain} if domain else {}),
        },
        "latency_ms": round(latency_ms, 3),
        "status": "failed" if not success else "success" if chunks else "empty",
        "knowledge_required": True,
        "retrieval_stage": "knowledge_prefetch",
        "agent_type": agent_type.value,
        "retrieved_chunks": chunks,
        "chunk_summary": _retrieval_summary(chunks),
    }
    if error:
        trace["error"] = error
    context_text = _format_retrieval_context(chunks)
    policy_context = (
        "[Agent 执行边界] "
        + json.dumps(
            {
                "intents": intent_ids,
                "knowledge_mode": policy["knowledge_mode"],
                "allowed_tools": policy["allowed_tools"],
                "blocking_fields": policy["blocking_fields"],
                "missing_fields_action": policy["missing_fields_action"],
                "departments": policy["departments"],
            },
            ensure_ascii=False,
        )
    )
    merged_context = (
        f"{request.context}\n\n{context_text}\n\n{policy_context}"
        if request.context
        else f"{context_text}\n\n{policy_context}"
    )
    prepared_request = replace(request, context=merged_context)
    return prepared_request, trace, None


def _all_request_intent_ids(request: Request) -> List[str]:
    values: List[str] = []
    for candidate in [request.intent, *(request.secondary_intents or [])]:
        value = getattr(candidate, "value", candidate)
        if value and str(value) in BEAUTY_INTENT_TICKET_BY_ID:
            values.append(str(value))
    for item in request.sub_requests or []:
        value = item.get("intent") or item.get("intent_id") if isinstance(item, Mapping) else None
        value = getattr(value, "value", value)
        if value and str(value) in BEAUTY_INTENT_TICKET_BY_ID:
            values.append(str(value))
    return list(dict.fromkeys(values))


def _intent_ids_for_agent(request: Request, agent_type: AgentType) -> List[str]:
    target = str(getattr(agent_type, "value", agent_type))
    return [
        intent_id
        for intent_id in _all_request_intent_ids(request)
        if BEAUTY_INTENT_TICKET_BY_ID[intent_id]["agent"] == target
    ]


def _request_for_agent(request: Request, agent_type: AgentType) -> Request:
    """给并行领域 Agent 只传本域子请求及对应策略，避免各 Agent 重复处理整句。"""
    intent_ids = _intent_ids_for_agent(request, agent_type)
    if not intent_ids:
        return request

    target = str(getattr(agent_type, "value", agent_type))
    scoped_sub_requests: List[Dict[str, Any]] = []
    for item in request.sub_requests or []:
        if not isinstance(item, Mapping):
            continue
        raw_intent = item.get("intent") or item.get("intent_id") or ""
        sub_intent_id = str(getattr(raw_intent, "value", raw_intent))
        if sub_intent_id in intent_ids:
            scoped_sub_requests.append(dict(item))
    diagnostics = dict(request.intent_diagnostics or {})
    scoped_queries = diagnostics.get("scoped_retrieval_queries")
    scoped_message = (
        str(scoped_queries.get(target) or "").strip()
        if isinstance(scoped_queries, Mapping)
        else ""
    )
    if not scoped_message and scoped_sub_requests:
        scoped_message = "；".join(
            dict.fromkeys(
                str(item.get("text") or "").strip()
                for item in scoped_sub_requests
                if str(item.get("text") or "").strip()
            )
        )
    scoped_message = scoped_message or request.message

    primary_intent_id = str(getattr(request.intent, "value", request.intent))
    if primary_intent_id not in intent_ids:
        primary_intent_id = intent_ids[0]
    scoped_secondary = [
        IntentCategory(intent_id)
        for intent_id in intent_ids
        if intent_id != primary_intent_id
    ]
    scoped_sub_request_ids = [str(item.get("id") or "") for item in scoped_sub_requests]
    diagnostics["agent_scope"] = {
        "agent": target,
        "intent_ids": intent_ids,
        "sub_request_ids": scoped_sub_request_ids,
    }
    if isinstance(scoped_queries, Mapping):
        diagnostics["scoped_retrieval_queries"] = {target: scoped_message}
    branch_knowledge_required = any(
        should_prefetch_knowledge(intent_id, scoped_message)
        for intent_id in intent_ids
    )
    return replace(
        request,
        message=scoped_message,
        intent=IntentCategory(primary_intent_id),
        secondary_intents=scoped_secondary,
        sub_requests=scoped_sub_requests,
        knowledge_required=branch_knowledge_required,
        intent_diagnostics=diagnostics,
    )


def _policy_scoped_tools(
    agent: Any,
    request: Request,
    agent_type: AgentType,
    searcher: Optional[Any],
    *,
    knowledge_prefetched: bool,
) -> Optional[List[AgentToolSpec]]:
    """把 Agent 的静态能力收窄到本轮意图注册表许可的动作集合。"""
    tools = getattr(agent, "tools", None)
    if tools is None:
        return None
    intent_ids = _intent_ids_for_agent(request, agent_type)
    allowed = set(policy_for_intents(intent_ids)["allowed_tools"])
    scoped: List[AgentToolSpec] = []
    for tool in tools:
        if tool.name not in allowed:
            continue
        if tool.name == "search_knowledge_base":
            if knowledge_prefetched:
                continue

            async def search_handler(params: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
                if not isinstance(params, Mapping) or not isinstance(params.get("query"), str):
                    return {"success": False, "available": False, "error": "query 必须是字符串"}
                query = str(params["query"]).strip()
                top_k = max(1, min(int(params.get("top_k", 5) or 5), 10))
                try:
                    raw = await _call_knowledge_search(
                        searcher, query, top_k, request, domain=agent_type.value
                    )
                    success, chunks, error = _extract_retrieval_payload(raw)
                except Exception as exc:
                    success, chunks, error = False, [], f"{type(exc).__name__}: {exc}"
                result: Dict[str, Any] = {
                    "success": success,
                    "available": bool(chunks),
                    "chunks": chunks,
                    "message": "已检索到可引用资料。" if chunks else "没有检索到可确认的资料。",
                }
                if error:
                    result["error"] = error
                return result

            scoped.append(
                AgentToolSpec(tool.name, tool.description, tool.input_schema, search_handler)
            )
        else:
            scoped.append(tool)
    return scoped


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

    agent_request = replace(
        request,
        case_facts=CaseFacts.from_dict(request.case_facts).to_dict(),
    )
    prepared_request, retrieval_trace, _ = await _prepare_agent_request(
        agent_request, agent_type, searcher
    )
    if retrieval_trace and retrieval_trace.get("knowledge_required") and retrieval_trace.get("status") != "success":
        # 必需检索未获得任何证据时直接短路；不让模型在空上下文上编造业务结论。
        blocked_response = AgentResponse(
            agent_type=agent_type,
            content="",
            success=False,
            confidence=0.0,
            latency_ms=float(retrieval_trace.get("latency_ms", 0.0) or 0.0),
        )
        return _attach_trace(blocked_response, retrieval_trace)
    if retrieval_trace and retrieval_trace.get("status") == "success":
        facts = CaseFacts.from_dict(prepared_request.case_facts)
        facts.record_tool_result(
            "search_knowledge_base",
            {"success": True, "chunks": retrieval_trace.get("retrieved_chunks", [])},
            source_ref=request.request_id,
        )
        prepared_request.case_facts = facts.to_dict()
    tools_override = _policy_scoped_tools(
        agent,
        prepared_request,
        agent_type,
        searcher,
        knowledge_prefetched=bool(retrieval_trace and retrieval_trace.get("status") == "success"),
    )
    try:
        response = await _invoke_agent(agent, prepared_request, tools_override)
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        logger.exception("%s Agent 执行异常: %s", agent_type.value, error_text)
        response = _friendly_failure(
            agent_type,
            retrieval_trace["latency_ms"] if retrieval_trace else 0.0,
        )
        response.tool_traces.append({
            "stage": "agent_execution",
            "agent_type": agent_type.value,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
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


_CONSULT_INTENTS = frozenset({
    IntentCategory.INGREDIENT_FIT, IntentCategory.SHADE_MATCH,
    IntentCategory.USAGE_HOWTO, IntentCategory.PRODUCT_INFO,
    IntentCategory.PROMO_RULES, IntentCategory.PREGNANCY_SAFETY,
})
_ORDER_INTENTS = frozenset({
    IntentCategory.ORDER_STATUS_BEAUTY, IntentCategory.LOGISTICS_STALLED,
    IntentCategory.SIGNED_NOT_RECEIVED, IntentCategory.INTERCEPT_CHANGE_ADDRESS,
    IntentCategory.MISSING_ITEM, IntentCategory.WRONG_ITEM,
    IntentCategory.DAMAGED_ON_ARRIVAL,
})
_AFTER_SALES_INTENTS = frozenset({
    IntentCategory.RETURN_REFUND, IntentCategory.ADVERSE_REACTION,
    IntentCategory.SUSPECTED_ABNORMAL_REFUND, IntentCategory.PRICE_PROTECTION,
    IntentCategory.REFUND_NOT_RECEIVED, IntentCategory.SHIPPING_FEE_REIMBURSEMENT,
})
_CONSULT_KEYWORDS = ("成分", "肤质", "色号", "怎么用", "用法", "适合", "活动", "孕妇")
_ORDER_KEYWORDS = ("订单", "物流", "快递", "签收", "漏发", "错发", "破损", "改址", "拦截")
_AFTER_SALES_KEYWORDS = ("退款", "退货", "不适", "过敏", "发红", "价保", "补差", "运费")
CLARIFICATION_PROMPT = "我还不能确定您要处理的是哪类问题。请补充一下是产品咨询、订单物流，还是售后退款与使用不适？"


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
    scores = _DomainScores({AgentType.INQUIRY: 0.1, AgentType.FULFILLMENT: 0.0, AgentType.REFUND_RISK: 0.0})
    if intent in _CONSULT_INTENTS:
        scores[AgentType.INQUIRY] += 0.75
    if intent in _ORDER_INTENTS:
        scores[AgentType.FULFILLMENT] += 0.75
    if intent in _AFTER_SALES_INTENTS:
        scores[AgentType.REFUND_RISK] += 0.75
    scores[AgentType.INQUIRY] += min(0.35, sum(k in message for k in _CONSULT_KEYWORDS) * 0.12)
    scores[AgentType.FULFILLMENT] += min(0.45, sum(k in message for k in _ORDER_KEYWORDS) * 0.15)
    scores[AgentType.REFUND_RISK] += min(0.45, sum(k in message for k in _AFTER_SALES_KEYWORDS) * 0.15)
    if entities.get("order_id"):
        scores[AgentType.FULFILLMENT] += 0.1
    return _DomainScores({domain: round(score, 3) for domain, score in scores.items()})


def _agent_for_intent(intent: IntentCategory) -> AgentType:
    """将已确认意图映射到其领域 Agent。"""

    domain = _intent_domain(intent)
    return {
        "consult": AgentType.INQUIRY,
        "order": AgentType.FULFILLMENT,
        "after_sales": AgentType.REFUND_RISK,
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

    # Why：HUMAN_HANDOFF 是终止型意图，最低 0.8 才能稳定表达升级决策。
    # Why：任一层确认人工意图都不能被普通域路由覆盖，避免越过人机协同边界。
    if intent is IntentCategory.HUMAN_HANDOFF or request_intent is IntentCategory.HUMAN_HANDOFF:
        return RoutingDecision(
            primary_agent=AgentType.ESCALATION,
            reason="意图为 human_handoff，触发升级路由",
            confidence=max(confidence, 0.8),
        )
    # 明确的安全高紧急度请求仍由业务 Agent 基于事实作答，同时在结果中标记人工接管。
    # 未识别到受控美妆意图时保留快速升级路径，不把未知高风险文本派给普通工具集。
    if urgency == UrgencyLevel.CRITICAL or request_urgency == UrgencyLevel.CRITICAL:
        intent_id = intent.value
        if intent_id in BEAUTY_INTENT_TICKET_BY_ID:
            return RoutingDecision(
                primary_agent=_agent_for_intent(intent),
                reason=f"CRITICAL 风险由 {intent_id} 领域 Agent 基于事实答复并升级",
                confidence=1.0,
            )
        return RoutingDecision(
            primary_agent=AgentType.ESCALATION,
            reason="未知或非美妆意图紧急度为 CRITICAL，触发安全升级",
            confidence=1.0,
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
        self._usage_by_request: Dict[str, Dict[str, int]] = {}
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
        facts = CaseFacts.from_dict(request.case_facts).to_dict()
        field_state = build_business_field_state(request)
        tool_fact_instruction = (
            "部分领域分支未能生成答复，但本轮成功工具结果仍有效；请将工具来源事实与已完成分支合并，逐项回应用户诉求。\n"
            if request.requires_tool_fact_composition
            else ""
        )
        return (
            "你是 ShopCare 的客户回复合成器。请以主处理分支的结论为主体，"
            f"吸收补充分支中不重复且不冲突的信息，删除重复开场白并平滑处理冲突。{_CUSTOMER_CONTINUITY_RULES}\n"
            "只输出面向客户的纯净中文答复，直接回答问题；仅在情绪明显或安全场景需要时简短回应情绪。"
            "不要输出‘不会承诺退款’‘不判断责任’等内部边界声明，不要暴露 Agent、工具或系统状态。"
            "严禁无中生有编造订单发货、退款成功或后台查询结果；未核实的内容明确说明尚未查到。"
            "追问必须严格遵守下面的业务字段状态：只可询问 askable_text_fields 中的首个缺失文本字段，"
            "最多一句、一个字段；枚举字段由页面选项收集，不要在回复正文重复提问；"
            "known_fields 与未列入 blocking_fields 的附加问题不得再问。"
            "以下 CaseFacts 是共享字段视图，工具来源事实优先于消息推测；不要重写或丢弃已核实事实。\n\n"
            f"{tool_fact_instruction}"
            f"最近对话（用于识别已答内容，不要复述）：\n{request.context or '无'}\n\n"
            f"当前业务字段状态：\n{json.dumps(field_state, ensure_ascii=False, default=str)}\n\n"
            f"用户问题：{request.message}\n\n"
            f"CaseFacts：\n{json.dumps(facts, ensure_ascii=False, default=str)}\n\n"
            f"主处理分支类型（仅供内部排序，不要在输出中提及）：{primary.agent_type.value}\n"
            f"主处理分支结论：\n{primary.content}\n\n"
            f"补充信息：\n{support_text}"
        )

    @staticmethod
    def _current_tool_facts(request: Request) -> Dict[str, List[Dict[str, Any]]]:
        """只提取本轮成功工具写入的事实，避免把旧事实当成本轮查询结果。"""

        facts = CaseFacts.from_dict(request.case_facts)
        current: Dict[str, List[Dict[str, Any]]] = {}
        for field_name, records in facts.facts.items():
            matches = [
                dict(record)
                for record in records
                if str(record.get("source_type", "")).startswith("tool:")
                and str(record.get("source_ref", "")) == str(request.request_id)
                and record.get("confirmation_status") != "cleared"
                and record.get("value") is not None
            ]
            if matches:
                current[field_name] = matches
        return current

    @classmethod
    def _build_tool_fact_only_prompt(cls, request: Request) -> str:
        """领域 Agent 未生成答复时，让最终合成模型直接基于工具事实回应。"""

        facts = cls._current_tool_facts(request)
        return (
            "你是 ShopCare 的客户回复合成器。领域 Agent 未能完成最终答复，但本轮工具已成功返回以下事实。"
            f"请只根据这些事实直接回答用户问题；{_CUSTOMER_CONTINUITY_RULES}不要提及内部 Agent、工具或系统故障。"
            "不得补造事实；未覆盖的问题明确说明当前查到的范围。只输出简洁自然的中文答复。\n\n"
            f"最近对话（用于识别已答内容，不要复述）：\n{request.context or '无'}\n\n"
            f"当前业务字段状态（只问 askable_text_fields；枚举字段由页面选项收集）：\n"
            f"{json.dumps(build_business_field_state(request), ensure_ascii=False, default=str)}\n\n"
            f"用户问题：{request.message}\n\n"
            f"本轮工具核实事实：\n{json.dumps(facts, ensure_ascii=False, default=str)}"
        )

    @classmethod
    def _tool_facts_fallback(
        cls,
        request: Request,
        responses: List[AgentResponse],
        primary_agent: Optional[AgentType],
    ) -> str:
        """最终模型不可用时，返回可读的本轮工具事实而不是空泛失败占位。"""

        labels = {
            "order_id": "订单号", "product_name": "商品", "sku": "规格",
            "quantity": "数量", "order_status": "订单状态", "tracking_no": "物流单号",
            "carrier": "承运商", "logistics_status": "物流状态", "refund_status": "退款状态",
            "refund_amount": "退款金额", "price_difference": "价差", "gift_items": "赠品",
        }
        availability_labels = {
            "lookup_order_available": "订单信息",
            "lookup_logistics_available": "物流信息",
            "check_refund_status_available": "退款信息",
        }
        items: List[str] = []
        for field_name, records in cls._current_tool_facts(request).items():
            if field_name in availability_labels and records[-1].get("value") is False:
                items.append(f"暂未查询到{availability_labels[field_name]}")
                continue
            label = labels.get(field_name)
            if not label:
                continue
            value = records[-1].get("value")
            if isinstance(value, (list, tuple, set)):
                value = "、".join(str(item) for item in value if item)
            elif isinstance(value, Mapping):
                value = json.dumps(value, ensure_ascii=False, default=str)
            text = str(value).strip()
            if text:
                items.append(f"{label}：{text}")
        if not items:
            return cls._fallback(responses, primary_agent)
        fact_text = "已核实信息：" + "；".join(items) + "。"
        if not responses:
            return fact_text
        return f"{cls._fallback(responses, primary_agent)}\n\n{fact_text}"

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
        compose_tool_facts = bool(request.requires_tool_fact_composition)
        if not successful and not compose_tool_facts:
            return self._fallback(successful, primary_agent)
        if len(successful) == 1 and not compose_tool_facts:
            # Why：单分支无需再次请求模型，避免额外 Token 与合成延迟。
            return successful[0].content

        primary = next((item for item in successful if item.agent_type is primary_agent), None)
        if primary is None and successful:
            primary = successful[0]
        supporting = [item for item in successful if item is not primary]
        if self.client is None:
            logger.warning("ResponseComposer 未配置 Anthropic client，使用确定性拼接降级")
            return self._tool_facts_fallback(request, successful, primary_agent)
        try:
            prompt = (
                self._build_tool_fact_only_prompt(request)
                if not successful
                else self._build_prompt(request, primary, supporting)
            )
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                # Why：DeepSeek 的 Anthropic 兼容接口默认可能返回 thinking block，
                # 合成器只需要客户可见文本；关闭 thinking 可避免预算被推理块耗尽。
                extra_body={"thinking": {"type": "disabled"}},
                system="只输出面向客户的中文答复。不要输出分析过程、内部 Agent 名称或未被证实的外部业务结果。",
                messages=[{"role": "user", "content": prompt}],
            )
            self._usage_by_request[request.request_id] = _usage_from_response(response)
            text = self._extract_text(response)
            placeholders = ("我已了解您的问题", "已完成必要的工具处理", "正在核对相关信息", "所有 Agent 均处理失败")
            if not text or (compose_tool_facts and any(text.startswith(item) for item in placeholders)):
                raise ValueError("合成模型未返回可用的事实答复")
            return text
        except Exception as exc:
            # Why：合成器故障不能让已成功的领域答案白屏，确定性拼接保留事实信息。
            logger.warning("ResponseComposer 合成失败，降级拼接：%s", exc)
            return self._tool_facts_fallback(request, successful, primary_agent)

    async def compose_human_suggested_reply(
        self,
        request: Request,
        handoff_summary: Mapping[str, Any],
        fallback_replies: List[str],
    ) -> Tuple[str, Dict[str, int]]:
        """把人工下一步润色为可直接采纳的话术，不改写消费者已收到的答复。"""
        fallback = str((fallback_replies or [""])[0]).strip()
        steps = [str(item) for item in handoff_summary.get("next_steps", []) if item]
        if self.client is None or not steps:
            return fallback, {}
        facts = CaseFacts.from_dict(request.case_facts).to_prompt_dict()
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=420,
                temperature=self.temperature,
                extra_body={"thinking": {"type": "disabled"}},
                system=(
                    "你为已经接手会话的人工美妆客服润色一条建议回复。"
                    "仅根据人工下一步和 CaseFacts 写一段自然、简洁、可直接发送的中文。"
                    "直接询问或说明需要核对的事项，不要寒暄、复述用户原话、说已转人工、"
                    "虚构查证结果、退款承诺或把内部步骤写成已经完成。"
                ),
                messages=[{
                    "role": "user",
                    "content": (
                        f"消费者当前消息：{request.message}\n"
                        f"人工下一步：{json.dumps(steps, ensure_ascii=False)}\n"
                        f"可用事实：{json.dumps(facts, ensure_ascii=False, default=str)}\n"
                        "只输出建议回复正文。"
                    ),
                }],
            )
            text = self._extract_text(response)
            banned = ("已为您转接人工", "为您转接人工", "已转人工", "我已了解您的问题")
            if not text or any(token in text for token in banned) or "DSML" in text:
                raise ValueError("模型建议回复越界或为空")
            return text, _usage_from_response(response)
        except Exception as exc:
            logger.warning("人工建议回复润色失败，保留动作型基础话术：%s", exc)
            return fallback, {}

    def usage_for(self, request_id: str) -> Dict[str, int]:
        """读取并移除指定请求的合成器 usage，避免跨请求串计费。"""

        return dict(self._usage_by_request.pop(str(request_id), {}) or {})


async def _execute(
    request: Request,
    agent_type: AgentType,
    *,
    knowledge_search: Optional[Any] = None,
    agent_pool: Optional[Mapping[AgentType, Any]] = None,
) -> AgentResponse:
    """执行指定业务域 Agent；失败不跨业务域转派。

    Args:
        request: 当前内部请求。
        agent_type: 路由选定的 Agent 类型。
        knowledge_search: 可注入的知识检索函数或提供 `search` 方法的对象。
        agent_pool: 可选 Agent 池；未传入时使用模块级默认池。

    Returns:
        目标 Agent 的响应；异常被转换为友好失败结果并保留本域 Trace。
    """

    pool = AGENT_POOL if agent_pool is None else agent_pool
    agent = _resolve_agent(pool, agent_type)
    if agent is None:
        return _friendly_failure(agent_type)

    response = await _execute_once(request, agent_type, agent, knowledge_search)

    if response.success:
        return response

    # 保留本业务域 Trace 供客服排查，但不把订单/售后失败交给咨询 Agent 猜答。
    return replace(
        _friendly_failure(agent_type, response.latency_ms),
        tools_used=list(response.tools_used),
        tool_traces=list(response.tool_traces),
        usage=dict(response.usage),
        # Agent 可能已经成功执行工具，只在后续答复模型阶段失败。保留其事实，
        # 供编排层持久化与人工处理；失败文本仍不会进入消费者答复合成。
        case_facts=dict(response.case_facts),
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
    usage: Dict[str, int] = field(default_factory=dict)
    usage_by_stage: Dict[str, Dict[str, int]] = field(default_factory=dict)
    handoff_summary: Optional[Dict[str, Any]] = None
    ticket_draft: Optional[Dict[str, Any]] = None
    suggested_replies: List[str] = field(default_factory=list)
    case_facts: Dict[str, Any] = field(default_factory=dict)


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
        每个已执行分支的 `AgentResponse`，包含失败分支的工具 Trace 和已核实事实；
        单个协程异常会被记录并跳过，消费者答复由后续成功结果筛选。
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
    scoped_requests = [
        _request_for_agent(request, agent_type)
        for agent_type in agent_types
    ]
    tasks = [
        _execute(scoped_request, agent_type, **execute_kwargs)
        for agent_type, scoped_request in zip(agent_types, scoped_requests)
    ]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    completed: List[AgentResponse] = []
    for agent_type, scoped_request, response in zip(agent_types, scoped_requests, responses):
        if isinstance(response, Exception):
            logger.error("%s Agent Fan-out 异常：%s", agent_type.value, response)
            continue
        if not isinstance(response, AgentResponse):
            logger.error("%s Agent Fan-out 返回了无效响应类型：%s", agent_type.value, type(response).__name__)
            continue
        response.tool_traces.append(
            {
                "stage": "agent_fanout",
                "agent_type": agent_type.value,
                "intent_ids": list(scoped_request.intent_diagnostics.get("agent_scope", {}).get("intent_ids", [])),
                "sub_request_ids": list(scoped_request.intent_diagnostics.get("agent_scope", {}).get("sub_request_ids", [])),
                "status": "success" if response.success else "failed",
            }
        )
        completed.append(response)
    return completed


class AgentOrchestrator:
    """负责 Fan-out、知识预检索、结果合成和 Trace 聚合的轻量编排入口。

    Args:
        agent_pool: 可注入的三个业务域 Agent 与确定性升级处理映射。
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
            return ResponseComposer._tool_facts_fallback(request, responses, primary_agent)
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
            return ResponseComposer._tool_facts_fallback(request, responses, primary_agent)

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
        merged_facts = CaseFacts.from_dict(request.case_facts)
        for response in valid:
            merged_facts.merge(response.case_facts)
        request.case_facts = merged_facts.to_dict()
        has_current_tool_facts = any(
            str(record.get("source_type", "")).startswith("tool:")
            and str(record.get("source_ref", "")) == str(request.request_id)
            and record.get("confirmation_status") != "cleared"
            and record.get("value") is not None
            for records in merged_facts.facts.values()
            for record in records
        )
        failed_after_tool_success = any(
            not response.success
            and any(
                isinstance(trace, Mapping)
                and (trace.get("status") == "success" or trace.get("success") is True)
                for trace in response.tool_traces
            )
            for response in valid
        )
        request.requires_tool_fact_composition = bool(
            has_current_tool_facts and failed_after_tool_success
        )
        tools_used = _dedupe_preserving_order(
            [tool_name for response in valid for tool_name in response.tools_used]
        )
        tool_traces = [trace for response in valid for trace in response.tool_traces]
        retrieval_stage = [
            trace
            for trace in tool_traces
            if isinstance(trace, Mapping) and trace.get("retrieval_stage")
        ]
        diagnostics = dict(request.intent_diagnostics or {})
        diagnostics["retrieval_stage"] = retrieval_stage
        knowledge_blocked = any(
            bool(trace.get("knowledge_required"))
            and str(trace.get("status") or "") != "success"
            for trace in retrieval_stage
        )
        diagnostics["knowledge_retrieval_blocked"] = knowledge_blocked
        usage = _merge_usage(diagnostics.get("usage"), *(response.usage for response in successful))
        usage_by_stage: Dict[str, Dict[str, int]] = {}
        intent_usage = diagnostics.get("usage")
        if isinstance(intent_usage, Mapping):
            usage_by_stage = _merge_usage_by_stage({"intent": intent_usage})
        for response in successful:
            stage = f"agent_{response.agent_type.value}"
            usage_by_stage = _merge_usage_by_stage(
                usage_by_stage,
                {stage: response.usage},
            )
        primary_response = next(
            (response for response in successful if response.agent_type is decision.primary_agent),
            successful[0] if successful else None,
        )
        if knowledge_blocked:
            # 必需知识分支失败时跳过 ResponseComposer；只保留已成功领域分支的原答复，
            # 并明确说明未检索到依据的部分不作猜测。
            grounded_answers = [
                str(response.content).strip()
                for response in successful
                if str(response.content or "").strip()
            ]
            response_text = "\n\n".join(
                [*grounded_answers, _KNOWLEDGE_UNAVAILABLE_RESPONSE]
            )
            result_agent_type = (
                primary_response.agent_type
                if primary_response is not None
                else decision.primary_agent or AgentType.GENERAL
            )
            agent_types = [response.agent_type for response in successful]
        elif primary_response is None and not request.requires_tool_fact_composition:
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
            composer_usage_getter = getattr(self.composer, "usage_for", None)
            composer_usage = (
                composer_usage_getter(request.request_id)
                if callable(composer_usage_getter)
                else {}
            )
            usage = _merge_usage(usage, composer_usage)
            usage_by_stage = _merge_usage_by_stage(
                usage_by_stage,
                {"composer": composer_usage},
            )
            result_agent_type = (
                primary_response.agent_type
                if primary_response is not None
                else decision.primary_agent or AgentType.GENERAL
            )
            agent_types = [response.agent_type for response in successful]

        # 所有分支共用同一答复出口：安全检查、已说内容去重和槽位追问校验。
        # 知识检索失败/Agent 失败分支也要经过守卫，但这里不增加模型调用。
        response_text = customer_friendly_response(
            _customer_safe_text(response_text, request),
            intent=request.intent,
            emotion=str(request.intent_diagnostics.get("emotion", "neutral")),
        )
        # 后续承诺摘要必须以真正发给消费者的文本为准。
        response_text = guard_response(response_text)
        response_text, removed_clauses = filter_repeated_answer_content(
            response_text,
            list(request.history or []),
            current_message=request.message,
            current_message_ref=request.source_message_ref,
            case_facts=request.case_facts,
        )
        field_state = build_business_field_state(request)
        response_text, removed_field_questions = filter_redundant_field_questions(
            response_text,
            field_state["known_fields"],
            allowed_fields=(
                set(field_state["askable_text_fields"])
                if field_state["blocking_fields"]
                else None
            ),
        )
        if removed_clauses:
            logger.info(
                "已移除与历史答复重复的内容：request_id=%s clauses=%s",
                request.request_id,
                removed_clauses,
            )
        if removed_field_questions:
            logger.info(
                "已移除已知或超出业务必需范围的追问：request_id=%s fields=%s",
                request.request_id,
                removed_field_questions,
            )
        if not response_text:
            response_text = "前面说明过的内容我不再重复。您可以告诉我这次希望继续处理的具体事项。"
        urgency_policy = score_temporary_urgency(
            request.message,
            load_handoff_urgency_config(),
            intent=str(getattr(request.intent, "value", request.intent) or ""),
            intents=_all_request_intent_ids(request),
        )
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
            usage=usage,
            usage_by_stage=usage_by_stage,
            # 转人工单独由配置规则和 Agent 建议决定，不再从 P0/P1 紧急度反推。
            escalated=bool(
                urgency_policy["automatic_handoff"]
                or request.intent is IntentCategory.HUMAN_HANDOFF
            ),
            agent_recommended_handoff=any(response.escalate for response in successful),
            case_facts=merged_facts.to_dict(),
        )
        handoff_required = bool(result.escalated or result.agent_recommended_handoff)
        merged_facts.record(
            "ai_response", result.response, source_type="assistant_message",
            source_ref=request.request_id, confidence=1.0,
            confirmation_status="confirmed",
        )
        handoff = build_handoff_payload(
            message=request.message,
            intent=request.intent,
            entities=request.entities,
            emotion=str(request.intent_diagnostics.get("emotion", "neutral")),
            emotion_intensity=float(request.intent_diagnostics.get("emotion_intensity", 0.0) or 0.0),
            urgency=request.urgency,
            should_handoff=handoff_required,
            conv_id=request.conv_id,
            buyer_nickname=request.user_id,
            case_facts=merged_facts,
            secondary_intents=request.secondary_intents,
            sub_requests=request.sub_requests,
            ai_response=result.response,
        )
        suggested_reply_usage: Dict[str, int] = {}
        suggestion_composer = getattr(self.composer, "compose_human_suggested_reply", None)
        if handoff_required and handoff.get("summary") and callable(suggestion_composer):
            try:
                suggested_reply, suggested_reply_usage = await suggestion_composer(
                    request,
                    handoff["summary"],
                    list(handoff.get("suggested_replies", [])),
                )
                if suggested_reply:
                    handoff["suggested_replies"] = [suggested_reply]
            except Exception as exc:
                logger.warning("人工建议回复生成失败，保留基础话术：%s", exc)
        result = replace(
            result,
            case_facts=merged_facts.to_dict(),
            usage=_merge_usage(result.usage, suggested_reply_usage),
            usage_by_stage=_merge_usage_by_stage(
                result.usage_by_stage,
                {"handoff_suggestion": suggested_reply_usage},
            ),
            handoff_summary=handoff.get("summary"),
            ticket_draft=handoff.get("ticket_draft"),
            suggested_replies=list(handoff.get("suggested_replies", [])),
        )
        self._recent_traces.append(
            {
                "request_id": result.request_id,
                "retrieval_stage": retrieval_stage,
                "tools_used": list(result.tools_used),
                "tool_traces": list(result.tool_traces),
                "executed_agents": [agent.value for agent in result.agent_types],
                "usage": dict(result.usage),
                "usage_by_stage": {
                    stage: dict(stage_usage)
                    for stage, stage_usage in result.usage_by_stage.items()
                },
            }
        )
        return result

    async def run(self, request: Request) -> OrchestratorResult:
        """计算路由并进入 Step 6.2 Fan-out 主链路。"""

        # 明确要求人工或已达到安全临界级别时，不再等待升级 Agent 的模型调用。
        # 交接摘要、工单和人工话术均由确定性契约生成，先把可执行结果返回给用户。
        if request.intent is IntentCategory.HUMAN_HANDOFF:
            handoff_intent = request.intent
            text = request.message or ""
            acknowledgement = "为您转接人工，请您稍等。"
            facts = CaseFacts.from_dict(request.case_facts)
            facts.record(
                "ai_response", acknowledgement, source_type="assistant_message",
                source_ref=request.request_id, confidence=1.0,
                confirmation_status="confirmed",
            )
            if any(word in text for word in ("发红", "红肿", "发痒", "烂脸", "肿胀", "过敏")):
                handoff_intent = IntentCategory.ADVERSE_REACTION
            handoff = build_handoff_payload(
                message=text,
                intent=handoff_intent,
                entities=request.entities,
                emotion=str(request.intent_diagnostics.get("emotion", "neutral")),
                emotion_intensity=float(request.intent_diagnostics.get("emotion_intensity", 0.0) or 0.0),
                urgency=request.urgency or UrgencyLevel.HIGH,
                should_handoff=True,
                conv_id=request.conv_id,
                buyer_nickname=request.user_id,
                case_facts=facts,
                secondary_intents=request.secondary_intents,
                sub_requests=request.sub_requests,
                ai_response=acknowledgement,
            )
            return OrchestratorResult(
                request_id=request.request_id,
                response=acknowledgement,
                agent_type=AgentType.ESCALATION,
                intent=handoff_intent,
                escalated=True,
                agent_recommended_handoff=True,
                agent_types=[AgentType.ESCALATION],
                primary_agent=AgentType.ESCALATION,
                routing_reason="安全风险或明确人工诉求，使用确定性快速交接",
                routing_confidence=1.0,
                intent_diagnostics=dict(request.intent_diagnostics),
                knowledge_required=False,
                handoff_summary=handoff.get("summary"),
                ticket_draft=handoff.get("ticket_draft"),
                suggested_replies=list(handoff.get("suggested_replies", [])),
                case_facts=facts.to_dict(),
            )

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
                usage=_merge_usage(request.intent_diagnostics.get("usage")),
                usage_by_stage=_merge_usage_by_stage(
                    {"intent": request.intent_diagnostics.get("usage", {})}
                ),
            )
        return await self.run_parallel(request, decision)

    def get_recent_traces(self, limit: int = 20) -> List[Dict[str, Any]]:
        """返回最近的内存 Trace 快照，不持久化任何敏感凭证。"""

        bounded = max(1, min(int(limit or 20), len(self._recent_traces)))
        return list(reversed(list(self._recent_traces)[-bounded:]))
