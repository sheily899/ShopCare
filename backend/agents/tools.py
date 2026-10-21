"""ShopCare Agent 工具契约与美妆业务工具。

本模块属于 Agent 能力层，负责定义安全的 Tool Use 契约和当前美妆业务工具；
工具只读取已知上下文或生成本地草稿，不连接外部订单、支付或坐席系统。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import inspect
import json
import math
from pathlib import Path
from numbers import Real
from typing import Any, Callable, Dict, Mapping, Optional

from agents.case_facts import CaseFacts

Handler = Callable[..., Any]


@dataclass(frozen=True)
class AgentToolSpec:
    """封装一个 Agent 工具的元信息、JSON Schema 与执行函数。

    Args:
        name: 工具在 Anthropic Tool Use 中使用的唯一名称。
        description: 面向模型的工具用途说明。
        input_schema: JSON Schema 对象；必须显式禁止额外属性。
        handler: 接收参数字典的同步或异步执行函数。

    Raises:
        TypeError: 工具名称、描述、schema 或 handler 类型不符合契约时抛出。
        ValueError: schema 不是 object，或未显式设置
            ``additionalProperties=False`` 时抛出。
    """

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Handler

    def __post_init__(self) -> None:
        """在注册前锁定 Anthropic 工具声明的安全边界。"""

        if not isinstance(self.name, str) or not self.name.strip():
            raise TypeError("工具 name 必须是非空字符串")
        if not isinstance(self.description, str) or not self.description.strip():
            raise TypeError("工具 description 必须是非空字符串")
        if not isinstance(self.input_schema, dict):
            raise TypeError("工具 input_schema 必须是字典")
        if self.input_schema.get("type") != "object":
            raise ValueError('工具 input_schema.type 必须为 "object"')
        # 显式关闭额外字段，避免模型通过未声明参数越过工具契约边界。
        if self.input_schema.get("additionalProperties") is not False:
            raise ValueError("工具 input_schema 必须显式设置 additionalProperties=False")
        if not callable(self.handler):
            raise TypeError("工具 handler 必须可调用")

    def to_anthropic_tool(self) -> Dict[str, Any]:
        """导出 Anthropic SDK 使用的工具声明。

        Returns:
            包含 ``name``、``description`` 和独立 schema 副本的字典。
        """

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": deepcopy(self.input_schema),
        }

    async def execute(
        self,
        params: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """执行工具处理器，并将未预期异常转换为安全错误结果。

        Args:
            params: 模型传入的工具参数。
            context: 可选的当前请求内存快照，仅传给声明支持该参数的 handler。

        Returns:
            handler 的结果；异常时返回 ``success=False`` 的错误字典。
        """

        try:
            signature = inspect.signature(self.handler)
            supports_context = "context" in signature.parameters or any(
                parameter.kind is parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            if supports_context:
                result = self.handler(params, context=context)
            else:
                result = self.handler(params)
            if inspect.isawaitable(result):
                return await result
            return result
        except Exception as exc:  # 工具边界必须隔离异常，避免模型调用链崩溃。
            return {"success": False, "error": f"工具执行失败: {exc}"}


def make_tool(
    name: str,
    description: str,
    input_schema: Dict[str, Any],
    handler: Handler,
) -> AgentToolSpec:
    """创建并立即校验一个 ``AgentToolSpec``。

    Args:
        name: 工具名称。
        description: 工具描述。
        input_schema: 符合 object 与 additionalProperties=false 约束的 schema。
        handler: 同步或异步执行函数。

    Returns:
        校验完成的工具规格实例。
    """

    return AgentToolSpec(name, description, input_schema, handler)


def _error(message: str) -> Dict[str, Any]:
    """构造通用工具的确定性错误结果。"""

    return {"success": False, "error": message}


def _mask_user_id(user_id: str) -> str:
    """对用户标识做不可逆的展示级脱敏。"""

    if not user_id or user_id == "anonymous":
        return "anonymous"
    if len(user_id) <= 2:
        return "*" * len(user_id)
    if len(user_id) <= 4:
        return f"{user_id[0]}***"
    return f"{user_id[:2]}***{user_id[-2:]}"


async def inspect_request_context(
    params: Any,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """返回当前请求的只读、脱敏上下文快照。

    Args:
        params: 工具入参，契约要求为空对象。
        context: 当前请求的内存上下文，不能包含外部业务写操作句柄。

    Returns:
        包含 request_id、intent、脱敏 user_id 与系统时间戳的字典；入参异常时
        返回 ``success=False`` 的安全错误。
    """

    if not isinstance(params, dict):
        return _error("工具参数必须是对象")
    if params:
        return _error("inspect_request_context 不接受额外参数")
    source = context if isinstance(context, Mapping) else {}
    request_id = str(source.get("request_id", ""))
    intent = str(source.get("intent", "other"))
    user_id = str(source.get("user_id", "anonymous"))
    # 测试或上游已有可信时间戳时复用它，否则使用当前 UTC 时间完成快照。
    timestamp = str(source.get("timestamp") or datetime.now(timezone.utc).isoformat())
    return {
        "success": True,
        "request_id": request_id,
        "intent": intent,
        "user_id": _mask_user_id(user_id),
        "timestamp": timestamp,
    }


from intent.beauty_intent_ticket_registry import BEAUTY_INTENT_TICKET_BY_ID


_REQUIRED_FIELDS_BY_INTENT: Dict[str, tuple[str, ...]] = {
    intent_id: tuple(item["blocking_fields"])
    for intent_id, item in BEAUTY_INTENT_TICKET_BY_ID.items()
}

_FIELD_LABELS = {
    "order_id": "订单号",
    "return_reason": "退货原因",
    "opened_or_used": "商品拆封/使用状态",
    "missing_item": "缺少的商品",
    "quantity": "数量",
    "wrong_item": "收到的商品",
    "shade": "色号",
    "damaged_area": "破损部位",
    "symptom": "不适症状",
    "tracking_no": "物流单号",
    "signed_info": "签收情况",
    "intercept_or_change": "拦截或改地址需求",
    "medical_visit": "就医情况",
    "refund_amount": "退款金额",
    "price_difference": "价差",
}
_EMPTY_FIELD_VALUES = frozenset({
    "", "unknown", "null", "none", "未知", "不清楚", "不确定", "不知道", "未提供", "待确认"
})


def _has_field_value(values: Any) -> bool:
    items = values if isinstance(values, (list, tuple, set)) else [values]
    return any(
        value is not None
        and str(value).strip().casefold() not in _EMPTY_FIELD_VALUES
        for value in items
    )


async def suggest_required_fields(
    params: Any,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """根据意图给出下一轮追问的字段建议。

    Args:
        params: 必须包含非空字符串字段 ``intent`` 的对象。
        context: 当前请求的实体与 CaseFacts，只读取已在会话中出现的字段事实。

    Returns:
        包含 ``required_fields`` 与中文 ``suggestion`` 的字典；参数非法时返回
        ``success=False`` 的安全错误。
    """

    if not isinstance(params, dict):
        return _error("工具参数必须是对象")
    intent = params.get("intent")
    if not isinstance(intent, str) or not intent.strip():
        return _error("intent 必须是非空字符串")
    if not isinstance(context, Mapping) or not any(
        key in context for key in ("entities", "case_facts")
    ):
        return _error("当前请求字段上下文不可用，无法判断哪些字段仍缺失")

    requested_intent = intent.strip().lower()
    has_authoritative_intent_context = "intent_ids" in context or "intent" in context
    context_intents = context.get("intent_ids")
    if not isinstance(context_intents, (list, tuple)):
        context_intents = [context.get("intent")]
    active_intents = list(dict.fromkeys(
        str(value).strip().lower()
        for value in context_intents
        if isinstance(value, str)
        and value.strip().lower() in _REQUIRED_FIELDS_BY_INTENT
    ))
    # 上下文中的注册表 ID 是权威来源；模型传入的自然语言意图描述仅用于旧入口兼容。
    if (
        not active_intents
        and not has_authoritative_intent_context
        and requested_intent in _REQUIRED_FIELDS_BY_INTENT
    ):
        active_intents = [requested_intent]
    if not active_intents:
        return _error("当前意图无法映射到已注册的业务字段规则")

    normalized_intent = active_intents[0]
    required_fields = list(dict.fromkeys(
        field
        for active_intent in active_intents
        for field in _REQUIRED_FIELDS_BY_INTENT.get(active_intent, ())
    ))
    facts = CaseFacts.from_dict(context.get("case_facts"))
    effective_values = facts.effective_values(
        context.get("entities") if isinstance(context.get("entities"), Mapping) else None
    )
    tracked_fields = list(dict.fromkeys(
        required_fields
        + [
            field
            for active_intent in active_intents
            for field in BEAUTY_INTENT_TICKET_BY_ID[active_intent].get("ticket_fields", [])
        ]
        + list(effective_values)
    ))
    known_fields = {
        field: list(effective_values[field])
        for field in tracked_fields
        if _has_field_value(effective_values.get(field, []))
    }
    missing_fields = [field for field in required_fields if field not in known_fields]
    if missing_fields:
        missing_labels = [
            _FIELD_LABELS.get(field, field) for field in missing_fields
        ]
        known_labels = [
            _FIELD_LABELS.get(field, field) for field in known_fields
        ]
        known_text = f"已从上下文取得：{'、'.join(known_labels)}；" if known_labels else ""
        suggestion = (
            f"{known_text}只建议补充仍缺少的字段：{'、'.join(missing_labels)}；"
            "不要再次询问已取得的字段。"
        )
    elif required_fields:
        suggestion = "该意图的必需字段已从上下文取得，不要重复追问，可继续核验或回答。"
    else:
        suggestion = "当前意图没有固定阻塞字段；只有处理问题确实需要且上下文缺少的信息才可追问。"
    return {
        "success": True,
        "intent": normalized_intent,
        "intents": active_intents,
        # 兼容旧消费者字段名；现在返回的明确是尚缺字段，而非全量必填字段。
        "required_fields": missing_fields,
        "missing_fields": missing_fields,
        "known_fields": known_fields,
        "suggestion": suggestion,
    }


async def lookup_order(params: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """读取 Demo Order DB 中的订单事实；未命中时明确返回，不编造状态。"""
    if not isinstance(params, dict):
        return _error("工具参数必须是对象")
    order_id = params.get("order_id")
    if order_id is not None and not isinstance(order_id, str):
        return _error("order_id 必须是字符串")
    source = context if isinstance(context, Mapping) else {}
    entities = source.get("entities") if isinstance(source.get("entities"), Mapping) else {}
    entity_order_id = (entities or {}).get("order_id")
    if isinstance(entity_order_id, (list, tuple)):
        entity_order_id = entity_order_id[-1] if entity_order_id else ""
    resolved = str(order_id or entity_order_id or "").strip()
    orders_path = Path(__file__).resolve().parents[1] / "data" / "demo_orders.json"
    try:
        orders = json.loads(orders_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        orders = []
    user_id = str(source.get("user_id") or "")
    matched = next((item for item in orders if item.get("order_id") == resolved and (not user_id or item.get("user_id") == user_id)), None)
    return {
        "success": True,
        "available": matched is not None,
        "order_id": resolved or None,
        "source": "demo_order_db",
        "order": matched,
        "message": "已查到演示订单。" if matched else ("暂未查到对应订单，请人工继续核对。" if resolved else "缺少订单号，需先澄清。"),
    }


async def lookup_logistics(params: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """从演示订单源读取已知物流字段，不生成虚假物流节点。"""
    if not isinstance(params, dict):
        return _error("工具参数必须是对象")
    tracking_no = params.get("tracking_no")
    order_id = params.get("order_id")
    if tracking_no is not None and not isinstance(tracking_no, str):
        return _error("tracking_no 必须是字符串")
    if order_id is not None and not isinstance(order_id, str):
        return _error("order_id 必须是字符串")
    source = context if isinstance(context, Mapping) else {}
    entities = source.get("entities") if isinstance(source.get("entities"), Mapping) else {}
    context_order = entities.get("order_id")
    context_tracking = entities.get("tracking_no")
    if isinstance(context_order, (list, tuple)):
        context_order = context_order[-1] if context_order else ""
    if isinstance(context_tracking, (list, tuple)):
        context_tracking = context_tracking[-1] if context_tracking else ""
    resolved_order = str(order_id or context_order or "").strip()
    resolved_tracking = str(tracking_no or context_tracking or "").strip()
    if not resolved_order and not resolved_tracking:
        return {"success": False, "available": False, "message": "缺少订单号或物流单号，无法查询物流。"}
    orders_path = Path(__file__).resolve().parents[1] / "data" / "demo_orders.json"
    try:
        orders = json.loads(orders_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        orders = []
    user_id = str(source.get("user_id") or "")
    matched = next((
        item for item in orders
        if (not user_id or item.get("user_id") == user_id)
        and ((resolved_order and item.get("order_id") == resolved_order)
             or (resolved_tracking and item.get("tracking_no") == resolved_tracking))
    ), None)
    if not matched:
        return {"success": True, "available": False, "order_id": resolved_order or None,
                "tracking_no": resolved_tracking or None, "source": "demo_order_db",
                "message": "演示订单源中没有匹配的物流记录。"}
    return {
        "success": True,
        "available": bool(matched.get("tracking_no") or matched.get("logistics_status")),
        "order_id": matched.get("order_id"),
        "tracking_no": matched.get("tracking_no"),
        "logistics_status": matched.get("logistics_status"),
        "carrier": matched.get("carrier"),
        "signed_info": matched.get("signed_info"),
        "received_by": matched.get("received_by"),
        "source": "demo_order_db",
        "message": "已返回演示订单源中已有的物流字段；未提供的轨迹不作推测。",
    }


async def create_ticket_draft(params: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """根据美妆意图注册表生成一张结构化工单草稿，不直接提交正式工单。"""
    if not isinstance(params, dict):
        return _error("工具参数必须是对象")
    intent = str(params.get("intent") or "").strip().lower()
    contract = BEAUTY_INTENT_TICKET_BY_ID.get(intent)
    if contract is None or not contract["needs_ticket"]:
        return _error("该意图不允许创建工单")
    supplied_fields = params.get("fields") or {}
    if not isinstance(supplied_fields, dict):
        return _error("fields 必须是对象")
    source = context if isinstance(context, Mapping) else {}
    entities = source.get("entities") if isinstance(source.get("entities"), Mapping) else {}
    fields = {}
    for name in contract["ticket_fields"]:
        value = supplied_fields.get(name)
        if value is None:
            value = entities.get(name)
        if isinstance(value, (list, tuple)):
            value = value[-1] if value else None
        if value is not None:
            fields[name] = value
    return {
        "success": True,
        "status": "draft",
        "needs_review": True,
        "intent": intent,
        "ticket_type": contract["ticket_type"],
        "risk_level": contract["risk_level"],
        "fields": fields,
        "missing_blocking_fields": [name for name in contract["blocking_fields"] if not fields.get(name)],
    }


async def check_refund_status(params: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """读取演示订单库内已知退款字段；未接入真实支付平台时不外推到账状态。"""
    if not isinstance(params, dict):
        return _error("工具参数必须是对象")
    order_id = params.get("order_id")
    if order_id is not None and not isinstance(order_id, str):
        return _error("order_id 必须是字符串")
    source = context if isinstance(context, Mapping) else {}
    entities = source.get("entities") if isinstance(source.get("entities"), Mapping) else {}
    context_order = entities.get("order_id")
    if isinstance(context_order, (list, tuple)):
        context_order = context_order[-1] if context_order else ""
    resolved = str(order_id or context_order or "").strip()
    if not resolved:
        return {"success": False, "available": False, "message": "缺少订单号，无法核对退款记录。"}
    orders_path = Path(__file__).resolve().parents[1] / "data" / "demo_orders.json"
    try:
        orders = json.loads(orders_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        orders = []
    user_id = str(source.get("user_id") or "")
    matched = next((item for item in orders if item.get("order_id") == resolved and (not user_id or item.get("user_id") == user_id)), None)
    if not matched:
        return {"success": True, "available": False, "order_id": resolved,
                "refund_status": None, "source": "demo_order_db",
                "message": "演示订单库中没有匹配订单的退款记录。"}
    return {"success": True, "available": matched.get("refund_status") is not None,
            "order_id": resolved, "refund_status": matched.get("refund_status"),
            "source": "demo_order_db", "message": "已读取演示订单库退款字段，不代表真实支付平台到账结果。"}


async def calculate_price_difference(params: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """计算价保差额，不判断是否符合价保规则。"""
    if not isinstance(params, dict):
        return _error("工具参数必须是对象")
    paid = params.get("paid_amount")
    current = params.get("current_amount")
    if not isinstance(paid, Real) or isinstance(paid, bool) or not isinstance(current, Real) or isinstance(current, bool):
        return _error("paid_amount 和 current_amount 必须是数值")
    return {"success": True, "price_difference": max(0.0, float(paid) - float(current)), "read_only": True}


async def record_risk_event(params: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """生成待写入风险事件的结构，不直接写入外部系统。"""
    if not isinstance(params, dict):
        return _error("工具参数必须是对象")
    if not params.get("risk_type"):
        return _error("risk_type 必须存在")
    return {"success": True, "status": "pending_review", "risk_type": str(params["risk_type"]), "level": str(params.get("level") or "P2"), "reason": str(params.get("reason") or "")}


INSPECT_REQUEST_CONTEXT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": False,
}

SUGGEST_REQUIRED_FIELDS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "description": "兼容字段；系统以请求上下文中的规范意图 ID 为准，不采用自然语言转述。",
        },
    },
    "required": ["intent"],
    "additionalProperties": False,
}

INSPECT_REQUEST_CONTEXT_TOOL = make_tool(
    "inspect_request_context",
    "返回当前请求的只读脱敏上下文快照，不调用外部业务系统。",
    INSPECT_REQUEST_CONTEXT_SCHEMA,
    inspect_request_context,
)

SUGGEST_REQUIRED_FIELDS_TOOL = make_tool(
    "suggest_required_fields",
    "读取当前会话已记录的字段事实与规范业务意图，只返回仍缺少的必需字段；不重新询问已知项，也不执行外部写操作。",
    SUGGEST_REQUIRED_FIELDS_SCHEMA,
    suggest_required_fields,
)

BEAUTY_LOOKUP_ORDER_TOOL = make_tool(
    "lookup_order", "读取订单号和当前请求中的订单事实，不伪造订单状态。",
    {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": [], "additionalProperties": False},
    lookup_order,
)
BEAUTY_LOOKUP_LOGISTICS_TOOL = make_tool(
    "lookup_logistics", "读取物流单号和已知物流字段，不生成虚假轨迹。",
    {"type": "object", "properties": {"order_id": {"type": "string"}, "tracking_no": {"type": "string"}}, "required": [], "additionalProperties": False},
    lookup_logistics,
)
BEAUTY_CREATE_TICKET_TOOL = make_tool(
    "create_ticket_draft", "按美妆工单契约生成待客服审核的工单草稿。",
    {"type": "object", "properties": {"intent": {"type": "string"}, "fields": {"type": "object"}}, "required": ["intent"], "additionalProperties": False},
    create_ticket_draft,
)
BEAUTY_REFUND_STATUS_TOOL = make_tool(
    "check_refund_status", "读取已知退款状态，不确认到账或退款结果。",
    {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": [], "additionalProperties": False},
    check_refund_status,
)
BEAUTY_PRICE_DIFFERENCE_TOOL = make_tool(
    "calculate_price_difference", "计算价保差额，不判断价保资格。",
    {"type": "object", "properties": {"paid_amount": {"type": "number"}, "current_amount": {"type": "number"}}, "required": ["paid_amount", "current_amount"], "additionalProperties": False},
    calculate_price_difference,
)
BEAUTY_RISK_EVENT_TOOL = make_tool(
    "record_risk_event", "生成待人工审核的美妆风险事件记录。",
    {"type": "object", "properties": {"risk_type": {"type": "string"}, "level": {"type": "string"}, "reason": {"type": "string"}}, "required": ["risk_type"], "additionalProperties": False},
    record_risk_event,
)

BEAUTY_TOOLS: Dict[str, AgentToolSpec] = {
    item.name: item
    for item in (
        INSPECT_REQUEST_CONTEXT_TOOL,
        SUGGEST_REQUIRED_FIELDS_TOOL,
        BEAUTY_LOOKUP_ORDER_TOOL,
        BEAUTY_LOOKUP_LOGISTICS_TOOL,
        BEAUTY_CREATE_TICKET_TOOL,
        BEAUTY_REFUND_STATUS_TOOL,
        BEAUTY_PRICE_DIFFERENCE_TOOL,
        BEAUTY_RISK_EVENT_TOOL,
    )
}


def beauty_tools() -> Dict[str, AgentToolSpec]:
    """返回当前美妆业务工具白名单的浅拷贝。"""
    return dict(BEAUTY_TOOLS)

