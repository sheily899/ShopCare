"""美妆意图到工单、字段和工具白名单的统一业务契约。

本表是后续 Profile、Skill、工具注册和工单服务的唯一输入。
这里只描述业务边界，不执行模型调用或创建工单。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping


# 所有工单共享的系统字段与业务字段分开维护：系统字段由后端写入，不能由模型抽取。
SYSTEM_TICKET_FIELDS = ["ticket_id", "conv_id", "buyer_nickname", "created_at"]
COMMON_TICKET_FIELDS = ["order_id", "product_name", "sku"]
KNOWN_TOOLS = {
    "inspect_request_context", "suggest_required_fields", "search_knowledge_base",
    "lookup_order", "lookup_logistics", "create_ticket_draft", "check_refund_status",
    "calculate_price_difference", "record_risk_event",
}

# 知识策略是业务边界，不再由“工具列表是否包含搜索工具”反向推断。
KNOWLEDGE_MODE_BY_INTENT = {
    "ingredient_fit": "required", "shade_match": "required", "usage_howto": "required",
    "product_info": "required", "promo_rules": "required", "pregnancy_safety": "required",
    "order_status": "never", "logistics_stalled": "when_relevant",
    "signed_not_received": "when_relevant", "intercept_change_address": "required",
    "missing_item": "when_relevant", "wrong_item": "when_relevant",
    "damaged_on_arrival": "required", "return_refund": "required",
    "adverse_reaction": "required", "suspected_abnormal_refund": "never",
    "price_protection": "required", "refund_not_received": "never",
    "shipping_fee_reimbursement": "required",
}

DEPARTMENTS_BY_INTENT = {
    "signed_not_received": ["物流"], "intercept_change_address": ["物流"],
    "missing_item": ["仓储"], "wrong_item": ["仓储"],
    "damaged_on_arrival": ["质量", "仓储"], "return_refund": ["售后", "财务"],
    "adverse_reaction": ["质量", "售后"],
    "suspected_abnormal_refund": ["财务", "风控"],
    "price_protection": ["财务"], "refund_not_received": ["财务"],
    "shipping_fee_reimbursement": ["财务"],
}


def _entry(
    intent_id: str,
    agent: str,
    needs_ticket: bool,
    ticket_type: str | None,
    blocking_fields: List[str],
    ticket_fields: List[str],
    tools: List[str],
    risk_level: str = "P3",
    knowledge_required: bool | None = None,
    knowledge_mode: str | None = None,
) -> Dict[str, Any]:
    mode = knowledge_mode or KNOWLEDGE_MODE_BY_INTENT.get(intent_id)
    if mode not in {"required", "when_relevant", "never"}:
        raise ValueError(f"意图缺少明确知识策略: {intent_id}")
    tools = [tool for tool in tools if tool != "search_knowledge_base"]
    tools.append("inspect_request_context")
    if mode != "never":
        tools.append("search_knowledge_base")
    if blocking_fields and "suggest_required_fields" not in tools:
        tools = [*tools, "suggest_required_fields"]
    tools = list(dict.fromkeys(tools))
    return {
        "intent_id": intent_id,
        "agent": agent,
        "needs_ticket": needs_ticket,
        "ticket_type": ticket_type,
        "blocking_fields": blocking_fields,
        "ticket_fields": ticket_fields,
        "allowed_tools": tools,
        "knowledge_mode": mode,
        # 兼容旧调用方；新逻辑应读取 knowledge_mode。
        "knowledge_required": mode == "required" if knowledge_required is None else knowledge_required,
        "risk_level": risk_level,
        # 缺少必填字段只表示尚需追问/补录；转人工由统一升级规则单独判定。
        "missing_fields_action": "clarify",
        "departments": list(DEPARTMENTS_BY_INTENT.get(intent_id, [])),
    }


BEAUTY_INTENT_TICKET_REGISTRY: List[Dict[str, Any]] = [
    _entry("ingredient_fit", "consult", False, None, [], [], ["search_knowledge_base"]),
    _entry("shade_match", "consult", False, None, [], [], ["search_knowledge_base"]),
    _entry("usage_howto", "consult", False, None, [], [], ["search_knowledge_base"]),
    _entry("product_info", "consult", False, None, [], [], ["search_knowledge_base"]),
    _entry("promo_rules", "consult", False, None, [], [], ["search_knowledge_base", "lookup_order"]),
    _entry("pregnancy_safety", "consult", False, None, [], [], ["search_knowledge_base"], "P2"),
    _entry("order_status", "order", False, None, ["order_id"], [], ["lookup_order", "lookup_logistics"]),
    _entry("logistics_stalled", "order", False, None, ["order_id"], [], ["lookup_order", "lookup_logistics"], "P2"),
    _entry(
        "signed_not_received", "order", True, "logistics", ["order_id", "signed_info"],
        COMMON_TICKET_FIELDS + ["tracking_no", "carrier", "logistics_status", "signed_info", "received_by"],
        ["lookup_order", "lookup_logistics", "create_ticket_draft"], "P1",
    ),
    _entry(
        "intercept_change_address", "order", True, "logistics", ["order_id", "tracking_no", "intercept_or_change"],
        COMMON_TICKET_FIELDS + ["tracking_no", "carrier", "logistics_status", "intercept_or_change", "new_address"],
        ["lookup_order", "lookup_logistics", "create_ticket_draft"], "P1",
    ),
    _entry(
        "missing_item", "order", True, "reissue_exchange", ["order_id", "missing_item", "quantity"],
        COMMON_TICKET_FIELDS + ["missing_item", "quantity", "tracking_no"],
        ["lookup_order", "create_ticket_draft"], "P2",
    ),
    _entry(
        "wrong_item", "order", True, "reissue_exchange", ["order_id", "wrong_item", "shade", "quantity"],
        COMMON_TICKET_FIELDS + ["wrong_item", "shade", "quantity", "tracking_no"],
        ["lookup_order", "create_ticket_draft"], "P2",
    ),
    _entry(
        "damaged_on_arrival", "order", True, "reissue_exchange", ["order_id", "damaged_area"],
        COMMON_TICKET_FIELDS + ["damaged_area", "tracking_no"],
        ["lookup_order", "create_ticket_draft"], "P2",
    ),
    _entry(
        "return_refund", "after_sales", True, "return_refund", ["order_id", "return_reason", "opened_or_used"],
        COMMON_TICKET_FIELDS + ["return_reason", "return_scope", "opened_or_used", "gift_opened_or_used", "refund_amount", "refund_status"],
        ["lookup_order", "search_knowledge_base", "create_ticket_draft"], "P2",
    ),
    _entry(
        "adverse_reaction", "after_sales", True, "adverse_reaction",
        ["order_id", "symptom"],
        COMMON_TICKET_FIELDS + ["symptom", "symptom_area", "onset_time", "stopped_use", "medical_visit"],
        ["lookup_order", "search_knowledge_base", "create_ticket_draft", "record_risk_event"], "P1",
    ),
    _entry(
        "suspected_abnormal_refund", "after_sales", True, "refund_risk", ["order_id"],
        COMMON_TICKET_FIELDS + ["refund_amount", "refund_status"],
        ["lookup_order", "check_refund_status", "create_ticket_draft", "record_risk_event"], "P1",
    ),
    _entry(
        "price_protection", "after_sales", True, "price_protection", ["order_id", "price_difference"],
        COMMON_TICKET_FIELDS + ["refund_amount", "price_difference"],
        ["lookup_order", "search_knowledge_base", "calculate_price_difference", "create_ticket_draft"], "P2",
    ),
    _entry(
        "refund_not_received", "after_sales", True, "refund_payment", ["order_id"],
        COMMON_TICKET_FIELDS + ["refund_amount", "refund_status"],
        ["lookup_order", "check_refund_status", "create_ticket_draft"], "P2",
    ),
    _entry(
        "shipping_fee_reimbursement", "after_sales", True, "refund_payment", ["order_id", "refund_amount"],
        COMMON_TICKET_FIELDS + ["refund_amount", "refund_status"],
        ["lookup_order", "search_knowledge_base", "create_ticket_draft"], "P2",
    ),
]


BEAUTY_INTENT_TICKET_BY_ID = {
    item["intent_id"]: item for item in BEAUTY_INTENT_TICKET_REGISTRY
}


def validate_registry() -> None:
    """校验注册表与当前 19 意图契约一致。"""
    from intent.beauty_intent_registry import BEAUTY_INTENT_BY_ID

    expected = set(BEAUTY_INTENT_BY_ID)
    actual = set(BEAUTY_INTENT_TICKET_BY_ID)
    if expected != actual:
        raise ValueError(f"工单注册表意图不一致，缺失={expected - actual}，多出={actual - expected}")
    if any(item["agent"] not in {"consult", "order", "after_sales"} for item in BEAUTY_INTENT_TICKET_REGISTRY):
        raise ValueError("工单注册表包含未知 Agent")
    for item in BEAUTY_INTENT_TICKET_REGISTRY:
        if item["needs_ticket"] and not item["ticket_type"]:
            raise ValueError(f"建单意图缺少 ticket_type: {item['intent_id']}")
        unknown_tools = set(item["allowed_tools"]) - KNOWN_TOOLS
        if unknown_tools:
            raise ValueError(f"意图 {item['intent_id']} 含未知工具: {sorted(unknown_tools)}")
        if item["knowledge_mode"] == "never" and "search_knowledge_base" in item["allowed_tools"]:
            raise ValueError(f"禁止检索的意图暴露了知识库工具: {item['intent_id']}")
        if item["needs_ticket"] and not item["departments"]:
            raise ValueError(f"建单意图缺少部门路由: {item['intent_id']}")
        if item["needs_ticket"] and not set(item["blocking_fields"]) <= set(item["ticket_fields"]):
            raise ValueError(f"阻塞字段必须包含在工单字段中: {item['intent_id']}")
        if item["missing_fields_action"] != "clarify":
            raise ValueError(f"缺字段不得配置为自动转人工: {item['intent_id']}")


def should_prefetch_knowledge(intent_id: str, message: str = "") -> bool:
    """判断是否必须在 Agent 执行前检索；可选检索仍留给 Agent 自主调用。"""
    contract = BEAUTY_INTENT_TICKET_BY_ID.get(str(intent_id or ""))
    if contract is None:
        return False
    mode = contract["knowledge_mode"]
    if mode == "required":
        return True
    if mode == "never":
        return False
    text = str(message or "")
    relevance_markers = (
        "规则", "政策", "条件", "怎么处理", "怎么申请", "能不能退", "可以退吗",
        "赔付", "补偿", "时效", "承诺", "依据", "活动", "赠品", "小样",
    )
    return any(marker in text for marker in relevance_markers)


def policy_for_intents(intent_ids: List[str] | tuple[str, ...]) -> Dict[str, Any]:
    """合并复合诉求的可执行边界，保留首次出现顺序并去重工具与字段。"""
    policies = [BEAUTY_INTENT_TICKET_BY_ID[intent_id] for intent_id in dict.fromkeys(intent_ids)
                if intent_id in BEAUTY_INTENT_TICKET_BY_ID]
    if not policies:
        return {"allowed_tools": ["inspect_request_context"], "knowledge_mode": "never"}
    modes = {item["knowledge_mode"] for item in policies}
    mode = "required" if "required" in modes else "when_relevant" if "when_relevant" in modes else "never"
    return {
        "allowed_tools": list(dict.fromkeys(tool for item in policies for tool in item["allowed_tools"])),
        "knowledge_mode": mode,
        "blocking_fields": list(dict.fromkeys(field for item in policies for field in item["blocking_fields"])),
        "ticket_fields": list(dict.fromkeys(field for item in policies for field in item["ticket_fields"])),
        "needs_ticket": any(item["needs_ticket"] for item in policies),
        "departments": list(dict.fromkeys(department for item in policies for department in item["departments"])),
        # 复合意图仍由每个意图的槽位策略追问，不能因其中某字段未齐而升级人工。
        "missing_fields_action": "clarify",
        "risk_level": min((item["risk_level"] for item in policies), key=lambda value: {"P0": 0, "P1": 1, "P2": 2, "P3": 3}.get(value, 4)),
    }


validate_registry()

