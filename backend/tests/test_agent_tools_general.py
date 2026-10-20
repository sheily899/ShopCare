"""Step 5.1 通用 Agent 工具契约与只读行为测试。"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

import pytest

from agents.tools import (
    AgentToolSpec,
    general_tools,
    inspect_request_context,
    lookup_logistics,
    lookup_order,
    check_refund_status,
    create_ticket_draft,
    make_tool,
    suggest_required_fields,
)


def _run(coro: Any) -> Any:
    """在不引入 pytest-asyncio 的前提下运行异步工具处理器。"""

    return asyncio.run(coro)


def _valid_schema(properties: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """构造满足 Anthropic Tool Use 约束的最小对象 schema。"""

    return {
        "type": "object",
        "properties": properties or {},
        "required": [],
        "additionalProperties": False,
    }


def test_agent_tool_spec_enforces_object_schema_and_anthropic_shape() -> None:
    """工具声明必须显式关闭额外字段，并能导出 Anthropic 结构。"""

    with pytest.raises(ValueError, match="type"):
        AgentToolSpec("invalid", "无效工具", {"type": "string"}, lambda _: None)

    with pytest.raises(ValueError, match="additionalProperties"):
        AgentToolSpec("invalid", "无效工具", {"type": "object"}, lambda _: None)

    tool = make_tool(
        "demo",
        "演示工具",
        _valid_schema({"value": {"type": "string"}}),
        lambda params: params["value"],
    )

    exported = tool.to_anthropic_tool()
    assert exported["name"] == "demo"
    assert exported["description"] == "演示工具"
    assert exported["input_schema"]["type"] == "object"
    assert exported["input_schema"]["additionalProperties"] is False


def test_general_tools_factory_exposes_both_read_only_specs() -> None:
    """通用工具工厂应暴露两个预定义的只读工具。"""

    tools = general_tools()
    assert set(tools) == {"inspect_request_context", "suggest_required_fields"}
    assert all(isinstance(spec, AgentToolSpec) for spec in tools.values())
    assert all(
        spec.to_anthropic_tool()["input_schema"]["additionalProperties"] is False
        for spec in tools.values()
    )


def test_inspect_request_context_returns_sanitized_snapshot() -> None:
    """请求快照只返回脱敏用户标识与诊断字段，不泄漏原始 user_id。"""

    raw_user_id = "customer-abcdef"
    result = _run(
        inspect_request_context(
            {},
            context={
                "request_id": "req-123",
                "intent": "refund",
                "user_id": raw_user_id,
                "timestamp": "2026-10-04T10:00:00+00:00",
            },
        )
    )

    assert result["success"] is True
    assert result["request_id"] == "req-123"
    assert result["intent"] == "refund"
    assert result["user_id"] != raw_user_id
    assert "abcdef" not in result["user_id"]
    assert result["timestamp"] == "2026-10-04T10:00:00+00:00"


def test_suggest_required_fields_uses_current_beauty_intent_registry() -> None:
    """字段追问必须与当前 19 个美妆意图注册表保持一致。"""

    refund = _run(
        suggest_required_fields({"intent": "return_refund"}, context={"entities": {}})
    )
    missing_item = _run(
        suggest_required_fields({"intent": "missing_item"}, context={"entities": {}})
    )

    assert refund["success"] is True
    assert refund["required_fields"] == ["order_id", "return_reason", "opened_or_used"]
    assert "订单" in refund["suggestion"]
    assert missing_item["success"] is True
    assert missing_item["required_fields"] == ["order_id", "missing_item", "quantity"]


def test_suggest_required_fields_reads_known_context_and_returns_only_missing_fields() -> None:
    """字段建议必须结合会话事实，不能把意图的全量必填字段再次问给消费者。"""

    result = _run(
        suggest_required_fields(
            {"intent": "return_refund"},
            context={
                "entities": {
                    "order_id": ["SC-DEMO-4821"],
                    "opened_or_used": ["未拆封"],
                }
            },
        )
    )

    assert result["success"] is True
    assert result["required_fields"] == ["return_reason"]
    assert result["missing_fields"] == ["return_reason"]
    assert result["known_fields"] == {
        "order_id": ["SC-DEMO-4821"],
        "opened_or_used": ["未拆封"],
    }
    assert "opened_or_used" not in result["suggestion"]


def test_suggest_required_fields_treats_explicitly_cleared_fact_as_missing() -> None:
    """客服清空的历史字段不能继续被当成已知值。"""

    result = _run(
        suggest_required_fields(
            {"intent": "return_refund"},
            context={
                "case_facts": {
                    "facts": {
                        "opened_or_used": [
                            {"value": "未拆封", "confirmation_status": "unconfirmed"},
                            {"value": None, "confirmation_status": "cleared"},
                        ]
                    }
                }
            },
        )
    )

    assert result["success"] is True
    assert "opened_or_used" in result["missing_fields"]
    assert "opened_or_used" not in result["known_fields"]


def test_suggest_required_fields_uses_latest_fact_when_current_message_updates_it() -> None:
    """当前轮的新回答应覆盖历史槽位，不能继续使用旧状态判断缺失项。"""

    result = _run(
        suggest_required_fields(
            {"intent": "return_refund"},
            context={
                "entities": {"opened_or_used": ["已拆封未使用"]},
                "case_facts": {
                    "facts": {
                        "opened_or_used": [
                            {"value": "未拆封", "confirmation_status": "unconfirmed"}
                        ]
                    }
                },
            },
        )
    )

    assert result["known_fields"]["opened_or_used"] == ["已拆封未使用"]
    assert "opened_or_used" not in result["missing_fields"]


def test_suggest_required_fields_uses_canonical_context_intent_not_model_paraphrase() -> None:
    result = _run(
        suggest_required_fields(
            {"intent": "退款申请并核对商品状态"},
            context={
                "intent": "order_status",
                "intent_ids": ["return_refund"],
                "entities": {
                    "order_id": ["SC-DEMO-4821"],
                    "opened_or_used": ["未拆封"],
                },
            },
        )
    )

    assert result["success"] is True
    assert result["intent"] == "return_refund"
    assert result["missing_fields"] == ["return_reason"]


def test_suggest_required_fields_does_not_fallback_to_model_intent_when_context_is_authoritative() -> None:
    result = _run(
        suggest_required_fields(
            {"intent": "return_refund"},
            context={"intent": "other", "intent_ids": [], "entities": {}},
        )
    )

    assert result["success"] is False
    assert "无法映射" in result["error"]


def test_suggest_required_fields_supports_registered_intents_without_blocking_fields() -> None:
    """无固定阻塞字段的咨询意图也应读取上下文，而不是误报未注册。"""

    result = _run(
        suggest_required_fields(
            {"intent": "ingredient_fit"},
            context={
                "intent_ids": ["ingredient_fit"],
                "entities": {"product_name": ["澄润屏障修护精华水"]},
                "case_facts": {},
            },
        )
    )

    assert result["success"] is True
    assert result["missing_fields"] == []
    assert result["known_fields"] == {"product_name": ["澄润屏障修护精华水"]}
    assert "没有固定" in result["suggestion"]


def test_general_tools_invalid_inputs_return_safe_error_without_raising() -> None:
    """错误入参应转为结构化错误，不得让工具异常冒泡。"""

    snapshot_error = _run(inspect_request_context([], context={}))
    missing_intent = _run(suggest_required_fields({}))
    wrong_intent = _run(suggest_required_fields({"intent": 123}))

    assert snapshot_error["success"] is False
    assert snapshot_error["error"]
    assert missing_intent["success"] is False
    assert missing_intent["error"]
    assert wrong_intent["success"] is False
    assert wrong_intent["error"]
    no_context = _run(suggest_required_fields({"intent": "return_refund"}))
    assert no_context["success"] is False
    assert "上下文不可用" in no_context["error"]


def test_beauty_tools_resolve_order_dependencies_from_shared_context() -> None:
    context = {
        "user_id": "consumer-demo",
        "entities": {"order_id": ["SC-DEMO-4821"]},
    }
    order = _run(lookup_order({}, context=context))
    logistics = _run(lookup_logistics({"order_id": "SC-DEMO-4821"}, context=context))
    refund = _run(check_refund_status({}, context=context))
    ticket = _run(
        create_ticket_draft(
            {"intent": "adverse_reaction"},
            context={"entities": {"order_id": ["SC-DEMO-4821"], "symptom": ["发红"]}},
        )
    )

    assert order["success"] is True and order["available"] is True
    assert order["order"]["product_name"] == "轻透粉底液"
    assert logistics["success"] is True and logistics["available"] is True
    assert logistics["tracking_no"] == order["order"]["tracking_no"]
    assert logistics["logistics_status"] == order["order"]["logistics_status"]
    assert refund["refund_status"] == order["order"]["refund_status"] == "未申请"
    assert ticket["fields"]["order_id"] == "SC-DEMO-4821"
    assert ticket["fields"]["symptom"] == "发红"

