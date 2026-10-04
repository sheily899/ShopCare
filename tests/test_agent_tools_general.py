"""Step 5.1 通用 Agent 工具契约与只读行为测试。"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

import pytest

from agents.tools import (
    AgentToolSpec,
    general_tools,
    inspect_request_context,
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


def test_suggest_required_fields_maps_refund_and_technical_intents() -> None:
    """退款与技术意图应给出可执行的下一轮补字段建议。"""

    refund = _run(suggest_required_fields({"intent": "refund"}))
    technical = _run(suggest_required_fields({"intent": "technical_crash"}))

    assert refund["success"] is True
    assert refund["required_fields"] == ["order_id", "amount"]
    assert "订单" in refund["suggestion"]
    assert technical["success"] is True
    assert "error_code" in technical["required_fields"]


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

