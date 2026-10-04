"""Step 5.2 Technical/Billing 专属工具行为测试。"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agents.tools import (
    AgentToolSpec,
    billing_tools,
    build_diagnostic_plan,
    check_billing_fields,
    compare_amounts,
    lookup_error_code,
    technical_tools,
)


def _run(coro: Any) -> Any:
    """使用标准库运行异步工具，避免新增 pytest 插件依赖。"""

    return asyncio.run(coro)


@pytest.mark.parametrize(
    ("code", "meaning"),
    [
        ("401", "认证"),
        ("403", "权限"),
        ("404", "不存在"),
        ("500", "服务器"),
    ],
)
def test_lookup_error_code_uses_static_safe_catalog(
    code: str,
    meaning: str,
) -> None:
    """常见状态码应从静态目录解释，且明确未读取服务端日志。"""

    result = _run(lookup_error_code({"error_code": code}))

    assert result["success"] is True
    assert meaning in result["meaning"]
    assert result["server_log_checked"] is False
    assert result["source"] == "static_catalog"


def test_lookup_error_code_unknown_and_invalid_inputs_are_safe() -> None:
    """未知或错误格式不能伪造日志结果，也不能让异常冒泡。"""

    unknown = _run(lookup_error_code({"error_code": "599"}))
    invalid = _run(lookup_error_code({"error_code": 500}))

    assert unknown["success"] is True
    assert unknown["server_log_checked"] is False
    assert "未知" in unknown["meaning"]
    assert invalid["success"] is False
    assert invalid["server_log_checked"] is False
    assert invalid["error"]


def test_build_diagnostic_plan_returns_low_risk_structured_steps() -> None:
    """技术症状应生成可执行的低风险排查步骤，不执行系统重置。"""

    result = _run(
        build_diagnostic_plan(
            {"symptom": "登录后页面报错", "device_type": "Windows"}
        )
    )

    assert result["success"] is True
    assert result["device_type"] == "Windows"
    assert isinstance(result["steps"], list)
    assert len(result["steps"]) >= 3
    assert all(step["risk"] == "low" for step in result["steps"])
    assert result["high_risk_action"] is False


def test_check_billing_fields_never_confirms_refund() -> None:
    """账单字段检查只给出缺失项，不论字段是否齐全都不能确认退款。"""

    incomplete = _run(check_billing_fields({"order_id": "ORD-1"}))
    complete = _run(
        check_billing_fields(
            {
                "order_id": "ORD-1",
                "amount": "99.00",
                "date": "2026-10-04",
                "payment_channel": "card",
            }
        )
    )

    assert incomplete["success"] is True
    assert incomplete["can_confirm_refund"] is False
    assert "amount" in incomplete["missing_fields"]
    assert complete["success"] is True
    assert complete["complete"] is True
    assert complete["missing_fields"] == []
    assert complete["can_confirm_refund"] is False


def test_compare_amounts_is_pure_arithmetic_and_handles_invalid_values() -> None:
    """金额工具只计算差值与大小关系，不做重复扣款等法律定性。"""

    result = _run(
        compare_amounts({"expected_amount": 100.0, "actual_amount": 102.5})
    )
    invalid = _run(
        compare_amounts({"expected_amount": "one hundred", "actual_amount": 102.5})
    )

    assert result["success"] is True
    assert result["difference"] == 2.5
    assert result["comparison"] == "actual_higher"
    assert "duplicate" not in str(result).lower()
    assert invalid["success"] is False
    assert invalid["error"]


def test_domain_tool_registries_expose_anthropic_safe_specs() -> None:
    """Technical/Billing 注册表中的每个工具都应通过统一 schema 契约。"""

    tools = {**technical_tools(), **billing_tools()}

    assert set(tools) == {
        "lookup_error_code",
        "build_diagnostic_plan",
        "check_billing_fields",
        "compare_amounts",
    }
    assert all(isinstance(spec, AgentToolSpec) for spec in tools.values())
    assert all(
        spec.input_schema["type"] == "object"
        and spec.input_schema["additionalProperties"] is False
        for spec in tools.values()
    )

