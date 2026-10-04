"""Step 5.3 Escalation 交接摘要与安全边界测试。"""

from __future__ import annotations

import asyncio
import json
import re
import socket
from typing import Any

from agents.tools import (
    AgentToolSpec,
    create_handoff_summary,
    escalation_tools,
)


def _run(coro: Any) -> Any:
    """使用标准库运行异步工具，避免引入 pytest-asyncio。"""

    return asyncio.run(coro)


def test_create_handoff_summary_returns_structured_safe_snapshot() -> None:
    """正常升级请求应生成 request_id、原因、紧急度和实体摘要。"""

    result = _run(
        create_handoff_summary(
            {
                "reason": "用户对账单处理时效有争议",
                "urgency": 3,
                "entities": {"order_id": ["ORD-100"], "amount": ["99.00"]},
                "context_summary": "用户已提供订单号，等待进一步核验。",
            }
        )
    )

    assert result["success"] is True
    assert re.fullmatch(r"[0-9a-f]{8}", result["request_id"])
    assert result["reason"] == "用户对账单处理时效有争议"
    assert result["urgency"] == 3
    assert result["entities"]["order_id"] == ["ORD-100"]
    assert result["sensitive_data_required"] is False
    assert result["external_ticket_created"] is False
    assert "已接手" not in json.dumps(result, ensure_ascii=False)
    assert "已接入" not in json.dumps(result, ensure_ascii=False)


def test_handoff_summary_filters_sensitive_fields_and_prompts() -> None:
    """交接摘要不得输出或引导收集密码、验证码和完整卡号。"""

    result = _run(
        create_handoff_summary(
            {
                "reason": "需要升级处理",
                "urgency": 4,
                "entities": {
                    "order_id": ["ORD-1"],
                    "password": "secret",
                    "verification_code": "123456",
                    "full_card_number": "4111111111111111",
                },
                "context_summary": "用户未提供任何需要收集的敏感凭证。",
            }
        )
    )
    serialized = json.dumps(result, ensure_ascii=False).lower()

    assert result["success"] is True
    assert result["sensitive_data_required"] is False
    assert "password" not in serialized
    assert "verification_code" not in serialized
    assert "full_card_number" not in serialized
    assert "密码" not in serialized
    assert "验证码" not in serialized
    assert "银行卡号" not in serialized
    assert "order_id" in result["entities"]


def test_handoff_summary_rejects_invalid_input_without_external_side_effect(
    monkeypatch: Any,
) -> None:
    """非法入参安全失败，且执行期间不建立网络连接。"""

    def fail_socket(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("交接摘要不应建立网络连接")

    # 仅拦截显式外联入口，避免替换 socket.socket 破坏 pytest 的事件循环创建。
    monkeypatch.setattr(socket, "create_connection", fail_socket)
    invalid_reason = _run(create_handoff_summary({"reason": "", "urgency": 2}))
    invalid_urgency = _run(
        create_handoff_summary({"reason": "需要升级", "urgency": 5})
    )
    invalid_entities = _run(
        create_handoff_summary(
            {"reason": "需要升级", "urgency": 2, "entities": "not-a-dict"}
        )
    )

    assert invalid_reason["success"] is False
    assert invalid_reason["sensitive_data_required"] is False
    assert invalid_urgency["success"] is False
    assert invalid_entities["success"] is False


def test_escalation_tool_registry_uses_strict_anthropic_schema() -> None:
    """Escalation 工具应通过 AgentToolSpec 暴露且关闭额外参数。"""

    tools = escalation_tools()
    assert set(tools) == {"create_handoff_summary"}
    spec = tools["create_handoff_summary"]
    assert isinstance(spec, AgentToolSpec)
    exported = spec.to_anthropic_tool()
    assert exported["input_schema"]["type"] == "object"
    assert exported["input_schema"]["additionalProperties"] is False
    assert exported["input_schema"]["required"] == ["reason", "urgency"]
