"""Step 5.4 四大领域 Agent、Profile 与工具白名单测试。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from agents.agent_orchestrator import (
    AgentProfile,
    AgentResponse,
    AgentStats,
    AgentType,
    BaseAgent,
    BillingAgent,
    EscalationAgent,
    GeneralAgent,
    Request,
    TechnicalAgent,
)
from intent.intent_recognizer import IntentCategory, UrgencyLevel


def _run(coroutine: Any) -> Any:
    """使用标准库事件循环运行异步 Agent。"""

    return asyncio.run(coroutine)


def test_four_agent_profiles_lock_temperature_and_token_budget() -> None:
    """四类 Agent 的 Profile 超参必须与架构契约一致。"""

    agents = [
        (GeneralAgent(), AgentType.GENERAL, 0.3, 900),
        (TechnicalAgent(), AgentType.TECHNICAL, 0.1, 1200),
        (BillingAgent(), AgentType.BILLING, 0.0, 1100),
        (EscalationAgent(), AgentType.ESCALATION, 0.0, 500),
    ]

    for agent, expected_type, temperature, max_tokens in agents:
        assert agent.agent_type is expected_type
        assert isinstance(agent.profile, AgentProfile)
        assert agent.profile.temperature == temperature
        assert agent.profile.max_tokens == max_tokens
        assert isinstance(agent.stats, AgentStats)


def test_domain_agents_only_expose_their_whitelisted_tools() -> None:
    """Technical 与 Billing 不能互相看到对方的领域工具。"""

    technical_names = {tool.name for tool in TechnicalAgent().tools}
    billing_names = {tool.name for tool in BillingAgent().tools}

    assert "search_knowledge_base" in technical_names
    assert "lookup_error_code" in technical_names
    assert "build_diagnostic_plan" in technical_names
    assert "check_billing_fields" not in technical_names
    assert "compare_amounts" not in technical_names

    assert "search_knowledge_base" in billing_names
    assert "check_billing_fields" in billing_names
    assert "compare_amounts" in billing_names
    assert "lookup_error_code" not in billing_names
    assert "build_diagnostic_plan" not in billing_names


class _FakeClient:
    """记录消息调用次数的 Anthropic 客户端替身。"""

    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.response = response
        self.error = error
        self.messages = self

    async def create(self, **kwargs: Any) -> Any:
        """记录请求并返回预置响应。"""

        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


def test_escalation_agent_is_deterministic_and_skips_llm() -> None:
    """EscalationAgent 只生成升级建议，不消耗 LLM 或设置真实接入状态。"""

    client = _FakeClient(
        response=SimpleNamespace(content=[SimpleNamespace(type="text", text="不应调用")])
    )
    agent = EscalationAgent(client=client)
    request = Request(
        message="请马上转人工",
        request_id="abc12345",
        intent=IntentCategory.HUMAN_HANDOFF,
        urgency=UrgencyLevel.HIGH,
    )

    result = _run(agent.handle(request))

    assert result.success is True
    assert result.agent_type is AgentType.ESCALATION
    assert "已生成交接摘要与升级建议" in result.content
    assert "已接入" not in result.content
    assert client.calls == []
    assert agent.stats.total == 1
    assert agent.stats.success == 1


def test_base_agent_catches_llm_exception_and_records_latency() -> None:
    """LLM 异常不得冒泡，失败结果仍需有耗时与统计。"""

    agent = GeneralAgent(client=_FakeClient(error=RuntimeError("mock failure")))
    result = _run(agent.handle(Request(message="怎么查询订单")))

    assert isinstance(result, AgentResponse)
    assert result.success is False
    assert "mock failure" in result.content
    assert result.latency_ms >= 0.0
    assert agent.stats.total == 1
    assert agent.stats.success == 0
    assert agent.stats.total_ms >= result.latency_ms


def test_base_agent_limits_tool_use_rounds_and_sends_only_whitelist() -> None:
    """连续 Tool Use 最多执行三轮，模型每次只收到当前 Agent 的工具声明。"""

    response = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="tool_use",
                id="toolu-1",
                name="inspect_request_context",
                input={},
            )
        ]
    )
    client = _FakeClient(response=response)
    agent = GeneralAgent(client=client)

    result = _run(agent.handle(Request(message="请检查当前请求")))

    assert result.success is True
    assert len(client.calls) == 3
    for call in client.calls:
        assert {tool["name"] for tool in call["tools"]} == {
            "search_knowledge_base",
            "inspect_request_context",
            "suggest_required_fields",
        }
