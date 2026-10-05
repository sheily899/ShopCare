"""Step 6.3 ResponseComposer 与编排结果装配测试。"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

import agents.agent_orchestrator as orchestrator_module
from agents.agent_orchestrator import (
    AgentResponse,
    AgentType,
    AgentOrchestrator,
    Request,
    ResponseComposer,
    RoutingDecision,
)
from intent.intent_recognizer import IntentCategory


class _FakeComposerClient:
    """记录合成请求并返回预置文本的异步客户端替身。"""

    def __init__(self, text: str = "合并后的客户回复", error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.text = text
        self.error = error
        self.messages = self

    async def create(self, **kwargs: Any) -> Any:
        """记录 Anthropic 参数并返回文本响应。"""

        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.text)])


def _response(agent_type: AgentType, content: str, *, success: bool = True) -> AgentResponse:
    """构造最小 AgentResponse 测试数据。"""

    return AgentResponse(
        agent_type=agent_type,
        content=content,
        success=success,
        tools_used=[f"{agent_type.value}_tool"],
        tool_traces=[{"tool_name": f"{agent_type.value}_tool", "status": "success"}],
        latency_ms=4.0,
    )


def _run(coroutine: Any) -> Any:
    """用标准库事件循环执行异步测试。"""

    return asyncio.run(coroutine)


def test_composer_single_success_returns_directly_without_llm_call() -> None:
    """单结果应直出，避免无意义的额外 Token 消耗。"""

    client = _FakeComposerClient()
    composer = ResponseComposer(client=client)

    result = _run(
        composer.compose(
            Request(message="查询订单"),
            [_response(AgentType.GENERAL, "直接回答")],
            primary_agent=AgentType.GENERAL,
        )
    )

    assert result == "直接回答"
    assert client.calls == []


def test_composer_parallel_success_uses_locked_parameters_and_safe_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """多结果应调用合成模型，并传入温度、Token 和防脑补约束。"""

    monkeypatch.setenv("CUSTOMEROPS_COMPOSER_TEMPERATURE", "0.1")
    monkeypatch.setenv("CUSTOMEROPS_COMPOSER_MAX_TOKENS", "1000")
    client = _FakeComposerClient(text="统一回复")
    composer = ResponseComposer(client=client)

    result = _run(
        composer.compose(
            Request(message="电脑故障并咨询退款"),
            [
                _response(AgentType.TECHNICAL, "技术结论"),
                _response(AgentType.BILLING, "账单补充"),
            ],
            primary_agent=AgentType.TECHNICAL,
        )
    )

    assert result == "统一回复"
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["temperature"] == 0.1
    assert call["max_tokens"] == 1000
    assert call["extra_body"] == {"thinking": {"type": "disabled"}}
    prompt = call["messages"][0]["content"]
    assert "技术结论" in prompt
    assert "账单补充" in prompt
    assert "严禁无中生有编造订单发货、退款成功或后台查询结果" in prompt
    assert "不得暴露内部 Agent 专有名词" in prompt


def test_composer_failure_uses_exact_deterministic_fallback() -> None:
    """合成失败时必须保留主结论并按固定格式拼接辅助说明。"""

    client = _FakeComposerClient(error=TimeoutError("composer timeout"))
    composer = ResponseComposer(client=client)

    result = _run(
        composer.compose(
            Request(message="复合问题"),
            [
                _response(AgentType.TECHNICAL, "主结果"),
                _response(AgentType.BILLING, "辅助结果"),
            ],
            primary_agent=AgentType.TECHNICAL,
        )
    )

    assert result == "主结果\n\n补充说明：\n辅助结果"


def test_parallel_orchestrator_assembles_composed_text_and_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """编排结果应包含合成文本、Agent 类型、工具和路由元数据。"""

    responses = [
        _response(AgentType.TECHNICAL, "主结果"),
        _response(AgentType.BILLING, "辅助结果"),
    ]

    async def fake_fan_out(*args: Any, **kwargs: Any) -> list[AgentResponse]:
        """替代真实 Fan-out，专注验证结果装配。"""

        return responses

    monkeypatch.setattr(orchestrator_module, "_fan_out", fake_fan_out)
    client = _FakeComposerClient(text="合并结果")
    orchestrator = AgentOrchestrator(
        agent_pool={},
        composer=ResponseComposer(client=client),
    )
    request = Request(
        message="技术和账单问题",
        request_id="req12345",
        intent=IntentCategory.TECHNICAL,
        secondary_intents=[IntentCategory.BILLING],
        intent_diagnostics={"source": "test"},
    )
    decision = RoutingDecision(
        primary_agent=AgentType.TECHNICAL,
        supporting_agents=[AgentType.BILLING],
        reason="test route",
        confidence=0.91,
    )

    result = _run(orchestrator.run_parallel(request, decision))

    assert result.response == "合并结果"
    assert result.agent_type is AgentType.TECHNICAL
    assert result.agent_types == [AgentType.TECHNICAL, AgentType.BILLING]
    assert result.primary_agent is AgentType.TECHNICAL
    assert result.supporting_agents == [AgentType.BILLING]
    assert result.routing_reason == "test route"
    assert result.routing_confidence == 0.91
    assert result.tools_used == ["technical_tool", "billing_tool"]
    assert len(result.tool_traces) == 2
    assert result.latency_ms >= 0.0


def test_parallel_orchestrator_all_failures_returns_safe_fixed_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """全部 Agent 失败时返回固定保守文案且不调用合成模型。"""

    async def fake_fan_out(*args: Any, **kwargs: Any) -> list[AgentResponse]:
        """模拟 Fan-out 后无成功结果。"""

        return []

    monkeypatch.setattr(orchestrator_module, "_fan_out", fake_fan_out)
    client = _FakeComposerClient()
    orchestrator = AgentOrchestrator(agent_pool={}, composer=ResponseComposer(client=client))
    request = Request(message="不可处理的问题", intent=IntentCategory.OTHER)
    decision = RoutingDecision(primary_agent=AgentType.GENERAL, reason="route")

    result = _run(orchestrator.run_parallel(request, decision))

    assert result.response == "抱歉，所有 Agent 均处理失败。"
    assert result.agent_types == []
    assert result.tools_used == []
    assert client.calls == []
