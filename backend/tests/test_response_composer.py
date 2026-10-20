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
from agents.case_facts import CaseFacts
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
            [_response(AgentType.INQUIRY, "直接回答")],
            primary_agent=AgentType.INQUIRY,
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
            Request(message="订单漏发小样，同时我要申请退款"),
            [
                _response(AgentType.FULFILLMENT, "仓储正在核对漏发的小样"),
                _response(AgentType.REFUND_RISK, "售后正在核实退款条件"),
            ],
            primary_agent=AgentType.FULFILLMENT,
        )
    )

    assert result == "统一回复"
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["temperature"] == 0.1
    assert call["max_tokens"] == 1000
    assert call["extra_body"] == {"thinking": {"type": "disabled"}}
    prompt = call["messages"][0]["content"]
    assert "仓储正在核对漏发的小样" in prompt
    assert "售后正在核实退款条件" in prompt
    assert "严禁无中生有编造订单发货、退款成功或后台查询结果" in prompt
    assert "内部 Agent 名称" in call["system"]


def test_composer_failure_uses_exact_deterministic_fallback() -> None:
    """合成失败时必须保留主结论并按固定格式拼接辅助说明。"""

    client = _FakeComposerClient(error=TimeoutError("composer timeout"))
    composer = ResponseComposer(client=client)

    result = _run(
        composer.compose(
            Request(message="复合问题"),
            [
                _response(AgentType.FULFILLMENT, "主结果"),
                _response(AgentType.REFUND_RISK, "辅助结果"),
            ],
            primary_agent=AgentType.FULFILLMENT,
        )
    )

    assert result == "主结果\n\n补充说明：\n辅助结果"


def test_parallel_orchestrator_assembles_composed_text_and_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """编排结果应包含合成文本、Agent 类型、工具和路由元数据。"""

    responses = [
        _response(AgentType.FULFILLMENT, "主结果"),
        _response(AgentType.REFUND_RISK, "辅助结果"),
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
        message="订单漏发小样，同时我要申请退款",
        request_id="req12345",
        intent=IntentCategory.MISSING_ITEM,
        secondary_intents=[IntentCategory.RETURN_REFUND],
        intent_diagnostics={"source": "test"},
    )
    decision = RoutingDecision(
        primary_agent=AgentType.FULFILLMENT,
        supporting_agents=[AgentType.REFUND_RISK],
        reason="test route",
        confidence=0.91,
    )

    result = _run(orchestrator.run_parallel(request, decision))

    assert result.response == "合并结果"
    assert result.agent_type is AgentType.FULFILLMENT
    assert result.agent_types == [AgentType.FULFILLMENT, AgentType.REFUND_RISK]
    assert result.primary_agent is AgentType.FULFILLMENT
    assert result.supporting_agents == [AgentType.REFUND_RISK]
    assert result.routing_reason == "test route"
    assert result.routing_confidence == 0.91
    assert result.tools_used == ["order_tool", "after_sales_tool"]
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


def test_failed_agent_branch_keeps_successful_tool_facts_for_case_state() -> None:
    """工具已成功、随后答复失败时，失败分支仍须把核实事实交给共享 CaseFacts。"""
    facts = CaseFacts()
    facts.record_tool_result(
        "lookup_order",
        {
            "success": True,
            "available": True,
            "order_id": "SC-DEMO-4821",
            "order": {"product_name": "轻透粉底液", "status": "已签收"},
        },
        source_ref="request-tool-success",
    )

    class AgentFailedAfterSuccessfulTool:
        async def handle(self, request: Request) -> AgentResponse:
            del request
            return AgentResponse(
                agent_type=AgentType.FULFILLMENT,
                content="答复模型调用失败",
                success=False,
                tools_used=["lookup_order"],
                tool_traces=[{"tool_name": "lookup_order", "status": "success"}],
                case_facts=facts.to_dict(),
            )

    orchestrator = AgentOrchestrator(
        agent_pool={AgentType.FULFILLMENT: AgentFailedAfterSuccessfulTool()},
        composer=ResponseComposer(),
    )
    result = _run(
        orchestrator.run_parallel(
            Request(
                message="订单 SC-DEMO-4821 到哪里了？",
                intent=IntentCategory.ORDER_STATUS_BEAUTY,
                entities={"order_id": ["SC-DEMO-4821"]},
                request_id="request-tool-success",
            ),
            RoutingDecision(primary_agent=AgentType.FULFILLMENT),
        )
    )

    result_facts = CaseFacts.from_dict(result.case_facts)
    assert result_facts.values("order_id") == ["SC-DEMO-4821"]
    assert result_facts.values("product_name") == ["轻透粉底液"]
    assert result_facts.values("order_status") == ["已签收"]
    assert result.tools_used == ["lookup_order"]
    assert any(trace.get("tool_name") == "lookup_order" for trace in result.tool_traces)


def test_all_failed_agents_still_use_composer_for_current_successful_tool_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """工具已有核验事实但领域答复失败时，应让最终答复模型据事实生成回复。"""
    facts = CaseFacts()
    facts.record_tool_result(
        "lookup_order",
        {
            "success": True,
            "available": True,
            "order_id": "SC-DEMO-4821",
            "order": {"product_name": "轻透粉底液", "status": "已签收"},
        },
        source_ref="request-tool-success",
    )

    failed = AgentResponse(
        agent_type=AgentType.FULFILLMENT,
        content="Agent 执行失败: 最终答复模型不可用",
        success=False,
        tools_used=["lookup_order"],
        tool_traces=[{"tool_name": "lookup_order", "status": "success", "success": True}],
        case_facts=facts.to_dict(),
    )

    async def failed_fan_out(*args: Any, **kwargs: Any) -> list[AgentResponse]:
        del args, kwargs
        return [failed]

    monkeypatch.setattr(orchestrator_module, "_fan_out", failed_fan_out)
    client = _FakeComposerClient(text="SC-DEMO-4821 的轻透粉底液已签收。")
    orchestrator = AgentOrchestrator(agent_pool={}, composer=ResponseComposer(client=client))
    request = Request(
        message="订单 SC-DEMO-4821 到了吗？",
        request_id="request-tool-success",
        intent=IntentCategory.ORDER_STATUS_BEAUTY,
    )

    result = _run(
        orchestrator.run_parallel(
            request,
            RoutingDecision(primary_agent=AgentType.FULFILLMENT),
        )
    )

    assert result.response == "SC-DEMO-4821 的轻透粉底液已签收。"
    assert len(client.calls) == 1
    prompt = client.calls[0]["messages"][0]["content"]
    assert "tool:lookup_order" in prompt
    assert "已签收" in prompt


def test_partial_agent_failure_composes_successful_tool_facts_with_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """一个领域答复成功、另一个领域只有工具结果时，合成仍须读取工具事实。"""
    facts = CaseFacts()
    facts.record_tool_result(
        "check_refund_status",
        {
            "success": True,
            "available": True,
            "order_id": "SC-DEMO-4821",
            "refund_status": "退款处理中",
        },
        source_ref="request-compound",
    )
    responses = [
        _response(AgentType.FULFILLMENT, "订单物流已签收。"),
        AgentResponse(
            agent_type=AgentType.REFUND_RISK,
            content="Agent 执行失败: 最终答复模型不可用",
            success=False,
            tools_used=["check_refund_status"],
            tool_traces=[{"tool_name": "check_refund_status", "status": "success", "success": True}],
            case_facts=facts.to_dict(),
        ),
    ]

    async def partial_fan_out(*args: Any, **kwargs: Any) -> list[AgentResponse]:
        del args, kwargs
        return responses

    monkeypatch.setattr(orchestrator_module, "_fan_out", partial_fan_out)
    client = _FakeComposerClient(text="物流已签收，退款仍在处理中。")
    orchestrator = AgentOrchestrator(agent_pool={}, composer=ResponseComposer(client=client))
    request = Request(
        message="订单物流和退款状态怎么样？",
        request_id="request-compound",
        intent=IntentCategory.ORDER_STATUS_BEAUTY,
        secondary_intents=[IntentCategory.REFUND_NOT_RECEIVED],
    )

    result = _run(
        orchestrator.run_parallel(
            request,
            RoutingDecision(
                primary_agent=AgentType.FULFILLMENT,
                supporting_agents=[AgentType.REFUND_RISK],
            ),
        )
    )

    assert result.response == "物流已签收，退款仍在处理中。"
    assert len(client.calls) == 1
    prompt = client.calls[0]["messages"][0]["content"]
    assert "退款处理中" in prompt
    assert "tool:check_refund_status" in prompt


def test_tool_fact_fallback_remains_useful_if_composer_is_unavailable_or_returns_placeholder() -> None:
    """最终模型不可用或只返回占位时，仍返回本轮已核实事实。"""
    facts = CaseFacts()
    facts.record_tool_result(
        "lookup_order",
        {
            "success": True,
            "available": True,
            "order_id": "SC-DEMO-4821",
            "order": {"product_name": "轻透粉底液", "status": "已签收"},
        },
        source_ref="request-tool-fallback",
    )
    request = Request(
        message="订单 SC-DEMO-4821 到了吗？",
        request_id="request-tool-fallback",
        case_facts=facts.to_dict(),
        requires_tool_fact_composition=True,
    )

    for composer in (
        ResponseComposer(),
        ResponseComposer(client=_FakeComposerClient(text="已完成必要的工具处理。")),
    ):
        result = _run(composer.compose(request, [], primary_agent=AgentType.FULFILLMENT))
        assert "订单号：SC-DEMO-4821" in result
        assert "订单状态：已签收" in result
        assert "所有 Agent 均处理失败" not in result
