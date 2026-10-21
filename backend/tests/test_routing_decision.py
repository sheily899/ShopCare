"""Step 6.1 路由评分、澄清分支与执行降级测试。"""

from __future__ import annotations

import asyncio
from typing import Any

import agents.agent_orchestrator as orchestrator_module
from agents.agent_orchestrator import (
    AgentResponse,
    AgentType,
    Request,
    _domain_scores,
    _execute,
    _route_decision,
)
from intent.intent_recognizer import IntentCategory, IntentResult, UrgencyLevel


def _run(coroutine: Any) -> Any:
    """使用标准库事件循环运行异步执行路径。"""

    return asyncio.run(coroutine)


def _intent_result(
    intent: IntentCategory,
    confidence: float = 0.9,
    urgency: UrgencyLevel = UrgencyLevel.LOW,
    entities: dict[str, list[str]] | None = None,
    secondary: list[IntentCategory] | None = None,
) -> IntentResult:
    """构造最小可用的意图结果。"""

    return IntentResult(
        intent=intent,
        confidence=confidence,
        urgency=urgency,
        intent_group=intent.value,
        entities=entities or {},
        reasoning="test",
        latency_ms=0.1,
        secondary_intents=secondary or [],
    )


def test_critical_and_human_handoff_route_to_escalation() -> None:
    """CRITICAL 与 HUMAN_HANDOFF 必须绕过领域评分直接升级。"""

    critical_request = Request(
        message="服务完全中断",
        intent=IntentCategory.TECHNICAL,
        urgency=UrgencyLevel.CRITICAL,
        intent_confidence=0.6,
    )
    critical = _route_decision(
        critical_request,
        _intent_result(IntentCategory.TECHNICAL, confidence=0.6, urgency=UrgencyLevel.CRITICAL),
    )
    assert critical.primary_agent is AgentType.ESCALATION
    assert critical.confidence == 1.0

    handoff_request = Request(
        message="请转人工",
        intent=IntentCategory.HUMAN_HANDOFF,
        urgency=UrgencyLevel.HIGH,
        intent_confidence=0.62,
    )
    handoff = _route_decision(
        handoff_request,
        _intent_result(IntentCategory.HUMAN_HANDOFF, confidence=0.62, urgency=UrgencyLevel.HIGH),
    )
    assert handoff.primary_agent is AgentType.ESCALATION
    assert handoff.confidence == 0.8


def test_low_confidence_other_returns_clarification_without_agent() -> None:
    """长文本的低置信度 OTHER 只返回固定追问，不产生 Agent 目标。"""

    request = Request(
        message="我不知道该怎么处理这个问题",
        intent=IntentCategory.OTHER,
        intent_confidence=0.3,
        urgency=UrgencyLevel.LOW,
    )
    decision = _route_decision(request, _intent_result(IntentCategory.OTHER, confidence=0.3))

    assert decision.primary_agent is None
    assert decision.supporting_agents == []
    assert decision.confidence == 0.3
    assert "产品咨询、订单物流，还是售后退款与使用不适" in decision.reason


def test_domain_scores_use_order_intent_and_order_entities() -> None:
    """订单意图、订单号和物流词应使履约域分数最高。"""

    request = Request(
        message="订单 SC-DEMO-4821 的物流到哪里了",
        intent=IntentCategory.LOGISTICS_STALLED,
        entities={"order_id": ["SC-DEMO-4821"]},
    )
    result = _intent_result(
        IntentCategory.LOGISTICS_STALLED,
        entities={"order_id": ["SC-DEMO-4821"]},
    )
    scores = _domain_scores(request, result)

    assert scores["order"] > scores["consult"]
    assert scores["order"] > scores["after_sales"]
    decision = _route_decision(request, result)
    assert decision.primary_agent is AgentType.FULFILLMENT


def test_supporting_agents_only_map_explicit_cross_domain_secondary_intents() -> None:
    """辅助 Agent 只来自 secondary_intents，并过滤同域与人工项。"""

    request = Request(
        message="订单物流慢，同时我想确认价保",
        intent=IntentCategory.LOGISTICS_STALLED,
        intent_confidence=0.9,
        entities={"order_id": ["SC-DEMO-4821"]},
        secondary_intents=[
            IntentCategory.MISSING_ITEM,
            IntentCategory.PRICE_PROTECTION,
            IntentCategory.HUMAN_HANDOFF,
            IntentCategory.PRICE_PROTECTION,
        ],
    )
    result = _intent_result(
        IntentCategory.LOGISTICS_STALLED,
        entities=request.entities,
        secondary=request.secondary_intents,
    )

    decision = _route_decision(request, result)

    assert decision.primary_agent is AgentType.FULFILLMENT
    assert decision.supporting_agents == [AgentType.REFUND_RISK]


class _FakeAgent:
    """用于验证 `_execute` 降级链条的最小 Agent 替身。"""

    def __init__(self, response: AgentResponse | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls = 0

    async def handle(self, request: Request) -> AgentResponse:
        """返回预置结果或抛出预置异常。"""

        self.calls += 1
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


def test_execute_business_domain_failure_does_not_fall_back_to_consult(monkeypatch: Any) -> None:
    """订单域失败不能转给咨询 Agent 猜答或暴露原始异常。"""

    order_agent = _FakeAgent(error=RuntimeError("order lookup failed"))
    consult_agent = _FakeAgent(
        response=AgentResponse(
            agent_type=AgentType.INQUIRY,
            content="没有订单依据的猜测",
            success=True,
        )
    )
    monkeypatch.setattr(
        orchestrator_module,
        "AGENT_POOL",
        {
            AgentType.FULFILLMENT: order_agent,
            AgentType.INQUIRY: consult_agent,
        },
    )

    result = _run(_execute(Request(message="订单物流查询"), AgentType.FULFILLMENT))

    assert result.success is False
    assert result.agent_type is AgentType.FULFILLMENT
    assert "order lookup failed" not in result.content
    assert order_agent.calls == 1
    assert consult_agent.calls == 0
