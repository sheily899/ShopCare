from __future__ import annotations

from dataclasses import is_dataclass
from enum import IntEnum

from pydantic import BaseModel

from agents.agent_orchestrator import (
    AgentProfile,
    AgentResponse,
    AgentType,
    OrchestratorResult,
    Request,
    RoutingDecision,
)
from intent.beauty_intent_registry import BEAUTY_INTENT_BY_ID
from intent.intent_recognizer import IntentCategory, IntentResult, UrgencyLevel


EXPECTED_INTENTS = {
    *BEAUTY_INTENT_BY_ID,
    "human_handoff",
    "other",
}

EXPECTED_AGENTS = {
    "consult",
    "order",
    "after_sales",
    "escalation",
}


def _intent_result() -> IntentResult:
    return IntentResult(
        intent=IntentCategory.PRODUCT_INFO,
        confidence=0.8,
        urgency=UrgencyLevel.LOW,
        intent_group="consult",
        entities={
            "order_id": [],
            "product": [],
            "date": [],
            "amount": [],
            "error_code": [],
        },
        reasoning="contract test",
        latency_ms=1.0,
    )


def _agent_profile() -> AgentProfile:
    return AgentProfile(
        role="consult",
        mission="answer safely",
        workflow=("inspect",),
        input_contract=("message",),
        output_contract=("response",),
    )


def _agent_response() -> AgentResponse:
    return AgentResponse(
        agent_type=AgentType.INQUIRY,
        content="ok",
        success=True,
    )


def _request() -> Request:
    return Request(message="hello")


def _orchestrator_result() -> OrchestratorResult:
    return OrchestratorResult(
        request_id="req-1",
        response="ok",
        agent_type=AgentType.INQUIRY,
        intent=IntentCategory.PRODUCT_INFO,
    )


def test_intent_category_keeps_registered_beauty_and_control_values() -> None:
    assert EXPECTED_INTENTS <= {item.value for item in IntentCategory}
    assert len(BEAUTY_INTENT_BY_ID) == 19
    assert IntentCategory.ESCALATION is IntentCategory.HUMAN_HANDOFF
    assert IntentCategory.ESCALATION.value == "human_handoff"


def test_urgency_level_is_int_enum_with_values_one_to_four() -> None:
    assert issubclass(UrgencyLevel, IntEnum)
    assert [level.value for level in UrgencyLevel] == [1, 2, 3, 4]


def test_intent_and_agent_enums_are_string_enums() -> None:
    assert issubclass(IntentCategory, str)
    assert issubclass(AgentType, str)
    assert {item.value for item in AgentType} == EXPECTED_AGENTS


def test_dataclass_contracts_instantiate_with_spec_defaults() -> None:
    intent_result = _intent_result()
    profile = _agent_profile()
    agent_response = _agent_response()
    request = _request()
    routing = RoutingDecision(primary_agent=AgentType.GENERAL)
    result = _orchestrator_result()

    assert intent_result.source_scores == {}
    assert intent_result.source_intents == {}
    assert intent_result.secondary_intents == []
    assert intent_result.compound_trigger == ""
    assert profile.handoff_conditions == ()
    assert profile.tool_scope == ()
    assert profile.model is None
    assert profile.temperature == 0.2
    assert profile.max_tokens == 1024
    assert agent_response.confidence == 1.0
    assert agent_response.latency_ms == 0.0
    assert agent_response.escalate is False
    assert agent_response.tools_used == []
    assert agent_response.tool_traces == []
    assert request.user_id == "anonymous"
    assert request.conv_id is None
    assert request.context == ""
    assert request.history is None
    assert request.intent is None
    assert request.intent_group is None
    assert request.urgency is None
    assert request.intent_confidence == 1.0
    assert request.intent_diagnostics == {}
    assert request.secondary_intents == []
    assert request.knowledge_required is False
    assert routing.supporting_agents == []
    assert routing.reason == ""
    assert routing.confidence == 0.0
    assert result.escalated is False
    assert result.agent_recommended_handoff is False
    assert result.latency_ms == 0.0
    assert result.agent_types == []
    assert result.primary_agent is None
    assert result.supporting_agents == []
    assert result.tools_used == []
    assert result.tool_traces == []
    assert result.routing_reason == ""
    assert result.routing_confidence == 0.0
    assert result.intent_diagnostics == {}
    assert result.secondary_intents == []
    assert result.knowledge_required is False


def test_mutable_dataclass_defaults_are_independent() -> None:
    first_intent = _intent_result()
    second_intent = _intent_result()
    first_agent = _agent_response()
    second_agent = _agent_response()
    first_request = _request()
    second_request = _request()
    first_result = _orchestrator_result()
    second_result = _orchestrator_result()

    first_intent.source_scores["pattern"] = 0.5
    first_agent.tools_used.append("tool")
    first_request.entities["order_id"] = ["order-1"]
    first_result.tools_used.append("tool")

    assert second_intent.source_scores == {}
    assert second_agent.tools_used == []
    assert second_request.entities == {}
    assert second_result.tools_used == []


def test_request_generates_an_eight_character_request_id() -> None:
    request = _request()

    assert len(request.request_id) == 8
    assert request.request_id.isalnum()


def test_all_internal_models_are_dataclasses_and_not_pydantic_models() -> None:
    instances = [
        _intent_result(),
        _agent_profile(),
        _agent_response(),
        _request(),
        RoutingDecision(primary_agent=AgentType.GENERAL),
        _orchestrator_result(),
    ]

    assert all(is_dataclass(instance) for instance in instances)
    assert all(not isinstance(instance, BaseModel) for instance in instances)
