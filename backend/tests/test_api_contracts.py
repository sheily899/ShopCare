from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from api.main import ChatRequest, ChatResponse


def _required_response_fields() -> dict[str, object]:
    return {
        "conv_id": "conv-1",
        "response": "下一步请提供错误码。",
        "intent": "product_info",
        "agent_type": "consult",
        "escalated": False,
        "latency_ms": 12.5,
    }


def test_chat_request_parses_required_message_and_defaults() -> None:
    request = ChatRequest(message="登录失败")

    assert request.message == "登录失败"
    assert request.user_id == "anonymous"
    assert request.conv_id is None


@pytest.mark.parametrize("message", ["", "   ", "\t\n"])
def test_chat_request_rejects_empty_or_whitespace_message(message: str) -> None:
    with pytest.raises(ValidationError):
        ChatRequest(message=message)


def test_chat_response_initializes_required_and_default_fields() -> None:
    response = ChatResponse(**_required_response_fields())

    assert response.conv_id == "conv-1"
    assert response.request_id == ""
    assert response.secondary_intents == []
    assert response.intent_group == "other"
    assert response.agent_types == []
    assert response.primary_agent == ""
    assert response.supporting_agents == []
    assert response.tools_used == []
    assert response.routing_reason == ""
    assert response.routing_confidence == 0.0
    assert response.agent_recommended_handoff is False
    assert response.knowledge_used is False
    assert response.entities == {}
    assert response.intent_confidence == 0.0
    assert response.intent_source_scores == {}
    assert response.intent_diagnostics == {}


def test_chat_response_default_collections_are_not_shared() -> None:
    first = ChatResponse(**_required_response_fields())
    second = ChatResponse(**_required_response_fields())

    first.tools_used.append("search_knowledge_base")
    first.entities["error_code"] = ["500"]

    assert second.tools_used == []
    assert second.entities == {}


@pytest.mark.parametrize("field", ["routing_confidence", "intent_confidence"])
@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_chat_response_rejects_confidence_outside_zero_to_one(
    field: str, value: float
) -> None:
    payload = _required_response_fields()
    payload[field] = value

    with pytest.raises(ValidationError):
        ChatResponse(**payload)


def test_chat_response_rejects_negative_latency() -> None:
    payload = _required_response_fields()
    payload["latency_ms"] = -0.001

    with pytest.raises(ValidationError):
        ChatResponse(**payload)


def test_chat_response_serializes_to_dict_and_json() -> None:
    response = ChatResponse(**_required_response_fields())

    as_dict = response.model_dump()
    as_json = json.loads(response.model_dump_json())

    assert as_dict["conv_id"] == "conv-1"
    assert as_dict["latency_ms"] == 12.5
    assert as_json == as_dict
