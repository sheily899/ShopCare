from __future__ import annotations

import asyncio

import intent.intent_recognizer as recognizer
from intent.intent_recognizer import (
    EmbeddingIntentResult,
    IntentCategory,
    LLMIntentResult,
    build_scoped_retrieval_queries,
    detect_compound_intents,
    recognize_intent,
    split_compound_clauses,
)


def _run(coroutine):
    return asyncio.run(coroutine)


def _pattern(intent: IntentCategory, confidence: float = 0.0):
    return {
        "intent": intent,
        "confidence": confidence,
        "candidates": [intent],
        "scores": {intent.value: confidence},
        "hits": {},
        "specific_intent": IntentCategory.OTHER,
        "specific_score": 0.0,
    }


def test_single_beauty_request_does_not_enter_compound_branch() -> None:
    message = "这款精华适合敏感肌使用吗？"

    clauses = split_compound_clauses(message)
    secondary, trigger, diagnostics = detect_compound_intents(
        message, IntentCategory.INGREDIENT_FIT
    )

    assert len(clauses) == 1
    assert secondary == []
    assert trigger == ""
    assert diagnostics["compound_detected"] is False


def test_order_and_refund_request_split_into_scoped_subrequests() -> None:
    message = "订单里漏发了两个小样，同时我要申请退款"

    secondary, trigger, diagnostics = detect_compound_intents(
        message, IntentCategory.MISSING_ITEM
    )

    assert secondary == [IntentCategory.RETURN_REFUND]
    assert trigger == "同时"
    assert diagnostics["clause_intents"] == ["missing_item", "return_refund"]
    assert diagnostics["sub_requests"][0]["domain"] == "order"
    assert diagnostics["sub_requests"][1]["domain"] == "after_sales"
    assert build_scoped_retrieval_queries(
        message, secondary, diagnostics["clause_matches"]
    ) == {
        "order": "订单里漏发了两个小样",
        "after_sales": "我要申请退款",
    }


def test_same_domain_compound_intents_do_not_create_another_agent() -> None:
    message = "这款精华有哪些成分，而且敏感肌能不能用？"

    secondary, trigger, diagnostics = detect_compound_intents(
        message, IntentCategory.INGREDIENT_FIT
    )

    assert secondary == []
    assert trigger == "而且"
    assert diagnostics["clause_intents"] == ["ingredient_fit", "ingredient_fit"]
    assert len(diagnostics["sub_requests"]) == 2


def test_high_confidence_fast_path_still_emits_compound_intents() -> None:
    result = _run(
        recognize_intent("订单里漏发了两个小样，同时我要申请退款")
    )

    assert result.intent is IntentCategory.MISSING_ITEM
    assert result.secondary_intents == [IntentCategory.RETURN_REFUND]
    assert {item["domain"] for item in result.sub_requests} == {"order", "after_sales"}
    assert result.intent_diagnostics["scoped_retrieval_queries"] == {
        "order": "订单里漏发了两个小样",
        "after_sales": "我要申请退款",
    }


def test_structured_compound_requests_use_registered_beauty_intents(monkeypatch) -> None:
    async def fake_llm(message, history=None):
        return LLMIntentResult(
            intent=IntentCategory.HUMAN_HANDOFF,
            confidence=0.9,
            reasoning="未明确要求人工的复合业务请求",
            success=True,
            diagnostics={
                "llm_is_compound": True,
                "llm_sub_requests": [
                    {
                        "id": "order-child",
                        "text": "订单里漏发两个小样",
                        "domain": "order",
                        "intent": "missing_item",
                        "confidence": 0.9,
                    },
                    {
                        "id": "refund-child",
                        "text": "申请退款",
                        "domain": "after_sales",
                        "intent": "return_refund",
                        "confidence": 0.9,
                    },
                ],
            },
        )

    async def fake_embedding(message):
        return EmbeddingIntentResult(
            intent=IntentCategory.OTHER,
            score=0.0,
            success=False,
        )

    monkeypatch.setattr(recognizer, "_recognize_with_llm", fake_llm)
    monkeypatch.setattr(recognizer, "_recognize_with_embedding", fake_embedding)
    monkeypatch.setattr(
        recognizer,
        "_pattern_recognize",
        lambda message: _pattern(IntentCategory.OTHER),
    )

    result = _run(recognize_intent("请分别处理漏发小样和申请退款"))

    assert result.intent is IntentCategory.MISSING_ITEM
    assert result.secondary_intents == [IntentCategory.RETURN_REFUND]
    assert result.intent_diagnostics["primary_sub_request_id"] == "order-child"
    assert result.intent_diagnostics["compound_top_level_handoff_ignored"] is True
