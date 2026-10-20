from __future__ import annotations

import asyncio
import intent.intent_recognizer as recognizer
from intent.intent_recognizer import (
    EmbeddingIntentResult,
    EMBEDDING_TEMPLATES,
    IntentCategory,
    LLMIntentResult,
    _hash_ngram_vector,
    _recognize_with_embedding,
    recognize_intent,
)


def _run(coroutine):
    return asyncio.run(coroutine)


def test_local_embedding_fallback_is_fixed_256_dimension_and_bounded(monkeypatch) -> None:
    monkeypatch.setattr(recognizer, "_get_local_embedding_function", lambda: None)
    vector = _hash_ngram_vector("退款 refund")
    result = _run(_recognize_with_embedding("我要退款"))

    assert len(vector) == 256
    assert result.local_fallback is True
    assert result.success is True
    assert 0.0 <= result.score <= 1.0
    assert result.intent is IntentCategory.RETURN_REFUND


def test_intent_result_keeps_source_and_final_diagnostics(monkeypatch) -> None:
    async def fake_llm(message, history=None):
        return LLMIntentResult(
            intent=IntentCategory.ORDER_STATUS_BEAUTY,
            confidence=0.77,
            reasoning="包含技术问题",
            success=True,
        )

    async def fake_embedding(message):
        return EmbeddingIntentResult(
            intent=IntentCategory.ORDER_STATUS_BEAUTY,
            score=0.66,
            success=True,
            local_fallback=True,
        )

    monkeypatch.setattr(recognizer, "_recognize_with_llm", fake_llm)
    monkeypatch.setattr(recognizer, "_recognize_with_embedding", fake_embedding)
    monkeypatch.setattr(
        recognizer,
        "_pattern_recognize",
        lambda message: {
            "intent": IntentCategory.ORDER_STATUS_BEAUTY,
            "confidence": 0.5,
            "scores": {IntentCategory.ORDER_STATUS_BEAUTY.value: 0.5},
            "specific_intent": IntentCategory.OTHER,
            "specific_score": 0.0,
        },
    )

    result = _run(recognize_intent("我的订单到哪里了"))

    assert result.source_intents == {
        "llm": "order_status",
        "embedding": "order_status",
        "pattern": "order_status",
    }
    assert result.source_scores == {
        "llm": 0.77,
        "embedding": 0.66,
        "pattern": 0.5,
    }
    assert result.intent is IntentCategory.ORDER_STATUS_BEAUTY
    assert result.intent_diagnostics["final_intent"] == "order_status"
    assert result.intent_diagnostics["final_confidence"] == result.confidence
    assert result.intent_diagnostics["embedding_local_fallback"] is True
    assert result.latency_ms >= 0.0
