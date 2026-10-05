from __future__ import annotations

import asyncio

import intent.intent_recognizer as recognizer
from intent.intent_recognizer import (
    EmbeddingIntentResult,
    IntentCategory,
    LLMIntentResult,
    recognize_intent,
)


def _run(coroutine):
    return asyncio.run(coroutine)


def _llm(intent: IntentCategory, confidence: float, success: bool = True):
    return LLMIntentResult(
        intent=intent,
        confidence=confidence,
        reasoning="llm result",
        success=success,
    )


def _embedding(
    intent: IntentCategory,
    score: float,
    success: bool = True,
    local_fallback: bool = False,
):
    return EmbeddingIntentResult(
        intent=intent,
        score=score,
        success=success,
        local_fallback=local_fallback,
    )


def _pattern(intent: IntentCategory, confidence: float, specific_score: float = 0.0):
    return {
        "intent": intent,
        "confidence": confidence,
        "candidates": [intent],
        "scores": {intent.value: confidence},
        "hits": {},
        "specific_intent": IntentCategory.OTHER,
        "specific_score": specific_score,
    }


def test_three_sources_use_07_02_01_weighted_fusion(monkeypatch) -> None:
    async def fake_llm(message, history=None):
        return _llm(IntentCategory.QUERY, 0.8)

    async def fake_embedding(message):
        return _embedding(IntentCategory.REFUND, 0.9)

    monkeypatch.setattr(recognizer, "_recognize_with_llm", fake_llm)
    monkeypatch.setattr(recognizer, "_recognize_with_embedding", fake_embedding)
    monkeypatch.setattr(
        recognizer,
        "_pattern_recognize",
        lambda message: _pattern(IntentCategory.QUERY, 0.5),
    )

    result = _run(recognize_intent("查询订单"))

    assert result.intent is IntentCategory.QUERY
    assert result.confidence == 0.61
    assert result.intent_diagnostics["weights"] == {
        "llm": 0.7,
        "embedding": 0.2,
        "pattern": 0.1,
    }


def test_embedding_failure_switches_to_085_015_weights(monkeypatch) -> None:
    async def fake_llm(message, history=None):
        return _llm(IntentCategory.QUERY, 0.6)

    async def failed_embedding(message):
        return _embedding(IntentCategory.OTHER, 0.0, success=False)

    monkeypatch.setattr(recognizer, "_recognize_with_llm", fake_llm)
    monkeypatch.setattr(recognizer, "_recognize_with_embedding", failed_embedding)
    monkeypatch.setattr(
        recognizer,
        "_pattern_recognize",
        lambda message: _pattern(IntentCategory.QUERY, 0.4),
    )

    result = _run(recognize_intent("查询订单"))

    assert result.intent is IntentCategory.QUERY
    assert result.confidence == 0.57
    assert result.intent_diagnostics["weights"] == {
        "llm": 0.85,
        "embedding": 0.0,
        "pattern": 0.15,
    }


def test_weighted_fusion_below_050_is_hard_blocked_to_other(monkeypatch) -> None:
    async def fake_llm(message, history=None):
        return _llm(IntentCategory.QUERY, 0.49)

    async def failed_embedding(message):
        return _embedding(IntentCategory.OTHER, 0.0, success=False)

    monkeypatch.setattr(recognizer, "_recognize_with_llm", fake_llm)
    monkeypatch.setattr(recognizer, "_recognize_with_embedding", failed_embedding)
    monkeypatch.setattr(
        recognizer,
        "_pattern_recognize",
        lambda message: _pattern(IntentCategory.QUERY, 0.49),
    )

    result = _run(recognize_intent("普通咨询"))

    assert result.intent is IntentCategory.OTHER
    assert result.confidence == 0.0
    assert result.intent_diagnostics["fallback"] == "low_fused_score"


def test_failed_llm_uses_embedding_only_at_or_above_045(monkeypatch) -> None:
    async def failed_llm(message, history=None):
        return _llm(IntentCategory.OTHER, 0.0, success=False)

    monkeypatch.setattr(recognizer, "_recognize_with_llm", failed_llm)
    monkeypatch.setattr(
        recognizer,
        "_pattern_recognize",
        lambda message: _pattern(IntentCategory.OTHER, 0.0),
    )

    async def accepted_embedding(message):
        return _embedding(IntentCategory.REFUND, 0.46)

    monkeypatch.setattr(recognizer, "_recognize_with_embedding", accepted_embedding)
    accepted = _run(recognize_intent("退款咨询"))

    async def rejected_embedding(message):
        return _embedding(IntentCategory.REFUND, 0.44)

    monkeypatch.setattr(recognizer, "_recognize_with_embedding", rejected_embedding)
    rejected = _run(recognize_intent("退款咨询"))

    assert accepted.intent is IntentCategory.REFUND
    assert accepted.confidence == 0.46
    assert accepted.intent_diagnostics["fallback"] == "embedding"
    assert rejected.intent is IntentCategory.OTHER
    assert rejected.confidence == 0.0
    assert rejected.intent_diagnostics["fallback"] == "llm_and_embedding_unavailable"


def test_specific_pattern_refines_generic_only_below_080(monkeypatch) -> None:
    async def query_llm(message, history=None):
        return _llm(IntentCategory.QUERY, 0.6)

    async def failed_embedding(message):
        return _embedding(IntentCategory.OTHER, 0.0, success=False)

    monkeypatch.setattr(recognizer, "_recognize_with_llm", query_llm)
    monkeypatch.setattr(recognizer, "_recognize_with_embedding", failed_embedding)
    monkeypatch.setattr(
        recognizer,
        "_pattern_recognize",
        lambda message: {
            **_pattern(IntentCategory.QUERY, 0.0, specific_score=0.5),
            "specific_intent": IntentCategory.REFUND,
        },
    )

    refined = _run(recognize_intent("退款查询"))
    assert refined.intent is IntentCategory.REFUND
    assert refined.intent_diagnostics["refined_from"] == IntentCategory.QUERY.value

    async def strong_query_llm(message, history=None):
        return _llm(IntentCategory.QUERY, 1.0)

    monkeypatch.setattr(recognizer, "_recognize_with_llm", strong_query_llm)
    not_refined = _run(recognize_intent("退款查询"))

    assert not_refined.intent is IntentCategory.QUERY
    assert "refined_from" not in not_refined.intent_diagnostics


def test_500_override_is_applied_after_fusion(monkeypatch) -> None:
    async def fake_llm(message, history=None):
        return _llm(IntentCategory.QUERY, 0.6)

    async def failed_embedding(message):
        return _embedding(IntentCategory.OTHER, 0.0, success=False)

    monkeypatch.setattr(recognizer, "_recognize_with_llm", fake_llm)
    monkeypatch.setattr(recognizer, "_recognize_with_embedding", failed_embedding)
    monkeypatch.setattr(
        recognizer,
        "_pattern_recognize",
        lambda message: _pattern(IntentCategory.QUERY, 0.4),
    )

    result = _run(recognize_intent("页面返回 500"))

    assert result.intent is IntentCategory.TECHNICAL_CRASH
    assert result.confidence >= 0.8
    assert result.intent_diagnostics["status_code_override"] is True
