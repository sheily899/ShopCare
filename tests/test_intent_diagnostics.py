from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import core.intent_recognizer as recognizer
from core.intent_recognizer import (
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


def test_local_embedding_fallback_is_fixed_256_dimension_and_bounded() -> None:
    vector = _hash_ngram_vector("退款 refund")
    result = _run(_recognize_with_embedding("我要退款"))

    assert len(vector) == 256
    assert result.local_fallback is True
    assert result.success is True
    assert 0.0 <= result.score <= 1.0
    assert result.intent is IntentCategory.REFUND


def test_configured_remote_embedding_is_async_and_skips_local_fallback(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_API_URL", "https://embedding.test/v1/embeddings")
    texts = [
        text
        for templates in EMBEDDING_TEMPLATES.values()
        for text in templates
    ] + ["我要退款"]
    remote_client = SimpleNamespace(post=AsyncMock())
    remote_client.post.return_value = {
        "data": [
            {"embedding": _hash_ngram_vector(text)}
            for text in texts
        ]
    }

    result = _run(
        _recognize_with_embedding("我要退款", remote_client=remote_client)
    )

    assert result.local_fallback is False
    assert result.success is True
    assert result.intent is IntentCategory.REFUND
    remote_client.post.assert_awaited_once()
    request_payload = remote_client.post.await_args.kwargs["json"]
    assert request_payload["model"] == "voyage-3-lite"
    assert len(request_payload["input"]) == len(texts)


def test_intent_result_keeps_source_and_final_diagnostics(monkeypatch) -> None:
    async def fake_llm(message, history=None):
        return LLMIntentResult(
            intent=IntentCategory.TECHNICAL,
            confidence=0.77,
            reasoning="包含技术问题",
            success=True,
        )

    async def fake_embedding(message):
        return EmbeddingIntentResult(
            intent=IntentCategory.TECHNICAL,
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
            "intent": IntentCategory.TECHNICAL,
            "confidence": 0.5,
            "scores": {IntentCategory.TECHNICAL.value: 0.5},
            "specific_intent": IntentCategory.OTHER,
            "specific_score": 0.0,
        },
    )

    result = _run(recognize_intent("页面打不开"))

    assert result.source_intents == {
        "llm": "technical",
        "embedding": "technical",
        "pattern": "technical",
    }
    assert result.source_scores == {
        "llm": 0.77,
        "embedding": 0.66,
        "pattern": 0.5,
    }
    assert result.intent is IntentCategory.TECHNICAL
    assert result.intent_diagnostics["final_intent"] == "technical"
    assert result.intent_diagnostics["final_confidence"] == result.confidence
    assert result.intent_diagnostics["embedding_local_fallback"] is True
    assert result.latency_ms >= 0.0
