from __future__ import annotations

import asyncio

import core.intent_recognizer as recognizer
from core.intent_recognizer import (
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


def _llm(intent: IntentCategory, confidence: float = 0.95):
    return LLMIntentResult(
        intent=intent,
        confidence=confidence,
        reasoning="compound test",
        success=True,
    )


def _embedding(intent: IntentCategory, score: float = 0.9):
    return EmbeddingIntentResult(intent=intent, score=score, success=True)


def _pattern(intent: IntentCategory, confidence: float = 0.8):
    return {
        "intent": intent,
        "confidence": confidence,
        "candidates": [intent],
        "scores": {intent.value: confidence},
        "hits": {},
        "specific_intent": intent,
        "specific_score": confidence,
    }


def test_long_single_intent_without_split_signal_does_not_enter_compound_branch() -> None:
    message = "电脑开不了机页面也打不开但我想先了解一下如何排查"

    clauses = split_compound_clauses(message)
    secondary, trigger, diagnostics = detect_compound_intents(
        message, IntentCategory.TECHNICAL
    )

    assert len(clauses) == 1
    assert secondary == []
    assert trigger == ""
    assert diagnostics["compound_detected"] is False


def test_cross_domain_technical_and_billing_request_keeps_billing_secondary() -> None:
    message = "电脑开不了机，修不好怎么退货？"

    secondary, trigger, diagnostics = detect_compound_intents(
        message, IntentCategory.TECHNICAL
    )

    assert secondary == [IntentCategory.REFUND]
    assert trigger == "，"
    assert diagnostics["clauses"] == ["电脑开不了机", "修不好怎么退货"]
    assert diagnostics["clause_intents"] == [
        IntentCategory.TECHNICAL.value,
        IntentCategory.REFUND.value,
    ]
    scoped = build_scoped_retrieval_queries(
        message, secondary, diagnostics["clause_matches"]
    )
    assert scoped["billing"] == "修不好怎么退货"


def test_same_domain_technical_clauses_are_not_secondary_agents() -> None:
    message = "开不了机而且屏幕花屏"

    secondary, trigger, _ = detect_compound_intents(
        message, IntentCategory.TECHNICAL
    )

    assert secondary == []
    assert trigger == "而且"


def test_secondary_intents_are_deduplicated_and_hard_limited_to_three() -> None:
    message = "开不了机，而且我要退款，另外要开发票，以及物流在哪里，同时账户被盗"

    secondary, _, _ = detect_compound_intents(message, IntentCategory.TECHNICAL)

    assert len(secondary) == 3
    assert secondary == [
        IntentCategory.REFUND,
        IntentCategory.INVOICE,
        IntentCategory.LOGISTICS,
    ]


def test_human_handoff_primary_is_terminal_and_clears_all_secondary_intents() -> None:
    message = "立刻帮我转人工，顺便查个物流"

    secondary, trigger, diagnostics = detect_compound_intents(
        message, IntentCategory.HUMAN_HANDOFF
    )

    assert secondary == []
    assert trigger == ""
    assert diagnostics["terminal_handoff"] is True


def test_recognize_intent_attaches_compound_fields_and_scoped_queries(monkeypatch) -> None:
    async def fake_llm(message, history=None):
        return _llm(IntentCategory.TECHNICAL)

    async def fake_embedding(message):
        return _embedding(IntentCategory.TECHNICAL)

    monkeypatch.setattr(recognizer, "_recognize_with_llm", fake_llm)
    monkeypatch.setattr(recognizer, "_recognize_with_embedding", fake_embedding)
    monkeypatch.setattr(
        recognizer,
        "_pattern_recognize",
        lambda message: _pattern(IntentCategory.TECHNICAL),
    )

    result = _run(recognize_intent("电脑开不了机，修不好怎么退货？"))

    assert result.intent is IntentCategory.TECHNICAL
    assert result.secondary_intents == [IntentCategory.REFUND]
    assert result.compound_trigger == "，"
    assert result.intent_diagnostics["scoped_retrieval_queries"]["billing"] == (
        "修不好怎么退货"
    )


def test_recognize_intent_clears_secondary_for_handoff(monkeypatch) -> None:
    async def fake_llm(message, history=None):
        return _llm(IntentCategory.HUMAN_HANDOFF)

    async def fake_embedding(message):
        return _embedding(IntentCategory.HUMAN_HANDOFF)

    monkeypatch.setattr(recognizer, "_recognize_with_llm", fake_llm)
    monkeypatch.setattr(recognizer, "_recognize_with_embedding", fake_embedding)
    monkeypatch.setattr(
        recognizer,
        "_pattern_recognize",
        lambda message: _pattern(IntentCategory.HUMAN_HANDOFF),
    )

    result = _run(recognize_intent("立刻帮我转人工，顺便查个物流"))

    assert result.intent is IntentCategory.HUMAN_HANDOFF
    assert result.secondary_intents == []
    assert result.compound_trigger == ""
