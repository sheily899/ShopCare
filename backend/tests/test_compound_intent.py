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
    assert scoped["technical"] == "电脑开不了机"


def test_payment_processing_clause_is_classified_as_billing_subrequest() -> None:
    message = "页面打不开，同时我还想确认一笔支付是否处于处理中，请分别处理这两个问题。"

    secondary, _, diagnostics = detect_compound_intents(
        message, IntentCategory.TECHNICAL
    )

    assert secondary == [IntentCategory.PAYMENT_ISSUE]
    assert diagnostics["sub_requests"][0]["domain"] == "technical"
    assert diagnostics["sub_requests"][1]["domain"] == "billing"


def test_compound_subrequests_create_one_scoped_query_per_business_domain() -> None:
    message = "页面返回500，而且支付结果也没确认，请分别保留两边的待确认项。"
    secondary, _, diagnostics = detect_compound_intents(
        message, IntentCategory.TECHNICAL
    )

    assert secondary == [IntentCategory.PAYMENT_ISSUE]
    queries = build_scoped_retrieval_queries(
        message, secondary, diagnostics["clause_matches"]
    )
    assert set(queries) == {"technical", "billing"}
    assert "500" in queries["technical"]
    assert "支付" in queries["billing"]


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


def test_low_confidence_compound_message_uses_first_specific_subrequest_as_primary(monkeypatch) -> None:
    async def fake_llm(message, history=None):
        return _llm(IntentCategory.OTHER, confidence=0.2)

    async def fake_embedding(message):
        return _embedding(IntentCategory.OTHER, score=0.1)

    monkeypatch.setattr(recognizer, "_recognize_with_llm", fake_llm)
    monkeypatch.setattr(recognizer, "_recognize_with_embedding", fake_embedding)
    monkeypatch.setattr(
        recognizer,
        "_pattern_recognize",
        lambda message: _pattern(IntentCategory.OTHER, confidence=0.0),
    )

    result = _run(
        recognize_intent("页面打不开，同时支付是否处于处理中，请分别处理")
    )

    assert result.intent is IntentCategory.TECHNICAL
    assert result.secondary_intents == [IntentCategory.PAYMENT_ISSUE]
    assert result.intent_diagnostics["compound_detected"] is True


def test_valid_llm_sub_requests_enable_domain_scoped_queries_without_connector(monkeypatch) -> None:
    async def fake_llm(message, history=None):
        return LLMIntentResult(
            intent=IntentCategory.TECHNICAL,
            confidence=0.9,
            reasoning="两个独立业务问题",
            success=True,
            diagnostics={
                "llm_is_compound": True,
                "llm_sub_requests": [
                    {
                        "id": "a",
                        "text": "登录时报错",
                        "domain": "technical",
                        "intent": IntentCategory.TECHNICAL_LOGIN.value,
                        "confidence": 0.9,
                    },
                    {
                        "id": "b",
                        "text": "申请退款",
                        "domain": "billing",
                        "intent": IntentCategory.REFUND.value,
                        "confidence": 0.9,
                    },
                ],
            },
        )

    monkeypatch.setattr(recognizer, "_recognize_with_llm", fake_llm)
    async def fake_embedding(message):
        return _embedding(IntentCategory.TECHNICAL, score=0.9)

    monkeypatch.setattr(recognizer, "_recognize_with_embedding", fake_embedding)
    monkeypatch.setattr(
        recognizer,
        "_pattern_recognize",
        lambda message: _pattern(IntentCategory.TECHNICAL, confidence=0.8),
    )

    result = _run(recognize_intent("登录时报错 申请退款"))

    assert result.secondary_intents == [IntentCategory.REFUND]
    assert result.intent_diagnostics["scoped_retrieval_queries"] == {
        "technical": "登录时报错",
        "billing": "申请退款",
    }
