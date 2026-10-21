from __future__ import annotations

import pytest

from intent.intent_recognizer import (
    GENERIC_PATTERNS,
    SPECIFIC_PATTERNS,
    IntentCategory,
    UrgencyLevel,
    apply_500_override,
    calculate_urgency,
    detect_emotion,
    extract_entities,
    match_patterns,
)
from intent.beauty_intent_registry import LLM_INTENT_SCHEMA


ALL_INTENTS = {
    IntentCategory(intent_id) for intent_id in LLM_INTENT_SCHEMA["enum"]
} | {IntentCategory.HUMAN_HANDOFF, IntentCategory.OTHER}


def test_pattern_tables_cover_registered_beauty_intents_and_control_results() -> None:
    assert set(SPECIFIC_PATTERNS) | set(GENERIC_PATTERNS) == ALL_INTENTS


@pytest.mark.parametrize(
    ("message", "intent", "confidence"),
    [
        ("我想申请退货退款", IntentCategory.RETURN_REFUND, 0.75),
        ("物流到哪里了", IntentCategory.ORDER_STATUS_BEAUTY, 0.5),
        ("这个精华有哪些成分", IntentCategory.INGREDIENT_FIT, 0.5),
        ("我要找人工", IntentCategory.HUMAN_HANDOFF, 0.75),
    ],
)
def test_pattern_match_returns_expected_intent_and_structured_scores(
    message: str, intent: IntentCategory, confidence: float
) -> None:
    result = match_patterns(message)

    assert result["intent"] == intent
    assert result["confidence"] == pytest.approx(confidence)
    assert intent in result["candidates"]
    assert result["scores"][intent.value] == pytest.approx(confidence)
    assert result["specific_score"] == pytest.approx(confidence if intent in SPECIFIC_PATTERNS else 0.0)


def test_pattern_score_increases_for_two_and_three_distinct_hits() -> None:
    two_hits = match_patterns("申请退款和退款申请")
    three_hits = match_patterns("申请退款、退款申请、退货")

    assert two_hits["intent"] == IntentCategory.RETURN_REFUND
    assert two_hits["confidence"] == pytest.approx(0.75)
    assert two_hits["hits"][IntentCategory.RETURN_REFUND.value] == 2
    assert three_hits["intent"] == IntentCategory.RETURN_REFUND
    assert three_hits["confidence"] == pytest.approx(1.0)
    assert three_hits["hits"][IntentCategory.RETURN_REFUND.value] == 3


def test_logistics_pattern_score_increases_for_multiple_rules() -> None:
    result = match_patterns("物流不更新，快递也卡在中转站，帮我催查")

    assert result["intent"] == IntentCategory.LOGISTICS_STALLED
    assert result["confidence"] >= 0.75
    assert result["hits"][IntentCategory.LOGISTICS_STALLED.value] >= 2


def test_pattern_result_exposes_specific_score_for_generic_and_specific_candidates() -> None:
    result = match_patterns("订单 SC-DEMO-4821，申请退款")

    assert result["intent"] == IntentCategory.RETURN_REFUND
    assert result["specific_intent"] == IntentCategory.RETURN_REFUND
    assert result["specific_score"] == pytest.approx(0.5)
    assert result["confidence"] >= result["specific_score"]


def test_extract_entities_uses_beauty_case_fields_and_keeps_unprovided_values_empty() -> None:
    message = "订单号: AB123456，色号 02，数量 2 件，脸部发红"

    entities = extract_entities(message)

    assert set(entities) == {
        "order_id", "product_name", "sku", "shade", "quantity", "symptom",
        "symptom_area", "onset_time", "stopped_use", "medical_visit",
        "signed_info", "intercept_or_change", "missing_item", "wrong_item", "damaged_area", "tracking_no",
        "return_reason", "opened_or_used", "return_scope", "gift_opened_or_used",
        "refund_amount", "price_difference",
    }
    assert entities["order_id"] == ["AB123456"]
    assert entities["shade"] == ["02"]
    assert entities["quantity"] == ["2"]
    assert entities["symptom"] == ["发红"]
    assert entities["symptom_area"] == ["脸部"]
    assert entities["product_name"] == []


def test_extract_entities_supports_order_hash_and_beauty_case_entities() -> None:
    entities = extract_entities("#ORD-2024，色号 N2，已停用")

    assert entities["order_id"] == ["ORD-2024"]
    assert entities["shade"] == ["N2"]
    assert entities["stopped_use"] == ["已停用"]
    assert entities["product_name"] == []


def test_urgency_thresholds_come_from_the_shared_policy() -> None:
    # 关键词转人工本身不抬高风险；时间紧迫性仍按配置进入 P2。
    assert calculate_urgency("立刻今天处理") is UrgencyLevel.MEDIUM


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("今天处理", UrgencyLevel.MEDIUM),
        ("这周处理", UrgencyLevel.MEDIUM),
        ("普通咨询", UrgencyLevel.LOW),
    ],
)
def test_urgency_keyword_levels(message: str, expected: UrgencyLevel) -> None:
    assert calculate_urgency(message) is expected


def test_urgency_uses_intent_fallback_when_no_urgency_keyword_exists() -> None:
    assert calculate_urgency("请帮我", IntentCategory.HUMAN_HANDOFF) is UrgencyLevel.LOW
    assert calculate_urgency("服务不好", IntentCategory.ESCALATION) is UrgencyLevel.LOW
    assert calculate_urgency("服务不好", IntentCategory.COMPLAINT) is UrgencyLevel.MEDIUM
    assert calculate_urgency("我想咨询", IntentCategory.QUERY) is UrgencyLevel.LOW


@pytest.mark.parametrize(
    ("message", "refund_is_positive"),
    [
        ("我不要退款", False),
        ("我不要退款，只想换货", False),
        ("我不要只退款，我还要你赔偿", True),
        ("我不只是想退款，还想要赔偿", True),
        ("我不是不想退款", True),
        ("我之前不想退款，现在想申请退款", True),
    ],
)
def test_refund_keyword_respects_negation_scope_and_additive_meaning(
    message: str, refund_is_positive: bool
) -> None:
    result = match_patterns(message)

    assert (IntentCategory.RETURN_REFUND in result["scores"]) is refund_is_positive


@pytest.mark.parametrize(
    ("message", "expected_symptoms", "expected_reason"),
    [
        ("我没有红肿，只是想咨询", [], []),
        ("我之前脸红过，现在已经好了，但我想退款", [], ["使用不适"]),
        ("现在脸上出现红肿，我想退款", ["红肿"], ["使用不适"]),
        ("我想退款，不是因为过敏，是买错了", [], ["买错/重复购买"]),
    ],
)
def test_symptom_and_refund_reason_slots_respect_negation_and_time_scope(
    message: str, expected_symptoms: list[str], expected_reason: list[str]
) -> None:
    entities = extract_entities(message)

    assert entities["symptom"] == expected_symptoms
    assert entities["return_reason"] == expected_reason


@pytest.mark.parametrize(
    ("message", "expected_emotion"),
    [
        ("我不是要投诉，只想问退款", "calm"),
        ("我会向12315投诉", "threatening"),
        ("我之前投诉过，这次只问物流", "calm"),
    ],
)
def test_emotion_keywords_ignore_negated_or_historical_mentions(
    message: str, expected_emotion: str
) -> None:
    emotion, _, _ = detect_emotion(message)

    assert emotion == expected_emotion


def test_500_error_overrides_non_login_intent_and_raises_confidence_to_08() -> None:
    intent, confidence = apply_500_override(
        "页面返回 500", IntentCategory.TECHNICAL, 0.42
    )

    assert intent is IntentCategory.TECHNICAL_CRASH
    assert confidence == pytest.approx(0.8)


def test_500_error_override_preserves_higher_confidence_and_login_exception() -> None:
    crash_intent, crash_confidence = apply_500_override(
        "服务 500", IntentCategory.QUERY, 0.95
    )
    login_intent, login_confidence = apply_500_override(
        "登录返回 500", IntentCategory.TECHNICAL_LOGIN, 0.42
    )

    assert crash_intent is IntentCategory.TECHNICAL_CRASH
    assert crash_confidence == pytest.approx(0.95)
    assert login_intent is IntentCategory.TECHNICAL_LOGIN
    assert login_confidence == pytest.approx(0.42)
