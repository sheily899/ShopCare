from __future__ import annotations

import pytest

from intent.intent_recognizer import (
    GENERIC_PATTERNS,
    SPECIFIC_PATTERNS,
    IntentCategory,
    UrgencyLevel,
    apply_500_override,
    calculate_urgency,
    extract_entities,
    match_patterns,
)


ALL_INTENTS = set(IntentCategory)


def test_pattern_tables_cover_all_intent_categories() -> None:
    assert set(SPECIFIC_PATTERNS) | set(GENERIC_PATTERNS) == ALL_INTENTS


@pytest.mark.parametrize(
    ("message", "intent"),
    [
        ("我要退款", IntentCategory.REFUND),
        ("物流到哪里了", IntentCategory.LOGISTICS),
        ("我要投诉", IntentCategory.COMPLAINT),
        ("我要找人工", IntentCategory.HUMAN_HANDOFF),
    ],
)
def test_pattern_match_returns_expected_intent_and_structured_scores(
    message: str, intent: IntentCategory
) -> None:
    result = match_patterns(message)

    assert result["intent"] == intent
    assert result["confidence"] == pytest.approx(0.5)
    assert intent in result["candidates"]
    assert result["scores"][intent.value] == pytest.approx(0.5)
    assert result["specific_score"] == pytest.approx(0.5 if intent in SPECIFIC_PATTERNS else 0.0)


def test_pattern_score_increases_for_two_and_three_distinct_hits() -> None:
    two_hits = match_patterns("我要退款并退货")
    three_hits = match_patterns("我要退款、退货，refund")

    assert two_hits["intent"] == IntentCategory.REFUND
    assert two_hits["confidence"] == pytest.approx(0.75)
    assert two_hits["hits"][IntentCategory.REFUND.value] == 2
    assert three_hits["intent"] == IntentCategory.REFUND
    assert three_hits["confidence"] == pytest.approx(1.0)
    assert three_hits["hits"][IntentCategory.REFUND.value] == 3


def test_logistics_pattern_score_increases_for_multiple_rules() -> None:
    result = match_patterns("物流和快递配送进度")

    assert result["intent"] == IntentCategory.LOGISTICS
    assert result["confidence"] == pytest.approx(1.0)
    assert result["hits"][IntentCategory.LOGISTICS.value] >= 3


def test_pattern_result_exposes_specific_score_for_generic_and_specific_candidates() -> None:
    result = match_patterns("账单退款问题")

    assert result["intent"] == IntentCategory.REFUND
    assert result["specific_intent"] == IntentCategory.REFUND
    assert result["specific_score"] == pytest.approx(0.5)
    assert result["confidence"] >= result["specific_score"]


def test_extract_entities_uses_exact_five_keys_and_keeps_product_empty() -> None:
    message = "订单号: AB123456，金额 ¥99.50，日期 2024-01-02，近一周错误码 500"

    entities = extract_entities(message)

    assert set(entities) == {"order_id", "product", "date", "amount", "error_code"}
    assert entities["order_id"] == ["AB123456"]
    assert entities["product"] == []
    assert "2024-01-02" in entities["date"]
    assert "近一周" in entities["date"]
    assert "¥99.50" in entities["amount"]
    assert entities["error_code"] == ["500"]


def test_extract_entities_supports_order_hash_and_common_amount_units() -> None:
    entities = extract_entities("#ORD-2024，另收 30 元，昨天出现 HTTP 404")

    assert entities["order_id"] == ["ORD-2024"]
    assert entities["amount"] == ["30 元"]
    assert entities["date"] == ["昨天"]
    assert entities["error_code"] == ["404"]
    assert entities["product"] == []


def test_urgency_critical_has_priority_over_high_and_medium() -> None:
    assert calculate_urgency("立刻今天处理") is UrgencyLevel.CRITICAL


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("今天处理", UrgencyLevel.HIGH),
        ("这周处理", UrgencyLevel.MEDIUM),
        ("普通咨询", UrgencyLevel.LOW),
    ],
)
def test_urgency_keyword_levels(message: str, expected: UrgencyLevel) -> None:
    assert calculate_urgency(message) is expected


def test_urgency_uses_intent_fallback_when_no_urgency_keyword_exists() -> None:
    assert calculate_urgency("请帮我", IntentCategory.HUMAN_HANDOFF) is UrgencyLevel.HIGH
    assert calculate_urgency("服务不好", IntentCategory.ESCALATION) is UrgencyLevel.HIGH
    assert calculate_urgency("服务不好", IntentCategory.COMPLAINT) is UrgencyLevel.MEDIUM
    assert calculate_urgency("我想咨询", IntentCategory.QUERY) is UrgencyLevel.LOW


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
