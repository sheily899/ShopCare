"""交接时的同步紧急度规则：转人工信号本身不加分。"""

from handoff.config import load_handoff_urgency_config
from handoff.urgency import score_temporary_urgency


def _score(message: str, intent: str | None = None) -> dict:
    return score_temporary_urgency(message, load_handoff_urgency_config(), intent=intent)


def test_direct_handoff_without_other_risk_stays_p3() -> None:
    result = _score("请帮我转人工")

    assert result["priority"] == "P3"
    assert result["score"] == 0
    assert result["reasons"][0]["rule_id"] == "default.no_higher_signal"
    assert result["automatic_handoff"] is False


def test_historical_resolved_reaction_does_not_trigger_current_safety_priority() -> None:
    result = _score("我之前对芒果过敏，这次没事，请帮我转人工")

    assert result["priority"] == "P3"
    assert result["reasons"][0]["evidence"] == ""


def test_ordinary_redness_with_refund_is_not_automatic_handoff() -> None:
    result = _score("用了之后脸红肿，我想退款", intent="adverse_reaction")

    assert result["priority"] == "P1"
    assert result["automatic_handoff"] is False
    assert result["reasons"][0]["rule_id"] == "safety.adverse_reaction"
    assert result["reasons"][0]["evidence"] == "红肿"
    assert result["rule_version"] == "handoff-urgency-v2"

    from intent.intent_recognizer import IntentCategory, UrgencyLevel, calculate_urgency
    assert calculate_urgency(
        "用了之后脸红肿，我想退款", IntentCategory.ADVERSE_REACTION
    ) is UrgencyLevel.HIGH


def test_severe_breathing_or_face_swelling_still_auto_handoffs() -> None:
    for message in ("我现在呼吸困难", "我嘴唇明显肿胀，症状迅速加重"):
        result = _score(message, intent="adverse_reaction")
        assert result["priority"] == "P0"
        assert result["automatic_handoff"] is True


def test_suspected_product_contamination_still_auto_handoffs() -> None:
    result = _score("打开后发现粉底液里有异物，疑似产品污染")

    assert result["priority"] == "P0"
    assert result["automatic_handoff"] is True


def test_negated_external_complaint_does_not_auto_handoff() -> None:
    result = _score("我不是要举报，也不准备投诉")

    assert result["priority"] == "P3"
    assert result["automatic_handoff"] is False


def test_external_escalation_still_auto_handoffs_but_anger_alone_does_not() -> None:
    external = _score("我要向12315投诉这款粉底液")
    angry = _score("这个产品太差劲了，客服一直没有解决", intent="complaint")

    assert external["priority"] == "P0"
    assert external["automatic_handoff"] is True
    assert angry["priority"] == "P2"
    assert angry["automatic_handoff"] is False


def test_abnormal_refund_handoff_does_not_raise_urgency_to_p0() -> None:
    result = _score("这笔退款记录疑似重复", intent="suspected_abnormal_refund")

    assert result["priority"] == "P1"
    assert result["automatic_handoff"] is True


def test_negated_current_reaction_does_not_trigger_safety_priority() -> None:
    result = _score("我没有脸红肿，想找人工问一下")

    assert result["priority"] == "P3"
    assert result["reasons"][0]["evidence"] == ""


def test_negated_handoff_phrase_does_not_request_transfer() -> None:
    from handoff.urgency import is_explicit_handoff_requested

    config = load_handoff_urgency_config()
    assert is_explicit_handoff_requested("我不想再找人工客服了", config) is False
    assert is_explicit_handoff_requested("我需要转人工", config) is True


def test_money_request_alone_does_not_escalate_but_adverse_intent_has_floor() -> None:
    refund = _score("我想申请退款", intent="return_refund")
    adverse = _score("用了之后很不舒服", intent="adverse_reaction")

    assert refund["priority"] == "P3"
    assert refund["automatic_handoff"] is False
    assert adverse["priority"] == "P1"
    assert adverse["automatic_handoff"] is False
    assert adverse["reasons"][-1]["rule_id"] == "intent.priority_floor.adverse_reaction"
