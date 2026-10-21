from agents.agent_orchestrator import Request, build_business_field_state
from agents.response_guard import asked_fields_in_text, filter_redundant_field_questions
from api.main import _pending_questions, _pending_text_fields_for_response
from intent.beauty_intent_ticket_registry import BEAUTY_INTENT_TICKET_REGISTRY
from intent.intent_recognizer import IntentCategory, extract_entities
from intent.slot_registry import (
    apply_pending_text_answer,
    get_slot,
    pending_text_answer_field_id,
)


def test_extract_entities_normalizes_natural_answers_for_refund_slots() -> None:
    entities = extract_entities(
        "我用了一个星期，第一天就红肿，目前没有在使用，商品已经拆封，小样没拆，我要整单退款。"
    )

    assert entities["onset_time"] == ["第一天"]
    assert entities["stopped_use"] == ["已停用"]
    assert entities["opened_or_used"] == ["已使用"]
    assert entities["return_scope"] == ["整单退款"]
    assert entities["gift_opened_or_used"] == ["未拆封"]
    assert entities["return_reason"] == ["使用不适"]


def test_extraction_does_not_confuse_refund_wait_time_with_product_usage() -> None:
    entities = extract_entities("商品未拆封，退款申请已经等了一周。")

    assert entities["opened_or_used"] == ["未拆封"]


def test_extraction_distinguishes_opened_but_unused_from_used() -> None:
    entities = extract_entities("正装已经拆封，但没有使用；小样也没拆。")

    assert entities["opened_or_used"] == ["已拆封未使用"]
    assert entities["gift_opened_or_used"] == ["未拆封"]


def test_refund_complaint_extracts_plain_language_redness_and_itching() -> None:
    entities = extract_entities("我用了你们家的粉底液，结果导致我的脸部现在全部泛红发痒，我要退款")

    assert entities["symptom"] == ["泛红", "发痒"]
    assert entities["symptom_area"] == ["脸部"]
    assert entities["return_reason"] == ["使用不适"]


def test_field_state_merges_all_active_intents_and_uses_latest_case_facts() -> None:
    request = Request(
        message="我已经停用了，想申请退款。",
        intent=IntentCategory.ADVERSE_REACTION,
        secondary_intents=[IntentCategory.RETURN_REFUND],
        entities={"stopped_use": ["已停用"]},
        case_facts={
            "facts": {
                "order_id": [
                    {"value": "SC-DEMO-4821", "confirmation_status": "confirmed"}
                ],
                "symptom": [
                    {"value": "脸部红肿", "confirmation_status": "confirmed"}
                ],
                "opened_or_used": [
                    {"value": "已使用", "confirmation_status": "confirmed"}
                ],
            }
        },
    )

    state = build_business_field_state(request)

    assert state["intent_ids"] == ["adverse_reaction", "return_refund"]
    assert state["known_fields"]["stopped_use"] == ["已停用"]
    assert state["known_fields"]["symptom"] == ["脸部红肿"]
    assert state["missing_fields"] == ["return_reason"]
    assert state["missing_text_fields"] == ["return_reason"]
    assert state["askable_text_fields"] == ["return_reason"]


def test_pending_questions_uses_all_intents_and_only_shows_missing_enum_slots() -> None:
    entities = {
        "order_id": ["SC-DEMO-4821"],
        "symptom": ["脸部红肿"],
        "stopped_use": ["已停用"],
        "medical_visit": ["没就医"],
        "return_reason": ["使用不适"],
        "return_scope": ["整单退款"],
    }

    questions = _pending_questions(
        IntentCategory.ADVERSE_REACTION,
        entities,
        secondary_intents=[IntentCategory.RETURN_REFUND],
    )

    assert [question["slot_id"] for question in questions] == ["opened_or_used"]
    assert questions[0]["options"]
    assert "medical_visit" not in [question["slot_id"] for question in questions]


def test_model_question_guard_removes_known_and_nonblocking_health_followups() -> None:
    text = "请问退款原因是什么？另外，脸部具体哪里红肿、有没有就医？"

    filtered, removed_fields = filter_redundant_field_questions(
        text,
        {
            "symptom": ["脸部红肿"],
            "symptom_area": ["脸部"],
            "medical_visit": ["没就医"],
        },
        allowed_fields={"return_reason"},
    )

    assert filtered == "请问退款原因是什么？"
    assert set(removed_fields) == {"symptom_area", "medical_visit"}


def test_model_question_guard_allows_only_first_missing_text_slot() -> None:
    filtered, removed_fields = filter_redundant_field_questions(
        "请问订单号是多少？另外，您这次退款原因是什么？",
        {},
        allowed_fields={"order_id"},
    )

    assert filtered == "请问订单号是多少？"
    assert removed_fields == ["return_reason"]


def test_every_registered_intent_blocking_field_has_a_slot_definition() -> None:
    missing = {
        item["intent_id"]: [
            field for field in item["blocking_fields"] if get_slot(field) is None
        ]
        for item in BEAUTY_INTENT_TICKET_REGISTRY
    }

    assert {intent_id: fields for intent_id, fields in missing.items() if fields} == {}


def test_short_free_text_reply_fills_the_field_asked_in_previous_turn() -> None:
    entities = apply_pending_text_answer(
        "因为用后脸红",
        {},
        [{"field_id": "return_reason", "intent_ids": ["return_refund"]}],
    )

    assert entities["return_reason"] == ["因为用后脸红"]


def test_pending_text_answer_does_not_capture_a_new_question_or_overwrite_known_value() -> None:
    new_question = apply_pending_text_answer(
        "我还想问一下这款精华的成分？",
        {},
        [{"field_id": "return_reason", "intent_ids": ["return_refund"]}],
    )
    canceled_request = apply_pending_text_answer(
        "我决定暂时不退款了",
        {},
        [{"field_id": "return_reason", "intent_ids": ["return_refund"]}],
    )
    known_value = apply_pending_text_answer(
        "因为用后脸红",
        {"return_reason": ["买错了"]},
        [{"field_id": "return_reason", "intent_ids": ["return_refund"]}],
    )

    assert "return_reason" not in new_question
    assert "return_reason" not in canceled_request
    assert known_value["return_reason"] == ["买错了"]


def test_pending_text_answer_does_not_capture_a_different_recognized_intent() -> None:
    entities = apply_pending_text_answer(
        "想问一下防晒霜适合敏感肌吗",
        {},
        [{
            "field_id": "return_reason",
            "intent_ids": ["return_refund"],
            "context_intent_ids": ["return_refund", "adverse_reaction"],
        }],
        current_intent_id="ingredient_fit",
    )

    assert "return_reason" not in entities


def test_recognized_answer_still_resumes_the_pending_field_context() -> None:
    message = "SC-DEMO-4821"
    pending = [{
        "field_id": "order_id",
        "intent_ids": ["return_refund"],
        "context_intent_ids": ["return_refund"],
    }]

    entities = apply_pending_text_answer(
        message,
        extract_entities(message),
        pending,
        current_intent_id="return_refund",
    )
    assert entities["order_id"] == ["SC-DEMO-4821"]
    assert pending_text_answer_field_id(
        message,
        pending,
        current_intent_id="return_refund",
    ) == "order_id"


def test_adverse_reaction_can_answer_a_pending_refund_reason() -> None:
    raw_entities = extract_entities("因为用后脸红")
    entities = apply_pending_text_answer(
        "因为用后脸红",
        raw_entities,
        [{
            "field_id": "return_reason",
            "intent_ids": ["return_refund"],
            "context_intent_ids": ["return_refund"],
        }],
        current_intent_id="adverse_reaction",
    )

    assert entities["return_reason"] == ["因为用后脸红"]


def test_pending_refund_reason_answer_closes_the_missing_field_in_composite_case() -> None:
    answer = "因为用后脸红"
    entities = apply_pending_text_answer(
        answer,
        extract_entities(answer),
        [{
            "field_id": "return_reason",
            "intent_ids": ["return_refund"],
            "context_intent_ids": ["return_refund"],
        }],
        current_intent_id="adverse_reaction",
    )
    request = Request(
        message=answer,
        intent=IntentCategory.ADVERSE_REACTION,
        secondary_intents=[IntentCategory.RETURN_REFUND],
        entities=entities,
        case_facts={"facts": {
            "order_id": [{"value": "SC-DEMO-4821", "confirmation_status": "confirmed"}],
            "opened_or_used": [{"value": "已使用", "confirmation_status": "confirmed"}],
        }},
    )

    state = build_business_field_state(request)

    assert state["known_fields"]["return_reason"] == [answer]
    assert state["known_fields"]["symptom"] == ["脸红"]
    assert state["missing_fields"] == []


def test_pending_text_field_is_persisted_only_when_reply_actually_asks_it() -> None:
    asked = asked_fields_in_text(
        "您这次申请退款的原因是什么？",
        ["return_reason", "order_id"],
    )
    natural_wording = asked_fields_in_text(
        "您这次想退货是因为什么？",
        ["return_reason"],
    )
    mentioned_but_not_asked = asked_fields_in_text(
        "退款原因我会记录，订单也已核对。",
        ["return_reason", "order_id"],
    )

    assert asked == ["return_reason"]
    assert natural_wording == ["return_reason"]
    assert mentioned_but_not_asked == []


def test_api_persists_only_the_first_missing_text_field_the_reply_actually_asks() -> None:
    pending = _pending_text_fields_for_response(
        "请问订单号是多少？另外，您这次退款原因是什么？",
        {
            "intent_ids": ["return_refund"],
            "askable_text_fields": ["order_id"],
            "missing_text_fields": ["order_id", "return_reason"],
        },
    )

    assert pending == [{
        "field_id": "order_id",
        "intent_ids": ["return_refund"],
        "context_intent_ids": ["return_refund"],
    }]
