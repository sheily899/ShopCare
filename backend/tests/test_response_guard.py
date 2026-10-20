from agents.response_guard import (
    build_return_refund_request_reply,
    filter_external_refund_instructions,
    filter_redundant_field_questions,
    filter_repeated_answer_content,
    guard_response,
)


def test_response_guard_preserves_safe_model_prose_verbatim() -> None:
    text = "理解您的心情，但不能保证一定退款，我先帮您核对退款状态。"

    assert guard_response(text) == text


def test_response_guard_replaces_internals_as_a_whole_instead_of_deleting_words() -> None:
    text = "已查到订单状态为已签收，order_id=SC-DEMO-4821。"

    assert guard_response(text) == "这项信息目前还无法确认，我会先按已核实的信息继续处理。"


def test_response_guard_blocks_positive_refund_guarantee_but_preserves_negation() -> None:
    assert guard_response("我们一定会退款。") == "这项信息目前还无法确认，我会先按已核实的信息继续处理。"
    assert guard_response("不能保证一定会退款，先核对订单规则。") == "不能保证一定会退款，先核对订单规则。"


def test_return_refund_request_uses_chat_and_does_not_direct_customer_to_another_page() -> None:
    reply = build_return_refund_request_reply({
        "status": "draft",
        "fields": {
            "order_id": "SC-DEMO-4821",
            "return_reason": "买错了",
            "opened_or_used": "正装未拆封",
        },
        "blocking_fields": ["order_id", "return_reason", "opened_or_used"],
    })

    assert "SC-DEMO-4821" in reply
    assert "买错了" in reply
    assert "正装未拆封" in reply
    assert "待客服审核" in reply
    assert "订单页" not in reply
    assert "已退款" not in reply


def test_return_refund_request_asks_only_first_missing_refund_field() -> None:
    reply = build_return_refund_request_reply({
        "status": "draft",
        "fields": {"order_id": "SC-DEMO-4821"},
        "blocking_fields": ["order_id", "return_reason", "opened_or_used"],
    })

    assert "这次退款的原因" in reply
    assert "拆封" not in reply
    assert "症状" not in reply
    assert "订单号是多少" not in reply


def test_external_refund_page_direction_is_removed_without_dropping_other_help() -> None:
    reply, removed = filter_external_refund_instructions(
        "我会先帮你核对。你可以在订单页发起退货退款。审核结果会在这里回复。"
    )

    assert removed is True
    assert "订单页" not in reply
    assert "我会先帮你核对" in reply
    assert "审核结果会在这里回复" in reply


def test_redundant_question_guard_removes_known_field_question_but_keeps_missing_one() -> None:
    text = "您已说明商品未拆封。请问商品是否拆封或使用过？退货原因是什么？"

    filtered, removed_fields = filter_redundant_field_questions(
        text,
        {"opened_or_used": ["未拆封"]},
    )

    assert filtered == "您已说明商品未拆封。退货原因是什么？"
    assert removed_fields == ["opened_or_used"]


def test_redundant_question_guard_handles_known_order_id_without_removing_other_questions() -> None:
    text = "我看到您提供了订单号。请再发一下订单号好吗？您要退哪一件商品？"

    filtered, removed_fields = filter_redundant_field_questions(
        text,
        {"order_id": ["SC-DEMO-4821"]},
    )

    assert filtered == "我看到您提供了订单号。您要退哪一件商品？"
    assert removed_fields == ["order_id"]


def test_redundant_question_guard_preserves_questions_about_a_different_field() -> None:
    text = "订单号对应的商品是什么？商品使用后有没有发红？"

    filtered, removed_fields = filter_redundant_field_questions(
        text,
        {
            "order_id": ["SC-DEMO-4821"],
            "opened_or_used": ["已使用"],
        },
    )

    assert filtered == text
    assert removed_fields == []


def test_redundant_question_guard_detects_known_quantity_question() -> None:
    filtered, removed_fields = filter_redundant_field_questions(
        "数量是多少？",
        {"quantity": ["2"]},
    )

    assert filtered == ""
    assert removed_fields == ["quantity"]


def test_redundant_question_guard_does_not_remove_factual_statements_about_known_fields() -> None:
    text = "退货原因是买错了，商品目前未拆封。"

    filtered, removed_fields = filter_redundant_field_questions(
        text,
        {
            "return_reason": ["买错了"],
            "opened_or_used": ["未拆封"],
        },
    )

    assert filtered == text
    assert removed_fields == []


def test_redundant_question_guard_removes_already_stated_refund_scope() -> None:
    filtered, removed_fields = filter_redundant_field_questions(
        "您希望整单退款，还是只退正装、保留小样？",
        {"return_scope": ["整单退款"]},
    )

    assert filtered == ""
    assert removed_fields == ["return_scope"]


def test_repeated_answer_guard_removes_prior_safety_advice_but_keeps_new_refund_answer() -> None:
    history = [
        {
            "role": "assistant",
            "content": "脸红发痒请停用产品，不要抓挠。如果症状加重请就医，并保留批次信息。",
        }
    ]
    response = (
        "我会按您已说明的情况继续处理退款申请。"
        "脸红发痒请停用产品，不要抓挠。如果症状加重请就医，并保留批次信息。"
    )

    filtered, removed = filter_repeated_answer_content(
        response,
        history,
        current_message="我已经说明过症状了，现在只想退款。",
    )

    assert filtered == "我会按您已说明的情况继续处理退款申请。"
    assert removed >= 1


def test_repeated_answer_guard_keeps_safety_advice_for_new_or_worsening_risk() -> None:
    history = [{"role": "assistant", "content": "先停用产品；如果红肿加重请及时就医。"}]
    response = "先停用产品并尽快就医。"

    filtered, removed = filter_repeated_answer_content(
        response,
        history,
        current_message="现在脸肿得更厉害了，还有呼吸困难，怎么办？",
    )

    assert filtered == response
    assert removed == 0


def test_new_safety_risk_keeps_safety_advice_without_restoring_duplicate_refund_answer() -> None:
    prior_refund_answer = "退款申请尚未提交，当前状态为待申请。"
    prior_safety = "先停用产品，如果症状加重请就医。"
    history = [{"role": "assistant", "content": f"{prior_refund_answer}{prior_safety}"}]
    response = "退款申请已经提交，当前状态为审核中。先停用产品，如果症状加重请就医。"

    filtered, removed = filter_repeated_answer_content(
        response,
        history,
        current_message="现在还有刺痛，退款到哪一步了？",
        case_facts={
            "facts": {
                "symptom": [{"value": "脸红", "source_ref": "old-message", "confirmation_status": "confirmed"}],
                "ai_response": [
                    {"value": f"{prior_refund_answer}{prior_safety}", "confirmation_status": "confirmed"}
                ],
            }
        },
    )

    assert filtered == response
    assert removed == 0


def test_new_safety_risk_does_not_disable_unrelated_answer_deduplication() -> None:
    prior_refund_answer = "退款需要按商品状态核验。"
    prior_safety = "先停用产品，如果症状加重请就医。"
    history = [{"role": "assistant", "content": f"{prior_refund_answer}{prior_safety}"}]
    response = f"退款状态为审核中。{prior_refund_answer}{prior_safety}"

    filtered, removed = filter_repeated_answer_content(
        response,
        history,
        current_message="现在还有刺痛，退款到哪一步了？",
        case_facts={
            "facts": {
                "symptom": [{"value": "脸红", "source_ref": "old-message", "confirmation_status": "confirmed"}],
                "ai_response": [
                    {"value": f"{prior_refund_answer}{prior_safety}", "confirmation_status": "confirmed"}
                ],
            }
        },
    )

    assert filtered == f"退款状态为审核中。{prior_safety}"
    assert removed == 1


def test_repeated_answer_guard_allows_customer_to_explicitly_request_repetition() -> None:
    history = [{"role": "assistant", "content": "退款需要按商品状态核验。"}]
    response = "退款需要按商品状态核验。"

    filtered, removed = filter_repeated_answer_content(
        response,
        history,
        current_message="麻烦把刚才退款流程再说一遍。",
    )

    assert filtered == response
    assert removed == 0


def test_repeated_advice_filter_keeps_new_refund_details_in_same_sentence() -> None:
    history = [
        {
            "role": "assistant",
            "content": "脸部发红后请停用产品，如果症状加重请就医。",
        }
    ]
    response = "请确认商品是否已拆封，退款申请尚未提交；此前提到的停用情况已记录。"

    filtered, removed = filter_repeated_answer_content(
        response,
        history,
        current_message="我只想了解退款申请到哪一步了。",
    )

    assert filtered == response
    assert removed == 0


def test_similar_wording_does_not_hide_changed_refund_status() -> None:
    history = [
        {
            "role": "assistant",
            "content": "退款申请尚未提交，当前状态为待处理。",
        }
    ]
    response = "退款申请已经提交，当前状态为待处理。"

    filtered, removed = filter_repeated_answer_content(
        response,
        history,
        current_message="退款现在到哪一步了？",
    )

    assert filtered == response
    assert removed == 0


def test_repeated_non_safety_answer_is_removed_when_current_turn_asks_new_question() -> None:
    history = [
        {"role": "assistant", "content": "退款申请需要按商品状态核验，当前还没有提交。"}
    ]
    response = "退款申请需要按商品状态核验，当前还没有提交。您这轮补充的信息我已经记录。"

    filtered, removed = filter_repeated_answer_content(
        response,
        history,
        current_message="赠品漏发的问题也请一起处理。",
    )

    assert filtered == "您这轮补充的信息我已经记录。"
    assert removed == 1


def test_same_known_symptom_does_not_reopen_safety_advice_when_asking_about_refund() -> None:
    history = [{"role": "assistant", "content": "脸部发红后请停用产品，如果症状加重请就医。"}]
    response = "关于退款，请按商品状态核验；脸部发红后请停用产品，如果症状加重请就医。"
    case_facts = {
        "facts": {
            "symptom": [{"value": "脸部发红", "confirmation_status": "confirmed"}],
            "ai_response": [
                {
                    "value": "脸部发红后请停用产品，如果症状加重请就医。",
                    "confirmation_status": "confirmed",
                }
            ],
        }
    }

    filtered, removed = filter_repeated_answer_content(
        response,
        [],
        current_message="我现在只问退款。脸还是发红。",
        case_facts=case_facts,
    )

    assert filtered == "关于退款，请按商品状态核验。"
    assert removed == 1


def test_safety_question_about_continued_use_keeps_the_answer() -> None:
    history = [{"role": "assistant", "content": "先停用产品。"}]
    response = "暂时不要继续使用，先停用产品并观察皮肤情况。"

    filtered, removed = filter_repeated_answer_content(
        response,
        history,
        current_message="脸还红，可以继续使用吗？",
    )

    assert filtered == response
    assert removed == 0


def test_symptom_extracted_from_current_message_is_not_treated_as_old_context() -> None:
    history = [{"role": "assistant", "content": "脸红后请停用产品。"}]
    response = "脸部刺痛是新增症状，请停用产品并尽快就医。"
    case_facts = {
        "facts": {
            "symptom": [
                {"value": "脸红", "source_ref": "old-message", "confirmation_status": "confirmed"},
                {"value": "脸部刺痛", "source_ref": "current-message", "confirmation_status": "unconfirmed"},
            ]
        }
    }

    filtered, removed = filter_repeated_answer_content(
        response,
        history,
        current_message="现在脸部有刺痛。",
        current_message_ref="current-message",
        case_facts=case_facts,
    )

    assert filtered == response
    assert removed == 0


def test_known_case_fields_are_not_asked_again_but_unanswered_fields_remain() -> None:
    response = "订单 SC-DEMO-4821 已确认。请再次提供订单号，并告诉我是否已经提交退款，以及商品是否拆封。"
    case_facts = {
        "facts": {
            "order_id": [{"value": "SC-DEMO-4821", "confirmation_status": "confirmed"}],
            "refund_status": [{"value": "未申请", "confirmation_status": "confirmed"}],
        }
    }

    filtered, removed = filter_repeated_answer_content(
        response,
        [],
        current_message="退款怎么处理？",
        case_facts=case_facts,
    )

    assert "请再次提供订单号" not in filtered
    assert "是否已经提交退款" not in filtered
    assert "商品是否拆封" in filtered
    assert removed == 2
