"""验证 Agent 对话历史的重复答复抑制与增量回复边界。"""

from agents.agent_orchestrator import BillingAgent, Request, ResponseComposer, AgentResponse, AgentType
from agents.response_guard import filter_repeated_answer_content
from intent.intent_recognizer import IntentCategory


def test_after_sales_system_prompt_makes_safety_reminders_conditional() -> None:
    prompt = BillingAgent()._system_prompt()

    assert "此前已说明且风险未变时，不重复整段提醒" in prompt
    assert "只回答消费者本轮新增或尚未解决的问题" in prompt
    assert "必须把症状和安全动作放在最前面并明确强调" not in prompt


def test_saved_case_facts_prevent_old_safety_advice_from_reappearing() -> None:
    prior_safety = "先停用产品，不要抓挠；如果症状加重请就医，并保留批次信息。"
    new_refund_answer = "您已说明商品已使用且小样未拆封，退款仍需按商品状态核验。"

    filtered, removed = filter_repeated_answer_content(
        f"{new_refund_answer}{prior_safety}",
        [],
        current_message="我前面已经说过症状了，现在只想知道退款怎么处理。",
        case_facts={
            "facts": {
                "ai_response": [
                    {
                        "value": prior_safety,
                        "confirmation_status": "confirmed",
                    }
                ]
            }
        },
    )

    assert filtered == new_refund_answer
    assert removed >= 1


def test_composer_prompt_includes_recent_dialogue_as_already_answered_context() -> None:
    request = Request(
        message="我现在只想知道退款下一步。",
        context="assistant: 先停用产品，症状加重请就医。",
    )
    response = AgentResponse(
        agent_type=AgentType.REFUND_RISK,
        content="请按商品状态核验退款申请。",
        success=True,
    )

    prompt = ResponseComposer()._build_prompt(request, response, [])

    assert "最近对话（用于识别已答内容，不要复述）" in prompt
    assert "先停用产品，症状加重请就医" in prompt


def test_composer_prompt_uses_registered_missing_fields_as_followup_boundary() -> None:
    request = Request(
        message="我已停用，想申请退款。",
        intent=IntentCategory.ADVERSE_REACTION,
        secondary_intents=[IntentCategory.RETURN_REFUND],
        entities={"stopped_use": ["已停用"]},
        case_facts={
            "facts": {
                "order_id": [{"value": "SC-DEMO-4821", "confirmation_status": "confirmed"}],
                "symptom": [{"value": "脸部红肿", "confirmation_status": "confirmed"}],
                "opened_or_used": [{"value": "已使用", "confirmation_status": "confirmed"}],
            }
        },
    )
    response = AgentResponse(
        agent_type=AgentType.REFUND_RISK,
        content="请告知退款原因。",
        success=True,
    )

    prompt = ResponseComposer()._build_prompt(request, response, [])

    assert "missing_text_fields" in prompt
    assert '"return_reason"' in prompt
    assert "symptom_area" not in prompt
    assert "枚举字段由页面选项收集" in prompt
