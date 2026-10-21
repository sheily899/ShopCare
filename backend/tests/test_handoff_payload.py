from agents.handoff import build_handoff_payload, emotion_band
from intent.intent_recognizer import IntentCategory, UrgencyLevel
from agents.agent_orchestrator import (
    AgentOrchestrator,
    AgentResponse,
    AgentType,
    Request,
    ResponseComposer,
    _customer_safe_text,
)
from agents.case_facts import CaseFacts
from agents.customer_tone import customer_friendly_response


async def _knowledge_search(query: str, top_k: int, **_kwargs):
    assert top_k == 5
    return [{"title": "测试规则", "content": f"关于“{query}”的已确认规则。", "score": 1.0}]


def test_emotion_uses_five_registered_customer_service_bands() -> None:
    assert emotion_band("neutral", 0.1) == "calm"
    assert emotion_band("disappointed", 0.4) == "upset"
    assert emotion_band("anxious", 0.8) == "anxious"
    assert emotion_band("angry", 0.4) == "angry"
    assert emotion_band("threatening", 0.1) == "threatening"


def test_handoff_emotion_and_evidence_prefer_shared_case_facts() -> None:
    facts = CaseFacts()
    facts.record("emotion", "威胁升级", source_type="emotion_llm", source_ref="req-2")
    facts.record("emotion_intensity", 0.95, source_type="emotion_llm", source_ref="req-2")
    facts.record("emotion_evidence", "明天再没结果我就打12315", source_type="emotion_llm", source_ref="req-2")

    payload = build_handoff_payload(
        message="明天再没结果我就打12315",
        intent=IntentCategory.RETURN_REFUND,
        entities={},
        case_facts=facts,
        emotion="neutral",
        should_handoff=True,
    )

    assert payload["summary"]["emotion"] == "threatening"
    assert payload["summary"]["emotion_source"] == "emotion_llm"
    assert payload["summary"]["emotion_evidence"] == ["明天再没结果我就打12315"]


def test_no_handoff_does_not_generate_summary_or_ticket() -> None:
    payload = build_handoff_payload(
        message="想查一下色号", intent=IntentCategory.SHADE_MATCH,
        entities={}, should_handoff=False,
    )
    assert payload["handoff_required"] is False
    assert payload["summary"] is None
    assert payload["ticket_draft"] is None


def test_unregistered_intent_without_handoff_does_not_require_ticket_contract() -> None:
    payload = build_handoff_payload(
        message="暂时无法识别的问题", intent=IntentCategory.OTHER,
        entities={}, should_handoff=False,
    )

    assert payload["handoff_required"] is False
    assert payload["summary"] is None
    assert payload["ticket_draft"] is None


def test_ticketable_intent_creates_review_draft_without_handoff() -> None:
    payload = build_handoff_payload(
        message="我想申请退款",
        intent=IntentCategory.RETURN_REFUND,
        entities={"order_id": ["SC-DEMO-4821"]},
        should_handoff=False,
    )

    assert payload["handoff_required"] is False
    assert payload["summary"] is None
    assert payload["suggested_replies"] == []
    assert payload["ticket_draft"]["status"] == "draft"
    assert payload["ticket_draft"]["needs_review"] is True
    assert payload["ticket_draft"]["fields"]["order_id"] == "SC-DEMO-4821"


def test_handoff_generates_summary_ticket_and_replies_for_adverse_reaction() -> None:
    payload = build_handoff_payload(
        message="用了粉底后脸很红，我很着急",
        intent=IntentCategory.ADVERSE_REACTION,
        entities={"symptom": ["发红"], "stopped_use": ["已停用"]},
        emotion="anxious", emotion_intensity=0.9,
        urgency=UrgencyLevel.CRITICAL, should_handoff=True,
    )
    assert payload["handoff_required"] is True
    assert payload["summary"]["emotion"] == "anxious"
    assert payload["ticket_draft"]["status"] == "draft"
    assert payload["ticket_draft"]["needs_review"] is True
    assert payload["suggested_replies"]
    assert "customer_response" not in payload


def test_customer_output_hides_internal_boundary_language() -> None:
    request = Request(message="用了粉底后发红", intent=IntentCategory.ADVERSE_REACTION)
    text = _customer_safe_text("不会承诺退款，也不判断责任。", request)
    assert text == "不会承诺退款，也不判断责任。"


def test_customer_tone_does_not_rewrite_the_agent_answer() -> None:
    text = customer_friendly_response(
        "请先暂停使用相关产品，客服会继续跟进。",
        intent=IntentCategory.ADVERSE_REACTION,
        emotion="anxious",
    )
    assert text == "请先暂停使用相关产品，客服会继续跟进。"
    calm = customer_friendly_response("订单正在核对。", intent=IntentCategory.ORDER_STATUS, emotion="neutral")
    assert calm == "订单正在核对。"


def test_compound_handoff_uses_one_ticket_and_deduplicated_department_subtasks() -> None:
    facts = CaseFacts.from_entities(
        {"order_id": ["SC-DEMO-4821"], "symptom": ["脸部发红"]},
        source_ref="message-1",
    )
    facts.record_tool_result(
        "lookup_order",
        {"success": True, "available": True, "order": {"product_name": "轻透粉底液", "order_status": "已签收"}},
        source_ref="trace-1",
    )
    payload = build_handoff_payload(
        message="使用粉底后发红，申请退款，赠品小样没收到。",
        intent=IntentCategory.ADVERSE_REACTION,
        secondary_intents=[IntentCategory.RETURN_REFUND, IntentCategory.MISSING_ITEM],
        entities={},
        case_facts=facts,
        emotion="anxious",
        urgency=UrgencyLevel.CRITICAL,
        should_handoff=True,
        conv_id="conv-1",
        buyer_nickname="buyer-1",
        ai_response="为您转接人工，请您稍等。",
    )

    ticket = payload["ticket_draft"]
    assert ticket["fields"]["order_id"] == "SC-DEMO-4821"
    assert ticket["fields"]["product_name"] == "轻透粉底液"
    assert "shop" not in ticket["system_fields"]
    assert "completed_at" not in ticket["system_fields"]
    assert {item["department"] for item in ticket["sub_tasks"]} == {"质量", "售后", "财务", "仓储"}
    assert len(ticket["sub_tasks"]) == 4
    assert payload["summary"]["request_items"] == [
        "商品使用不适", "退货退款", "商品或赠品漏发", "赠品/小样未收到",
    ]
    assert "已转接人工" not in " ".join(payload["suggested_replies"])
    assert payload["summary"]["commitments"] == []


def test_orchestrator_does_not_replace_ai_response_with_handoff_template() -> None:
    class StubAgent:
        async def handle(self, request):
            facts = CaseFacts.from_dict(request.case_facts)
            facts.record("symptom", "脸部发红", source_type="message", source_ref="m-1")
            return AgentResponse(
                agent_type=AgentType.REFUND_RISK,
                content="请先暂停使用产品；如果红肿加重或呼吸不适，请尽快就医。",
                success=True,
                escalate=True,
                case_facts=facts.to_dict(),
            )

    async def run():
        orchestrator = AgentOrchestrator(
            agent_pool={AgentType.REFUND_RISK: StubAgent()},
            composer=ResponseComposer(),
            knowledge_search=_knowledge_search,
        )
        return await orchestrator.run(
            Request(
                message="用了粉底后脸发红",
                user_id="consumer-demo",
                conv_id="conv-risk",
                intent=IntentCategory.ADVERSE_REACTION,
                urgency=UrgencyLevel.CRITICAL,
            )
        )

    result = __import__("asyncio").run(run())
    assert result.response == "请先暂停使用产品；如果红肿加重或呼吸不适，请尽快就医。"
    assert result.handoff_summary is not None
    assert "customer_response" not in result.handoff_summary


def test_adverse_handoff_does_not_assume_refund_or_missing_gift() -> None:
    payload = build_handoff_payload(
        message="用了精华后脸发红，已经停用，请转人工处理。",
        intent=IntentCategory.ADVERSE_REACTION,
        entities={"symptom": ["发红"], "stopped_use": ["已停用"]},
        should_handoff=True,
    )

    steps = payload["summary"]["next_steps"]
    request_items = payload["summary"]["request_items"]
    assert not any("退款" in step for step in steps)
    assert not any("赠品" in item or "小样" in item for item in request_items)
    assert not any("赠品" in step or "小样" in step for step in steps)


def test_handoff_does_not_mark_a_received_gift_as_missing() -> None:
    payload = build_handoff_payload(
        message="用了精华后脸发红，直播赠送的小样我已经收到了，请转人工。",
        intent=IntentCategory.ADVERSE_REACTION,
        entities={"symptom": ["发红"]},
        should_handoff=True,
    )

    assert not any("赠品" in item or "小样" in item for item in payload["summary"]["request_items"])
    assert not any("赠品" in step or "小样" in step for step in payload["summary"]["next_steps"])


def test_guarded_customer_answer_is_the_only_source_of_handoff_commitments() -> None:
    class UnsafePromiseAgent:
        async def handle(self, request):
            return AgentResponse(
                agent_type=AgentType.REFUND_RISK,
                content="我保证一定会退款，我会帮您处理。",
                success=True,
                escalate=True,
                case_facts=CaseFacts.from_dict(request.case_facts).to_dict(),
            )

    async def run():
        orchestrator = AgentOrchestrator(
            agent_pool={AgentType.REFUND_RISK: UnsafePromiseAgent()},
            composer=ResponseComposer(),
            knowledge_search=_knowledge_search,
        )
        return await orchestrator.run(
            Request(
                message="我想申请退款，请转人工。",
                user_id="consumer-demo",
                conv_id="conv-unsafe-promise",
                intent=IntentCategory.RETURN_REFUND,
            )
        )

    result = __import__("asyncio").run(run())

    assert result.response == "这项信息目前还无法确认，我会先按已核实的信息继续处理。"
    assert result.handoff_summary["commitments"] == []
