from intent.beauty_intent_registry import BEAUTY_INTENT_BY_ID
from intent.beauty_intent_ticket_registry import BEAUTY_INTENT_TICKET_BY_ID
from intent.intent_recognizer import (
    CLASSIFY_INTENT_TOOL,
    IntentCategory,
    _LLM_SYSTEM_PROMPT,
    _normalize_llm_source,
    match_patterns,
    recognize_intent,
)


def test_classifier_tool_exposes_only_registered_beauty_intents_and_control_values() -> None:
    allowed = set(BEAUTY_INTENT_BY_ID) | {
        IntentCategory.HUMAN_HANDOFF.value,
        IntentCategory.OTHER.value,
    }

    assert set(CLASSIFY_INTENT_TOOL["input_schema"]["properties"]["intent"]["enum"]) == allowed


def test_exchange_only_with_explicit_refund_negation_does_not_match_refund_or_wrong_item() -> None:
    result = match_patterns("我不要退款，只换货")

    assert result["intent"] is IntentCategory.OTHER
    assert IntentCategory.RETURN_REFUND.value not in result["scores"]
    assert IntentCategory.WRONG_ITEM.value not in result["scores"]
    assert "只换货" in _LLM_SYSTEM_PROMPT


def test_exchange_remains_supported_when_the_customer_reports_a_wrong_item() -> None:
    result = match_patterns("收到的色号发错了，我想换货")

    assert result["intent"] is IntentCategory.WRONG_ITEM


def test_unregistered_legacy_llm_intent_is_rejected_at_the_boundary() -> None:
    result = _normalize_llm_source(
        {"intent": "technical", "confidence": 0.95, "reasoning": "legacy result", "success": True}
    )

    assert result.intent is IntentCategory.OTHER
    assert result.success is False
    assert "未注册" in result.reasoning


def test_normal_logistics_lookup_and_stalled_logistics_use_distinct_intents() -> None:
    normal = match_patterns("我的订单号 SC-DEMO-4821，物流到哪里了？")
    stalled = match_patterns("物流三天没更新了，帮我催查")

    assert normal["intent"] is IntentCategory.ORDER_STATUS_BEAUTY
    assert stalled["intent"] is IntentCategory.LOGISTICS_STALLED
    assert "lookup_logistics" in BEAUTY_INTENT_TICKET_BY_ID["order_status"]["allowed_tools"]


def test_legacy_http_500_override_cannot_escape_beauty_intent_registry(monkeypatch) -> None:
    import asyncio
    import intent.intent_recognizer as recognizer

    async def llm(_message, _history=None):
        return {
            "intent": "product_info",
            "confidence": 0.9,
            "reasoning": "商品信息咨询",
            "success": True,
        }

    async def embedding(_message):
        return {"intent": "product_info", "confidence": 0.9, "success": True}

    monkeypatch.setattr(recognizer, "_recognize_with_llm", llm)
    monkeypatch.setattr(recognizer, "_recognize_with_embedding", embedding)

    result = asyncio.run(recognize_intent("系统提示错误代码 500"))

    assert result.intent_diagnostics.get("fast_path") is not True
    assert result.intent.value in set(BEAUTY_INTENT_BY_ID) | {
        IntentCategory.HUMAN_HANDOFF.value,
        IntentCategory.OTHER.value,
    }
    assert result.intent is not IntentCategory.TECHNICAL_CRASH


def test_fast_path_still_splits_cross_domain_compound_intents() -> None:
    import asyncio

    result = asyncio.run(
        recognize_intent("订单里漏发了两个小样，同时我要申请退款")
    )

    assert result.intent is IntentCategory.MISSING_ITEM
    assert result.secondary_intents == [IntentCategory.RETURN_REFUND]
    assert {item["domain"] for item in result.sub_requests} == {"order", "after_sales"}
    assert result.intent_diagnostics["scoped_retrieval_queries"] == {
        "order": "订单里漏发了两个小样",
        "after_sales": "我要申请退款",
    }
