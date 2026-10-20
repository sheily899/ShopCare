from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agents.agent_orchestrator import (
    SEARCH_KNOWLEDGE_BASE_TOOL,
    AgentType,
    Request,
    _policy_scoped_tools,
)
from agents.tools import beauty_tools
from intent.beauty_intent_ticket_registry import (
    BEAUTY_INTENT_TICKET_BY_ID,
    policy_for_intents,
    should_prefetch_knowledge,
)
from intent.intent_recognizer import IntentCategory, detect_emotion, extract_entities


def test_beauty_tool_allowlist_has_no_legacy_tools() -> None:
    names = set(beauty_tools())
    assert {
        "inspect_request_context",
        "suggest_required_fields",
        "lookup_order",
        "lookup_logistics",
        "create_ticket_draft",
        "check_refund_status",
        "calculate_price_difference",
        "record_risk_event",
    } <= names
    assert not {"lookup_error_code", "build_diagnostic_plan", "compare_amounts", "check_billing_fields"} & names


def test_ticket_registry_maps_ticket_boundary_and_required_fields() -> None:
    adverse = BEAUTY_INTENT_TICKET_BY_ID["adverse_reaction"]
    consult = BEAUTY_INTENT_TICKET_BY_ID["ingredient_fit"]
    assert adverse["needs_ticket"] is True
    assert {"order_id", "symptom"} <= set(adverse["blocking_fields"])
    assert {"symptom_area", "stopped_use", "medical_visit"} <= set(adverse["ticket_fields"])
    assert consult["needs_ticket"] is False


def test_all_intents_have_explicit_execution_policy() -> None:
    required = {
        "knowledge_mode",
        "allowed_tools",
        "blocking_fields",
        "ticket_fields",
        "missing_fields_action",
        "departments",
    }
    assert len(BEAUTY_INTENT_TICKET_BY_ID) == 19
    assert all(required <= contract.keys() for contract in BEAUTY_INTENT_TICKET_BY_ID.values())
    assert {contract["knowledge_mode"] for contract in BEAUTY_INTENT_TICKET_BY_ID.values()} <= {
        "required", "when_relevant", "never"
    }
    assert BEAUTY_INTENT_TICKET_BY_ID["order_status"]["knowledge_mode"] == "never"
    assert "search_knowledge_base" in BEAUTY_INTENT_TICKET_BY_ID["ingredient_fit"]["allowed_tools"]
    assert "search_knowledge_base" not in BEAUTY_INTENT_TICKET_BY_ID["order_status"]["allowed_tools"]


def test_runtime_allowed_tools_match_the_registry_for_all_19_intents() -> None:
    agent_types = {
        "consult": AgentType.INQUIRY,
        "order": AgentType.FULFILLMENT,
        "after_sales": AgentType.REFUND_RISK,
    }
    all_tools = [SEARCH_KNOWLEDGE_BASE_TOOL, *beauty_tools().values()]

    for intent_id, contract in BEAUTY_INTENT_TICKET_BY_ID.items():
        agent_type = agent_types[contract["agent"]]
        knowledge_prefetched = contract["knowledge_mode"] == "required"
        scoped = _policy_scoped_tools(
            SimpleNamespace(tools=all_tools),
            Request(message="测试", intent=IntentCategory(intent_id)),
            agent_type,
            searcher=None,
            knowledge_prefetched=knowledge_prefetched,
        )
        expected = set(contract["allowed_tools"])
        if knowledge_prefetched:
            # 必需检索已在 Agent 之前完成，检索器不会再次暴露给 Agent。
            expected.discard("search_knowledge_base")

        assert {tool.name for tool in scoped or []} == expected, intent_id


def test_order_and_after_sales_skills_use_the_same_handoff_boundary() -> None:
    skill_root = Path(__file__).resolve().parents[1] / "skills"
    order_skill = (skill_root / "order" / "SKILL.md").read_text(encoding="utf-8")
    after_sales_skill = (skill_root / "after_sales" / "SKILL.md").read_text(encoding="utf-8")

    assert "缺少阻塞字段时只追问" in order_skill
    assert "普通红肿加退款继续由 AI 处理" in after_sales_skill
    assert "只有统一运行时转人工规则命中" in after_sales_skill


def test_knowledge_prefetch_is_explicit_and_optional_retrieval_remains_available_to_agent() -> None:
    assert should_prefetch_knowledge("ingredient_fit", "澄润屏障修护精华水有哪些成分") is True
    assert should_prefetch_knowledge("order_status", "订单现在到哪里了") is False
    assert should_prefetch_knowledge("missing_item", "赠品漏发了，按什么规则处理") is True
    assert should_prefetch_knowledge("missing_item", "订单里漏发了一瓶粉底") is False
    assert "search_knowledge_base" in BEAUTY_INTENT_TICKET_BY_ID["missing_item"]["allowed_tools"]


def test_compound_policy_unions_only_registered_actions_and_departments() -> None:
    combined = policy_for_intents(["adverse_reaction", "return_refund", "missing_item"])
    assert combined["needs_ticket"] is True
    assert {"lookup_order", "search_knowledge_base", "create_ticket_draft", "record_risk_event"} <= set(combined["allowed_tools"])
    assert set(combined["departments"]) == {"质量", "售后", "财务", "仓储"}
    assert combined["missing_fields_action"] == "clarify"


def test_missing_fields_never_trigger_automatic_handoff_for_any_intent() -> None:
    """字段未收集齐时应追问或留待补充，不能把字段缺失当作转人工条件。"""
    assert all(
        contract["missing_fields_action"] == "clarify"
        for contract in BEAUTY_INTENT_TICKET_BY_ID.values()
    )


def test_beauty_entity_and_emotion_contract() -> None:
    entities = extract_entities("订单号 A1234，用粉底液后全脸发红，已停用，数量2")
    assert entities["order_id"] == ["A1234"]
    assert "发红" in entities["symptom"]
    assert entities["stopped_use"] == ["已停用"]
    emotion, intensity, evidence = detect_emotion("我很着急，脸红了怎么办")
    assert emotion == "anxious"
    assert intensity > 0
    assert evidence


def test_create_ticket_draft_schema_requires_intent_and_is_not_formal_submit() -> None:
    spec = beauty_tools()["create_ticket_draft"]
    assert spec.input_schema["required"] == ["intent"]
    assert "draft" in spec.name or "ticket" in spec.name
