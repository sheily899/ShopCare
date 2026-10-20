from __future__ import annotations

from agents.case_facts import CaseFacts


def test_case_facts_preserve_provenance_for_message_slots_and_tool_results() -> None:
    facts = CaseFacts.from_entities(
        {"order_id": ["SC-DEMO-4821"], "missing_item": ["小样"]},
        source_ref="message-123",
        confidence=0.94,
    )
    facts.record(
        "symptom",
        "发红",
        source_type="message",
        source_ref="message-124",
        confidence=0.97,
        confirmation_status="confirmed",
    )
    facts.record_tool_result(
        "lookup_order",
        {
            "success": True,
            "available": True,
            "order_id": "SC-DEMO-4821",
            "order": {"product_name": "轻透粉底液", "status": "已签收"},
        },
        source_ref="trace-456",
    )

    payload = facts.to_dict()
    order_fact = payload["facts"]["order_id"][-1]
    assert order_fact == {
        "value": "SC-DEMO-4821",
        "source_type": "tool:lookup_order",
        "source_ref": "trace-456",
        "confidence": 1.0,
        "confirmation_status": "confirmed",
    }
    assert facts.values("product_name") == ["轻透粉底液"]
    assert facts.values("order_status") == ["已签收"]
    assert facts.latest("symptom")["source_ref"] == "message-124"


def test_case_facts_round_trip_and_merge_without_losing_prior_sources() -> None:
    original = CaseFacts.from_entities({"order_id": ["A12345"]}, source_ref="m-1")
    saved = CaseFacts.from_dict(original.to_dict())
    saved.record("tracking_no", "YT1234567890", source_type="tool:lookup_order", source_ref="t-1")
    merged = CaseFacts.from_dict(original.to_dict())
    merged.merge(saved)

    assert merged.values("order_id") == ["A12345"]
    assert merged.values("tracking_no") == ["YT1234567890"]
    assert merged.to_dict()["schema_version"] == 1


def test_unavailable_lookup_does_not_turn_the_requested_order_id_into_a_verified_fact() -> None:
    facts = CaseFacts.from_entities({"order_id": ["SC-UNKNOWN-1"]}, source_ref="message-2")
    facts.record_tool_result(
        "lookup_order",
        {"success": True, "available": False, "order_id": "SC-UNKNOWN-1"},
        source_ref="trace-2",
    )

    order_records = facts.to_dict()["facts"]["order_id"]
    assert len(order_records) == 1
    assert order_records[0]["source_type"] == "message"
    assert facts.values("lookup_order_available") == [False]


def test_explicit_field_clear_invalidates_old_values_but_preserves_audit_provenance() -> None:
    facts = CaseFacts.from_entities({"order_id": ["SC-DEMO-4821"]}, source_ref="m-1")
    facts.clear("order_id", source_type="agent_edit", source_ref="agent-1")

    assert facts.values("order_id") == []
    assert facts.latest("order_id") == {
        "value": None,
        "source_type": "agent_edit",
        "source_ref": "agent-1",
        "confidence": 1.0,
        "confirmation_status": "cleared",
    }
    assert CaseFacts.from_dict(facts.to_dict()).values("order_id") == []


def test_current_message_entities_override_older_case_values_for_answer_context() -> None:
    facts = CaseFacts.from_entities(
        {"opened_or_used": ["未拆封"]}, source_ref="message-old"
    )
    current = facts.effective_values({"opened_or_used": ["已拆封未使用"]})

    assert current["opened_or_used"] == ["已拆封未使用"]


def test_empty_current_extraction_keeps_the_last_known_case_value() -> None:
    facts = CaseFacts.from_entities(
        {"opened_or_used": ["未拆封"]}, source_ref="message-old"
    )

    current = facts.effective_values({"opened_or_used": []})

    assert current["opened_or_used"] == ["未拆封"]


def test_explicit_clear_resets_prior_values_for_multi_value_facts() -> None:
    facts = CaseFacts.from_entities({"symptom": ["发红"]}, source_ref="message-1")
    facts.clear("symptom", source_type="agent_edit", source_ref="agent-1")

    assert facts.values("symptom") == []

    facts.record(
        "symptom",
        "刺痛",
        source_type="message",
        source_ref="message-2",
    )
    assert facts.values("symptom") == ["刺痛"]
