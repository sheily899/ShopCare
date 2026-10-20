"""客服手动刷新交接会话时，增量事实、工单草稿和交接输入保持一致。"""

from __future__ import annotations

from handoff.processor import _refresh_ticket_draft
from storage.sqlite_case_store import SQLiteCaseStore


def _start_handoff(store: SQLiteCaseStore) -> dict:
    store.ensure_session("consumer-demo", "refresh-case")
    return store.start_handoff(
        "refresh-case",
        user_id="consumer-demo",
        message="请转人工",
        idempotency_key="handoff:refresh-case:one",
        reason="consumer_requested_human",
        temporary_summary={"text": "顾客直接要求人工客服接手", "sources": []},
        temporary_priority="P3",
        temporary_priority_score=0,
        temporary_priority_reasons=["无额外规则命中"],
        temporary_priority_rule_version="test-v1",
        online_ack="已为您转接人工，请您稍等。",
        unknown_presence_ack="已为您登记，专员上线后会联系您。",
        callback_due_at=None,
    )


def test_refresh_cursor_and_ticket_draft_use_new_consumer_facts_without_overwrite(tmp_path, monkeypatch):
    monkeypatch.setenv("HANDOFF", "true")
    store = SQLiteCaseStore(tmp_path / "handoff-refresh.sqlite3")
    handoff = _start_handoff(store)
    store.update_case_state("refresh-case", {
        "ticket_draft": {
            "status": "draft", "needs_review": True, "intent": "return_refund",
            "fields": {"return_reason": "客服已核实的原因"},
            "blocking_fields": ["order_id", "return_reason", "opened_or_used"],
            "system_fields": {"ticket_id": "T-keep-existing"},
        },
    })
    store.append_message(
        "refresh-case", "consumer", "consumer-demo",
        "我想申请退款，订单 SC-DEMO-4821，买错了，商品还没拆封。",
    )

    pending = store.get_handoff_refresh_input(handoff["handoff_id"])
    assert len(pending["messages"]) == 1
    message = pending["messages"][0]
    assert message["content"].startswith("我想申请退款")
    assert message["history"][-1]["content"] == "已为您登记，专员上线后会联系您。"

    stored = store.save_handoff_intent_enrichment(
        handoff["handoff_id"],
        source_message_seq=message["message_seq"],
        source_message_id=message["message_id"],
        intent_id="return_refund",
        confidence=0.98,
        intent_group="after_sales",
        secondary_intents=[],
        entities={
            "order_id": ["SC-DEMO-4821"],
            "return_reason": ["买错了"],
            "opened_or_used": ["未拆封"],
        },
        sub_requests=[{"id": "return_refund:refresh", "intent": "return_refund"}],
        advance_refresh_cursor=True,
    )
    assert stored["status"] == "ready"
    assert store.get_handoff_refresh_input(handoff["handoff_id"])["messages"] == []

    assert _refresh_ticket_draft(store, "refresh-case", message["content"]) is True
    conversation = store.get_conversation("refresh-case", viewer_role="agent")
    ticket = conversation["case_state"]["ticket_draft"]
    assert ticket["status"] == "draft"
    assert ticket["needs_review"] is True
    assert ticket["fields"]["order_id"] == "SC-DEMO-4821"
    assert ticket["fields"]["return_reason"] == "客服已核实的原因"
    assert ticket["fields"]["opened_or_used"] == "未拆封"
    assert ticket["system_fields"]["ticket_id"] == "T-keep-existing"

    assert _refresh_ticket_draft(store, "refresh-case", message["content"]) is False
    assert store.get_conversation("refresh-case", viewer_role="agent")["case_state"]["ticket_draft"] == ticket

    generation_input = store.get_handoff_generation_input(handoff["handoff_id"])
    assert generation_input["conversation_messages"][-1]["content"] == message["content"]
    assert generation_input["ticket"]["fields"]["order_id"] == "SC-DEMO-4821"


def test_refresh_retries_ticket_merge_after_concurrent_agent_edit(tmp_path, monkeypatch):
    monkeypatch.setenv("HANDOFF", "true")
    store = SQLiteCaseStore(tmp_path / "handoff-refresh-race.sqlite3")
    _start_handoff(store)
    store.update_case_state("refresh-case", {
        "ticket_draft": {
            "status": "draft", "intent": "return_refund", "needs_review": True,
            "fields": {"order_id": "SC-DEMO-4821"},
            "blocking_fields": ["order_id", "return_reason", "opened_or_used"],
            "system_fields": {"ticket_id": "T-race"},
        },
    })
    store.update_case_state("refresh-case", {
        "case_facts": {"schema_version": 1, "facts": {
            "return_reason": [{"value": "买错了", "source_type": "message", "source_ref": "msg-1", "confidence": 0.99, "confirmation_status": "unconfirmed"}],
            "opened_or_used": [{"value": "未拆封", "source_type": "message", "source_ref": "msg-1", "confidence": 0.99, "confirmation_status": "unconfirmed"}],
        }},
        "handoff_intent_enrichment": {"intent": "return_refund"},
    })

    original_update = store.update_case_state
    first_call = True

    def concurrent_edit(conv_id, updates, expected_version=None):
        nonlocal first_call
        if first_call:
            first_call = False
            latest = store.get_conversation(conv_id, viewer_role="agent")["case_state"]["ticket_draft"]
            latest["fields"]["return_reason"] = "人工已确认：买错了"
            original_update(conv_id, {"ticket_draft": latest})
        return original_update(conv_id, updates, expected_version)

    monkeypatch.setattr(store, "update_case_state", concurrent_edit)
    assert _refresh_ticket_draft(store, "refresh-case", "订单 SC-DEMO-4821 申请退款") is True
    ticket = store.get_conversation("refresh-case", viewer_role="agent")["case_state"]["ticket_draft"]
    assert ticket["fields"]["return_reason"] == "人工已确认：买错了"
    assert ticket["fields"]["opened_or_used"] == "未拆封"
