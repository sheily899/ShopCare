"""转人工客服端的事件游标、快照和转人工后增量消息。"""

from __future__ import annotations

import json

from storage.sqlite_case_store import SQLiteCaseStore


def _start_handoff(store: SQLiteCaseStore) -> dict:
    store.ensure_session("consumer-demo", "events-case")
    return store.start_handoff(
        "events-case",
        user_id="consumer-demo",
        message="请转人工",
        idempotency_key="handoff:events-case:one",
        reason="consumer_requested_human",
        temporary_summary={"text": "顾客直接要求转人工，暂无其他诉求", "sources": []},
        temporary_priority="P3",
        temporary_priority_score=0,
        temporary_priority_reasons=["无额外规则命中"],
        temporary_priority_rule_version="urgency-test-v1",
        online_ack="已为您转接人工，请您稍等。",
        unknown_presence_ack="已为您登记，专员上线后会联系您。",
        callback_due_at=None,
    )


def test_agent_conversation_snapshot_contains_three_independent_blocks_and_resume_cursor(tmp_path) -> None:
    store = SQLiteCaseStore(tmp_path / "handoff-snapshot.sqlite3")
    handoff = _start_handoff(store)
    store.save_handoff_generated_block(
        handoff["handoff_id"], block_type="summary", status="degraded", state_hash="hash-1",
        payload={"text": "顾客直接要求转人工，暂无其他诉求。", "statements": []}, latency_ms=2.5,
        source_message_seq=handoff["source_message_seq"],
    )

    conversation = store.get_conversation("events-case", viewer_role="agent")
    blocks = conversation["handoff"]["blocks"]

    assert conversation["handoff"]["handoff_id"] == handoff["handoff_id"]
    assert conversation["handoff"]["source_message_seq"] == handoff["source_message_seq"]
    assert conversation["handoff"]["last_event_id"] > handoff["event_id"]
    assert blocks["summary"]["status"] == "degraded"
    assert blocks["summary"]["payload"]["text"]
    assert blocks["emotion_urgency"]["status"] == "pending"
    assert blocks["next_steps"]["status"] == "pending"
    assert conversation["messages"][0]["message_seq"] == handoff["source_message_seq"]


def test_post_handoff_consumer_message_is_a_resumable_incremental_event_without_body(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HANDOFF", "true")
    store = SQLiteCaseStore(tmp_path / "handoff-events.sqlite3")
    handoff = _start_handoff(store)
    initial_messages = store.get_conversation("events-case", viewer_role="agent")["messages"]
    initial_message_seq = max(item["message_seq"] for item in initial_messages)

    store.append_message("events-case", "consumer", "consumer-demo", "我还想补充目前有些刺痛。")
    events = store.list_handoff_events("events-case", after_event_id=handoff["event_id"])
    incremental_messages = store.get_conversation(
        "events-case", viewer_role="agent", after_message_seq=initial_message_seq
    )["messages"]

    assert len(events) == 1
    assert events[0]["event_type"] == "handoff.message.incremental"
    assert events[0]["event_id"] > handoff["event_id"]
    assert events[0]["data"]["message_id"]
    assert events[0]["data"]["message_seq"] > initial_message_seq
    assert "刺痛" not in json.dumps(events[0]["data"], ensure_ascii=False)
    assert store.list_handoff_events("events-case", after_event_id=events[0]["event_id"]) == []
    assert len(incremental_messages) == 1
    assert incremental_messages[0]["message_seq"] == events[0]["data"]["message_seq"]
    assert incremental_messages[0]["content"] == "我还想补充目前有些刺痛。"


def test_non_consumer_message_does_not_emit_post_handoff_incremental_event(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HANDOFF", "true")
    store = SQLiteCaseStore(tmp_path / "handoff-events-agent.sqlite3")
    handoff = _start_handoff(store)

    store.append_message("events-case", "customer_service", "agent-demo", "我来帮您核对。")

    assert store.list_handoff_events("events-case", after_event_id=handoff["event_id"]) == []
