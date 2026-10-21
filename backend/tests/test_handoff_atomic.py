"""原子转人工的落库、幂等、在线状态与回滚边界。"""

from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from storage.sqlite_case_store import SQLiteCaseStore


def _commit(
    store: SQLiteCaseStore,
    *,
    idempotency_key: str = "handoff:request-1",
    conv_id: str = "case-atomic",
) -> dict:
    return store.start_handoff(
        conv_id,
        user_id="consumer-demo",
        message="请帮我转人工",
        idempotency_key=idempotency_key,
        reason="consumer_requested_human",
        temporary_summary={"text": "顾客直接要求转人工，暂无其他诉求", "sources": [1]},
        temporary_priority="P3",
        temporary_priority_score=0,
        temporary_priority_reasons=["转人工本身不加分"],
        temporary_priority_rule_version="handoff-urgency-v2",
        online_ack="为您转接人工，请您稍等。",
        unknown_presence_ack="为您转接人工，请您稍等。已为您登记，专员上线后会联系您。",
        callback_due_at=None,
    )


def test_handoff_commits_messages_state_record_callback_and_pending_event_atomically(tmp_path) -> None:
    store = SQLiteCaseStore(tmp_path / "atomic.sqlite3")
    store.ensure_session("consumer-demo", "case-atomic")

    result = _commit(store)

    assert result["created"] is True
    assert result["ack_text"].endswith("专员上线后会联系您。")
    conversation = store.get_conversation("case-atomic", viewer_role="agent")
    assert [(item["sender_type"], item["content"]) for item in conversation["messages"]] == [
        ("consumer", "请帮我转人工"),
        ("ai", result["ack_text"]),
    ]
    assert conversation["status"] == "waiting_human"
    assert conversation["handler"] == "human"
    state = conversation["case_state"]
    assert state["handoff_id"] == result["handoff_id"]
    assert state["handoff_summary"]["temporary"] is True
    assert state["urgency"] == "P3"
    assert state["temporary_urgency"]["value"] == "P3"
    assert state["temporary_urgency"]["rule_version"] == "handoff-urgency-v2"
    assert state["handoff_summary"]["source_message_seq"] == result["source_message_seq"]
    with sqlite3.connect(store.path) as db:
        stored_summary = json.loads(db.execute(
            "SELECT temporary_summary_json FROM handoff_records WHERE handoff_id=?",
            (result["handoff_id"],),
        ).fetchone()[0])
    assert result["source_message_seq"] in stored_summary["sources"]

    with sqlite3.connect(store.path) as db:
        handoff = db.execute(
            "SELECT temporary_summary_json,temporary_priority,temporary_priority_score FROM handoff_records WHERE handoff_id=?",
            (result["handoff_id"],),
        ).fetchone()
        callback = db.execute(
            "SELECT due_at,status FROM callback_tasks WHERE handoff_id=?",
            (result["handoff_id"],),
        ).fetchone()
        event = db.execute(
            "SELECT event_type,data_json FROM handoff_events WHERE handoff_id=?",
            (result["handoff_id"],),
        ).fetchone()
    assert json.loads(handoff[0])["text"] == "顾客直接要求转人工，暂无其他诉求"
    assert handoff[1:] == ("P3", 0)
    assert callback == (None, "pending")
    assert event[0] == "handoff.started"
    assert json.loads(event[1])["delivery_status"] == "pending"
    with sqlite3.connect(store.path) as db:
        blocks = db.execute(
            "SELECT block_type,status FROM handoff_blocks WHERE handoff_id=? ORDER BY block_type",
            (result["handoff_id"],),
        ).fetchall()
    assert [(item[0], item[1]) for item in blocks] == [
        ("emotion_urgency", "pending"),
        ("next_steps", "pending"),
        ("summary", "pending"),
    ]


def test_handoff_can_create_a_new_conversation_inside_its_transaction(tmp_path) -> None:
    store = SQLiteCaseStore(tmp_path / "new-session-success.sqlite3")

    result = _commit(store, conv_id="new-case-atomic")

    assert result["created"] is True
    session = store.get_session("new-case-atomic")
    assert session is not None
    assert session["status"] == "waiting_human"
    assert [item["sender_type"] for item in store.get_conversation(
        "new-case-atomic", viewer_role="agent"
    )["messages"]] == ["consumer", "ai"]


def test_repeated_handoff_returns_same_id_and_never_duplicates_ack_or_block_group(tmp_path) -> None:
    store = SQLiteCaseStore(tmp_path / "idempotent.sqlite3")
    store.ensure_session("consumer-demo", "case-atomic")

    first = _commit(store)
    retry = _commit(store)
    repeated_click = _commit(store, idempotency_key="handoff:request-2")
    conversation = store.get_conversation("case-atomic", viewer_role="agent")

    assert retry["handoff_id"] == first["handoff_id"]
    assert repeated_click["handoff_id"] == first["handoff_id"]
    assert len([item for item in conversation["messages"] if item["sender_type"] == "ai"]) == 1
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT COUNT(*) FROM handoff_records").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM handoff_events").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM handoff_blocks").fetchone()[0] == 3


def test_concurrent_handoff_requests_serialize_to_one_active_record(tmp_path) -> None:
    store = SQLiteCaseStore(tmp_path / "concurrent.sqlite3")
    store.ensure_session("consumer-demo", "case-atomic")

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(
            lambda key: _commit(store, idempotency_key=key),
            ("handoff:parallel-1", "handoff:parallel-2"),
        ))

    assert results[0]["handoff_id"] == results[1]["handoff_id"]
    assert sum(bool(item["created"]) for item in results) == 1
    conversation = store.get_conversation("case-atomic", viewer_role="agent")
    assert len([item for item in conversation["messages"] if item["sender_type"] == "ai"]) == 1
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT COUNT(*) FROM handoff_records").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM handoff_blocks").fetchone()[0] == 3


def test_handoff_transaction_rolls_back_all_customer_visible_writes_on_event_failure(tmp_path) -> None:
    store = SQLiteCaseStore(tmp_path / "rollback.sqlite3")
    store.ensure_session("consumer-demo", "case-atomic")
    with sqlite3.connect(store.path) as db:
        db.execute(
            "CREATE TRIGGER reject_handoff_event BEFORE INSERT ON handoff_events "
            "BEGIN SELECT RAISE(ABORT, 'event write rejected'); END"
        )

    with pytest.raises(sqlite3.IntegrityError, match="event write rejected"):
        _commit(store)

    assert store.get_conversation("case-atomic", viewer_role="agent")["messages"] == []
    assert store.get_session("case-atomic")["status"] == "ai_processing"
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT COUNT(*) FROM handoff_records").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM callback_tasks").fetchone()[0] == 0


def test_handoff_transaction_rolls_back_new_session_when_any_handoff_write_fails(tmp_path) -> None:
    store = SQLiteCaseStore(tmp_path / "new-session-rollback.sqlite3")
    with sqlite3.connect(store.path) as db:
        db.execute(
            "CREATE TRIGGER reject_handoff_event BEFORE INSERT ON handoff_events "
            "BEGIN SELECT RAISE(ABORT, 'event write rejected'); END"
        )

    with pytest.raises(sqlite3.IntegrityError, match="event write rejected"):
        store.start_handoff(
            "new-case-atomic",
            user_id="consumer-demo",
            message="请帮我转人工",
            idempotency_key="handoff:new-case:one",
            reason="consumer_requested_human",
            temporary_summary={"text": "顾客直接要求转人工，暂无其他诉求", "sources": []},
            temporary_priority="P3",
            temporary_priority_score=0,
            temporary_priority_reasons=["转人工本身不加分"],
            temporary_priority_rule_version="test-v1",
            online_ack="为您转接人工，请您稍等。",
            unknown_presence_ack="已登记，专员上线后会联系您。",
            callback_due_at=None,
        )

    assert store.get_session("new-case-atomic") is None
    assert store.get_conversation("new-case-atomic", viewer_role="agent")["messages"] == []


def test_fresh_online_staff_uses_online_copy_and_does_not_create_callback(tmp_path) -> None:
    store = SQLiteCaseStore(tmp_path / "online.sqlite3")
    store.ensure_session("consumer-demo", "case-atomic")
    with sqlite3.connect(store.path) as db:
        db.execute(
            "INSERT INTO staff_presence(agent_id,status,observed_at,expires_at,source) "
            "VALUES ('agent-1','online','2099-01-01T00:00:00+00:00','2099-01-02T00:00:00+00:00','test')"
        )

    result = _commit(store)

    assert result["ack_text"] == "为您转接人工，请您稍等。"
    with sqlite3.connect(store.path) as db:
        assert db.execute("SELECT COUNT(*) FROM callback_tasks").fetchone()[0] == 0


def test_copy_and_callback_policy_come_from_config() -> None:
    from handoff.config import load_handoff_copy_config

    config = load_handoff_copy_config()
    assert config["online_ack"] == "已为您转接人工，请您稍等。"
    assert "专员上线后会联系您" in config["unknown_presence_ack"]
    assert config["callback_sla_minutes"] is None
