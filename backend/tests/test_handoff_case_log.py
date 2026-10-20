"""案件子诉求与动作日志的状态、幂等和敏感信息边界测试。"""

from __future__ import annotations

import sqlite3

import pytest

from storage.sqlite_case_store import SQLiteCaseStore


def test_sub_request_lifecycle_is_preserved_across_message_updates(tmp_path) -> None:
    store = SQLiteCaseStore(tmp_path / "sub-requests.sqlite3")
    store.ensure_session("consumer-demo", "case-1")
    store.sync_case_sub_requests(
        "case-1",
        [{"id": "sub-return", "intent": "return_refund"}],
        source_ref="message-1",
        source_message_seq=1,
    )

    store.set_case_sub_request_status("case-1", "sub-return", "in_progress", source_ref="tool-1")
    store.sync_case_sub_requests(
        "case-1",
        [{"id": "sub-return", "intent": "return_refund"}],
        source_ref="message-2",
        source_message_seq=2,
    )

    items = store.list_case_sub_requests("case-1")
    assert len(items) == 1
    assert items[0]["status"] == "in_progress"
    assert items[0]["source_ref"] == "message-2"
    with pytest.raises(ValueError, match="invalid sub-request status"):
        store.set_case_sub_request_status("case-1", "sub-return", "done")

    initialized = store.sync_case_sub_requests(
        "case-1",
        [{"id": "sub-gift", "intent": "missing_gift", "status": "blocked"}],
        source_ref="message-3",
        source_message_seq=3,
    )
    assert initialized[0]["status"] == "blocked"


def test_action_log_status_updates_are_idempotent_and_scrub_personal_data(tmp_path) -> None:
    store = SQLiteCaseStore(tmp_path / "actions.sqlite3")
    store.ensure_session("consumer-demo", "case-2")
    pending = store.append_case_action(
        "case-2", action_type="tool", description="调用订单查询工具，手机号13812345678",
        source_ref="message-7", idempotency_key="tool:request-7:call-1", status="pending",
    )
    completed = store.append_case_action(
        "case-2", action_type="tool", description="调用订单查询工具，手机号13812345678",
        source_ref="message-7", idempotency_key="tool:request-7:call-1", status="success",
    )

    rows = store.list_case_action_logs("case-2")
    assert pending["action_id"] == completed["action_id"]
    assert len(rows) == 1
    assert rows[0]["status"] == "success"
    assert "13812345678" not in rows[0]["description"]
    assert rows[0]["source_ref"] == "message-7"
    assert store.get_case_action_log_version("case-2") == 2
    store.append_case_action(
        "case-2", action_type="tool", description="调用订单查询工具，手机号13812345678",
        source_ref="message-7", idempotency_key="tool:request-7:call-1", status="success",
    )
    assert store.get_case_action_log_version("case-2") == 2


def test_visible_messages_and_ticket_writes_are_logged_only_when_handoff_enabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HANDOFF", "true")
    store = SQLiteCaseStore(tmp_path / "writes.sqlite3")
    store.ensure_session("consumer-demo", "case-3")

    message = store.append_message("case-3", "ai", "shopcare-ai", "订单查询结果已整理")
    ticket = store.create_ticket("consumer-demo", "case-3", "return_refund", "退货", ["return_refund"])
    rows = store.list_case_action_logs("case-3")

    assert any(item["action_type"] == "reply" and item["source_ref"] == message["message_id"] for item in rows)
    assert any(item["action_type"] == "ticket" and ticket["ticket_id"] in item["source_ref"] for item in rows)
    assert all("订单查询结果已整理" not in item["description"] for item in rows)


def test_handoff_action_logging_is_disabled_by_default(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("HANDOFF", raising=False)
    store = SQLiteCaseStore(tmp_path / "disabled.sqlite3")
    store.ensure_session("consumer-demo", "case-disabled")

    store.append_message("case-disabled", "ai", "shopcare-ai", "不记录回复正文")

    assert store.list_case_action_logs("case-disabled") == []


def test_only_explicit_positive_resolution_confirmation_closes_sub_requests(tmp_path) -> None:
    from api.main import _consumer_confirms_resolution

    store = SQLiteCaseStore(tmp_path / "resolution.sqlite3")
    store.ensure_session("consumer-demo", "case-resolution")
    store.sync_case_sub_requests(
        "case-resolution", [{"id": "return", "intent": "return_refund"}],
        source_ref="message-1", source_message_seq=1,
    )

    assert _consumer_confirms_resolution("之前过敏，这次用了没事") is False
    assert _consumer_confirms_resolution("问题还没解决") is False
    assert _consumer_confirms_resolution("现在问题解决了吗？") is False
    assert _consumer_confirms_resolution("问题解决了，但还有一个赠品问题") is False
    assert _consumer_confirms_resolution("已经解决了，谢谢") is True

    resolved = store.resolve_active_case_sub_requests("case-resolution", source_ref="message-2")
    assert resolved == 1
    assert store.list_case_sub_requests("case-resolution")[0]["status"] == "resolved"
    reopened = store.sync_case_sub_requests(
        "case-resolution", [{"id": "return", "intent": "return_refund"}],
        source_ref="message-3", source_message_seq=3,
    )
    assert reopened[0]["status"] == "open"
    assert len(store.list_case_sub_requests("case-resolution")) == 2




def test_action_log_rejects_invalid_types_and_states(tmp_path) -> None:
    store = SQLiteCaseStore(tmp_path / "constraints.sqlite3")
    store.ensure_session("consumer-demo", "case-4")
    with pytest.raises(sqlite3.IntegrityError):
        store.append_case_action(
            "case-4", action_type="unknown", description="不应写入", source_ref="message-1",
            idempotency_key="invalid:1", status="success",
        )
