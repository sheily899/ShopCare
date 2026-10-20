"""验证转人工交接迁移在旧 SQLite 数据库上的兼容与回滚。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from storage.sqlite_case_store import SQLiteCaseStore


MIGRATIONS = Path(__file__).resolve().parents[1] / "storage" / "migrations"


def _legacy_database(path: Path) -> None:
    """构造已有会话和工单数据、但尚未包含交接表的旧版数据库。"""
    with sqlite3.connect(path) as db:
        db.executescript(
            """
            CREATE TABLE sessions (
                conv_id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'ai_processing', handler TEXT NOT NULL DEFAULT 'ai',
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE messages (
                message_id TEXT PRIMARY KEY, conv_id TEXT NOT NULL,
                sender_type TEXT NOT NULL, sender_id TEXT NOT NULL, content TEXT NOT NULL,
                visible_to_consumer INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL
            );
            CREATE TABLE case_states (
                conv_id TEXT PRIMARY KEY, state_json TEXT NOT NULL DEFAULT '{}', updated_at TEXT NOT NULL
            );
            CREATE TABLE tickets (
                ticket_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, conv_id TEXT NOT NULL,
                main_type TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'draft',
                items_json TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            INSERT INTO sessions VALUES ('legacy-conv','consumer-demo','ai_processing','ai','2026-01-01','2026-01-01');
            INSERT INTO messages VALUES ('legacy-msg','legacy-conv','consumer','consumer-demo','旧会话原文',1,'2026-01-01');
            INSERT INTO case_states VALUES ('legacy-conv','{"urgency":"P2"}','2026-01-01');
            INSERT INTO tickets VALUES ('legacy-ticket','consumer-demo','legacy-conv','return_refund','draft','[]','2026-01-01','2026-01-01');
            """
        )


def test_handoff_migration_is_additive_and_preserves_legacy_records(tmp_path: Path) -> None:
    """旧会话/工单数据应保留，新增表及迁移版本应完整出现。"""
    path = tmp_path / "legacy.sqlite3"
    _legacy_database(path)

    store = SQLiteCaseStore(path)

    assert store.get_conversation("legacy-conv", viewer_role="agent")["messages"][0]["content"] == "旧会话原文"
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT ticket_id FROM tickets WHERE conv_id='legacy-conv'").fetchone()[0] == "legacy-ticket"
        assert db.execute("PRAGMA user_version").fetchone()[0] == 2
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {
        "case_sub_requests",
        "case_action_logs",
        "handoff_records",
        "handoff_blocks",
        "staff_presence",
        "callback_tasks",
        "handoff_events",
        "case_action_log_versions",
    } <= tables


def test_handoff_migration_rollback_is_reapplicable_and_keeps_core_tables(tmp_path: Path) -> None:
    """回滚脚本只移除交接扩展；既有业务表和数据仍可用，迁移可再次应用。"""
    path = tmp_path / "rollback.sqlite3"
    store = SQLiteCaseStore(path)
    store.ensure_session("consumer-demo", "rollback-conv")
    store.append_message("rollback-conv", "consumer", "consumer-demo", "保留这条消息")
    rollback_scripts = [
        (MIGRATIONS / "002_case_action_log_version_rollback.sql").read_text(encoding="utf-8"),
        (MIGRATIONS / "001_handoff_rollback.sql").read_text(encoding="utf-8"),
    ]

    with sqlite3.connect(path) as db:
        for rollback_sql in rollback_scripts:
            db.executescript(rollback_sql)
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert db.execute("PRAGMA user_version").fetchone()[0] == 0
        assert "handoff_records" not in tables
        assert {"sessions", "messages", "tickets", "case_states"} <= tables

    reopened = SQLiteCaseStore(path)
    assert reopened.get_conversation("rollback-conv", viewer_role="agent")["messages"][0]["content"] == "保留这条消息"
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 2


def test_handoff_migration_enforces_allowed_lifecycle_values(tmp_path: Path) -> None:
    """交接块与子诉求状态必须受数据库约束，防止写入漂移状态。"""
    store = SQLiteCaseStore(tmp_path / "constraints.sqlite3")
    store.ensure_session("consumer-demo", "constraints-conv")
    with sqlite3.connect(store.path) as db:
        with pytest.raises(sqlite3.IntegrityError):
            db.execute(
                "INSERT INTO case_sub_requests(sub_request_id,conv_id,intent_id,status,source_ref,updated_at) "
                "VALUES ('sub-1','constraints-conv','return_refund','unknown','msg-1','2026-01-01')"
            )

