"""ShopCare 的最小 SQLite 业务状态存储。

Redis 仍可作为短期缓存；本模块保存跨端口必须共享的事实，避免消费者端和客服端
各自维护一份会话状态。
"""

from __future__ import annotations

import json
import hashlib
import re
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from handoff.feature_flag import handoff_enabled


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_action_description(value: str) -> str:
    """只保留可读操作说明，并遮蔽常见手机号、邮箱和身份证号。"""
    text = re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "[手机号已隐藏]", str(value or ""))
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[邮箱已隐藏]", text)
    text = re.sub(r"(?<!\d)\d{17}[\dXx](?!\w)", "[证件号已隐藏]", text)
    return text.strip()[:240]


class SQLiteCaseStore:
    """保存会话、消息、案件状态、工单和风险事件的轻量存储。"""

    def __init__(self, path: str | Path = ".data/shopcare.sqlite3") -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._initialise()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialise(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    conv_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'ai_processing',
                    handler TEXT NOT NULL DEFAULT 'ai',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('consumer','agent')),
                    display_name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS customer_agent_assignments (
                    assignment_id TEXT PRIMARY KEY,
                    consumer_id TEXT NOT NULL REFERENCES users(user_id),
                    agent_id TEXT NOT NULL REFERENCES users(user_id),
                    status TEXT NOT NULL DEFAULT 'active',
                    assigned_at TEXT NOT NULL,
                    ended_at TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_active_consumer_assignment
                    ON customer_agent_assignments(consumer_id) WHERE status='active';
                CREATE TABLE IF NOT EXISTS auth_tokens (
                    token TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(user_id),
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conv_id TEXT NOT NULL REFERENCES sessions(conv_id),
                    sender_type TEXT NOT NULL,
                    sender_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    visible_to_consumer INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS case_states (
                    conv_id TEXT PRIMARY KEY REFERENCES sessions(conv_id),
                    state_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tickets (
                    ticket_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    conv_id TEXT NOT NULL REFERENCES sessions(conv_id),
                    buyer_nickname TEXT NOT NULL DEFAULT '',
                    shop TEXT NOT NULL DEFAULT 'ShopCare',
                    main_type TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'draft',
                    items_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_one_open_ticket
                    ON tickets(user_id) WHERE status NOT IN ('closed', 'done', 'rejected');
                CREATE TABLE IF NOT EXISTS risk_events (
                    risk_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    conv_id TEXT NOT NULL REFERENCES sessions(conv_id),
                    level TEXT NOT NULL,
                    risk_type TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'new',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_logs (
                    audit_id TEXT PRIMARY KEY,
                    conv_id TEXT NOT NULL REFERENCES sessions(conv_id),
                    actor_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    details_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS slot_values (
                    slot_value_id TEXT PRIMARY KEY,
                    conv_id TEXT NOT NULL REFERENCES sessions(conv_id),
                    slot_id TEXT NOT NULL,
                    value TEXT NOT NULL,
                    display_value TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_ref TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 1.0,
                    confirmation_status TEXT NOT NULL DEFAULT 'confirmed',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS stream_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conv_id TEXT NOT NULL REFERENCES sessions(conv_id),
                    event_type TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memory_short_term (
                    memory_id TEXT PRIMARY KEY, conv_id TEXT NOT NULL REFERENCES sessions(conv_id),
                    role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memory_long_term (
                    memory_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, conv_id TEXT NOT NULL REFERENCES sessions(conv_id),
                    summary TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY, profile_json TEXT NOT NULL DEFAULT '{}', updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS observability_events (
                    trace_id TEXT PRIMARY KEY, conv_id TEXT NOT NULL REFERENCES sessions(conv_id),
                    phases_json TEXT NOT NULL DEFAULT '{}', cache_hits_json TEXT NOT NULL DEFAULT '{}',
                    degradation_reason TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL
                );
                """
            )
            columns = {row[1] for row in db.execute("PRAGMA table_info(tickets)").fetchall()}
            for name, definition in (("buyer_nickname", "TEXT NOT NULL DEFAULT ''"), ("shop", "TEXT NOT NULL DEFAULT 'ShopCare'"), ("completed_at", "TEXT")):
                if name not in columns:
                    db.execute(f"ALTER TABLE tickets ADD COLUMN {name} {definition}")
            self._apply_migrations(db)
            self._seed_demo_users(db)

    @staticmethod
    def _apply_migrations(db: sqlite3.Connection) -> None:
        """按 PRAGMA user_version 顺序运行仅增量的 SQLite schema migrations。

        迁移脚本显式使用 BEGIN/COMMIT，保证单个版本要么整体应用，要么整体回滚。
        回滚脚本不自动执行，只供备份后维护和临时数据库验收使用。
        """
        migration_dir = Path(__file__).with_name("migrations")
        migration_paths = sorted(
            path for path in migration_dir.glob("[0-9][0-9][0-9]_*.sql")
            if not path.stem.endswith("_rollback")
        )
        current_version = int(db.execute("PRAGMA user_version").fetchone()[0])
        known_versions = [int(path.name.split("_", 1)[0]) for path in migration_paths]
        if known_versions and current_version > max(known_versions):
            raise RuntimeError(
                f"SQLite schema version {current_version} is newer than this application"
            )
        for migration_path in migration_paths:
            version = int(migration_path.name.split("_", 1)[0])
            if version <= current_version:
                continue
            db.executescript(migration_path.read_text(encoding="utf-8"))
            current_version = int(db.execute("PRAGMA user_version").fetchone()[0])
            if current_version != version:
                raise RuntimeError(
                    f"SQLite migration {migration_path.name} set schema version {current_version}, expected {version}"
                )

    @staticmethod
    def _password_hash(password: str) -> str:
        return hashlib.sha256(password.encode("utf-8")).hexdigest()

    def _seed_demo_users(self, db: sqlite3.Connection) -> None:
        now = _now()
        demo = [
            ("consumer-demo", "consumer-demo", "shopcare", "consumer", "演示消费者"),
            ("agent-demo", "agent-demo", "shopcare", "agent", "演示客服"),
        ]
        for user_id, username, password, role, display_name in demo:
            db.execute(
                "INSERT OR IGNORE INTO users(user_id,username,password_hash,role,display_name,created_at,updated_at) VALUES (?,?,?,?,?,?,?)",
                (user_id, username, self._password_hash(password), role, display_name, now, now),
            )
        db.execute(
            "INSERT OR IGNORE INTO customer_agent_assignments(assignment_id,consumer_id,agent_id,assigned_at) VALUES (?,?,?,?)",
            ("assignment-demo-1", "consumer-demo", "agent-demo", now),
        )

    def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM users WHERE username=? AND status='active'", (username,)).fetchone()
            if not row or row["password_hash"] != self._password_hash(password):
                return None
            token = secrets.token_urlsafe(24)
            db.execute("INSERT INTO auth_tokens(token,user_id,created_at) VALUES (?,?,?)", (token, row["user_id"], _now()))
        return {"token": token, "user_id": row["user_id"], "role": row["role"], "display_name": row["display_name"]}

    def register_consumer(self, username: str, password: str, display_name: str) -> Dict[str, Any]:
        now = _now()
        user_id = f"consumer-{uuid.uuid4().hex[:10]}"
        with self._connect() as db:
            if db.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone():
                raise ValueError("账号已存在")
            db.execute(
                "INSERT INTO users(user_id,username,password_hash,role,display_name,created_at,updated_at) VALUES (?,?,?,?,?,?,?)",
                (user_id, username, self._password_hash(password), "consumer", display_name, now, now),
            )
            db.execute(
                "INSERT INTO customer_agent_assignments(assignment_id,consumer_id,agent_id,assigned_at) VALUES (?,?,?,?)",
                (f"assignment-{uuid.uuid4().hex[:10]}", user_id, "agent-demo", now),
            )
        return {"user_id": user_id, "username": username, "role": "consumer", "display_name": display_name}

    def get_identity(self, token: str) -> Optional[Dict[str, Any]]:
        with self._connect() as db:
            row = db.execute("SELECT u.* FROM auth_tokens t JOIN users u ON u.user_id=t.user_id WHERE t.token=? AND u.status='active'", (token,)).fetchone()
        return dict(row) if row else None

    def get_user_display_name(self, user_id: str) -> str:
        with self._connect() as db:
            row = db.execute("SELECT display_name FROM users WHERE user_id=?", (user_id,)).fetchone()
        return str(row[0]) if row else user_id

    def is_assigned(self, agent_id: str, consumer_id: str) -> bool:
        with self._connect() as db:
            row = db.execute(
                "SELECT 1 FROM customer_agent_assignments WHERE agent_id=? AND consumer_id=? AND status='active'",
                (agent_id, consumer_id),
            ).fetchone()
        return row is not None

    def ensure_session(self, user_id: str, conv_id: str) -> Dict[str, Any]:
        now = _now()
        with self._connect() as db:
            db.execute(
                """INSERT INTO sessions(conv_id,user_id,created_at,updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(conv_id) DO UPDATE SET updated_at=excluded.updated_at""",
                (conv_id, user_id, now, now),
            )
            db.execute(
                "INSERT OR IGNORE INTO case_states(conv_id,state_json,updated_at) VALUES (?, ?, ?)",
                (conv_id, json.dumps({"version": 0}, ensure_ascii=False), now),
            )
            row = db.execute("SELECT * FROM sessions WHERE conv_id=?", (conv_id,)).fetchone()
        return dict(row)

    def get_session(self, conv_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM sessions WHERE conv_id=?", (conv_id,)).fetchone()
        return dict(row) if row else None

    def list_sessions(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        with self._connect() as db:
            if agent_id:
                rows = db.execute(
                    "SELECT s.conv_id,s.user_id,s.status,s.handler,s.created_at,s.updated_at FROM sessions s "
                    "JOIN customer_agent_assignments a ON a.consumer_id=s.user_id AND a.agent_id=? AND a.status='active' "
                    "ORDER BY s.updated_at DESC", (agent_id,)
                ).fetchall()
            else:
                rows = db.execute("SELECT conv_id,user_id,status,handler,created_at,updated_at FROM sessions ORDER BY updated_at DESC").fetchall()
            states = {
                row["conv_id"]: json.loads(row["state_json"])
                for row in db.execute("SELECT conv_id,state_json FROM case_states").fetchall()
            }
        result = []
        for row in rows:
            item = dict(row)
            state = states.get(item["conv_id"], {})
            summary = state.get("handoff_summary") or {}
            ticket = state.get("ticket_draft") or {}
            # 兼容旧会话：优先读取新的 P0-P3 字段，再从摘要/工单回填，避免列表继续显示旧 P3。
            item["urgency"] = state.get("urgency") or summary.get("urgency") or ticket.get("risk_level") or ("P1" if summary.get("intent") == "adverse_reaction" else "P3")
            item["risk_level"] = state.get("risk_level") or ticket.get("risk_level") or item["urgency"] or "待识别"
            result.append(item)
        return {"sessions": result}

    def get_risk_dashboard(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """从 SQLite 会话、case_state 和工单快照汇总客服风险看板。"""
        sessions = self.list_sessions(agent_id).get("sessions", [])
        total = len(sessions)
        handoff = sum(1 for item in sessions if item.get("handler") == "human" or item.get("status") == "waiting_human")
        p01 = sum(1 for item in sessions if item.get("urgency") in {"P0", "P1"})
        pending_tickets = 0
        users = []
        with self._connect() as db:
            states = {row["conv_id"]: json.loads(row["state_json"]) for row in db.execute("SELECT conv_id,state_json FROM case_states").fetchall()}
        for item in sessions:
            state = states.get(item["conv_id"], {})
            ticket = state.get("ticket_draft") or {}
            if ticket and ticket.get("status", "draft") not in {"submitted", "done", "closed"}:
                pending_tickets += 1
            summary = state.get("handoff_summary") or {}
            case_facts = state.get("case_facts") or {}
            fact_records = case_facts.get("facts") if isinstance(case_facts, dict) else {}
            fact_text_parts = []
            if isinstance(fact_records, dict):
                for field_name, records in fact_records.items():
                    if field_name in {"ai_response", "knowledge_evidence", "emotion_intensity"} or not isinstance(records, list):
                        continue
                    for record in records:
                        if isinstance(record, dict) and record.get("value") is not None:
                            fact_text_parts.append(f"{field_name}:{record['value']}")
            source_text = " ".join(
                str(value)
                for key in ("request", "handoff_reason", "intent")
                for value in [summary.get(key, "")]
            ) + " " + " ".join(str(value) for value in (summary.get("key_facts") or []))
            source_text += " " + " ".join(fact_text_parts)
            risk_tags = list(summary.get("risk_tags") or [])
            intent_id = str(state.get("intent") or summary.get("intent") or ticket.get("intent") or "")
            if intent_id == "adverse_reaction" or any(word in source_text for word in ("过敏", "发红", "红肿", "发痒", "烂脸")):
                risk_tags.append("不良反应")
            if any(word in source_text for word in ("破损", "破了", "漏液")):
                risk_tags.append("物流破损")
            if any(word in source_text for word in ("赠品", "小样")):
                risk_tags.append("赠品问题")
            if any(word in source_text for word in ("投诉", "12315", "曝光", "平台介入")):
                risk_tags.append("平台投诉风险")
            if summary.get("handoff_reason") and not risk_tags:
                risk_tags.append(summary["handoff_reason"])
            risk_tags = list(dict.fromkeys(risk_tags))
            users.append({
                "conv_id": item["conv_id"], "user_id": item["user_id"],
                "urgency": item.get("urgency", "P3"),
                "emotion": state.get("emotion") or summary.get("emotion", "calm"),
                "risk_tags": risk_tags,
                "ticket_id": (ticket.get("system_fields") or {}).get("ticket_id", "") if ticket else "",
                "ticket_status": ticket.get("status", "无工单") if ticket else "无工单",
                "status": item.get("status", "ai_processing"),
            })
        return {
            "metrics": {
                "total_sessions": total,
                "today_sessions": total,
                "ai_resolution_rate": round((total - handoff) / total * 100) if total else 0,
                "handoff_rate": round(handoff / total * 100) if total else 0,
                "pending_p01": p01,
                "pending_tickets": pending_tickets,
            },
            "alert": f"当前有 {p01} 个 P0/P1 风险会话待处理。" if p01 else "当前没有待处理的 P0/P1 风险会话。",
            "users": users,
        }

    def append_message(
        self,
        conv_id: str,
        sender_type: str,
        sender_id: str,
        content: str,
        *,
        visible_to_consumer: bool = True,
        message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        message = {
            "message_id": str(message_id or uuid.uuid4().hex),
            "conv_id": conv_id,
            "sender_type": sender_type,
            "sender_id": sender_id,
            "content": content,
            "visible_to_consumer": bool(visible_to_consumer),
            "created_at": _now(),
        }
        message_seq = 0
        with self._connect() as db:
            insert_sql = """INSERT INTO messages(message_id,conv_id,sender_type,sender_id,content,
                visible_to_consumer,created_at) VALUES (?, ?, ?, ?, ?, ?, ?)"""
            cursor = db.execute(
                insert_sql if message_id is None else insert_sql.replace("INSERT INTO", "INSERT OR IGNORE INTO", 1),
                tuple(message.values()),
            )
            if cursor.rowcount == 0:
                # Why：客服重复点击时复用稳定消息 ID；若首次写入已成功，不重复写短期记忆或对话消息。
                existing = db.execute(
                    "SELECT rowid AS message_seq,* FROM messages WHERE message_id=?",
                    (message["message_id"],),
                ).fetchone()
                if (
                    existing is None
                    or str(existing["conv_id"]) != conv_id
                    or str(existing["sender_type"]) != sender_type
                    or str(existing["content"]) != content
                    or bool(existing["visible_to_consumer"]) != bool(visible_to_consumer)
                ):
                    raise ValueError("message_id 已被不同消息占用")
                return {**dict(existing), "message_seq": int(existing["message_seq"])}
            message_seq = int(cursor.lastrowid)
            db.execute("UPDATE sessions SET updated_at=? WHERE conv_id=?", (_now(), conv_id))
            db.execute("INSERT INTO memory_short_term(memory_id,conv_id,role,content,created_at) VALUES (?,?,?,?,?)", (uuid.uuid4().hex, conv_id, sender_type, content, message["created_at"]))
            db.execute("DELETE FROM memory_short_term WHERE conv_id=? AND memory_id NOT IN (SELECT memory_id FROM memory_short_term WHERE conv_id=? ORDER BY created_at DESC LIMIT 10)", (conv_id, conv_id))
            if handoff_enabled() and sender_type == "consumer":
                active_handoff = db.execute(
                    "SELECT handoff_id FROM handoff_records WHERE conv_id=? AND status IN ('open','in_progress')",
                    (conv_id,),
                ).fetchone()
                if active_handoff:
                    db.execute(
                        "INSERT INTO handoff_events(handoff_id,conv_id,event_type,data_json,created_at) VALUES (?,?,?,?,?)",
                        (
                            str(active_handoff["handoff_id"]), conv_id, "handoff.message.incremental",
                            json.dumps({"message_id": message["message_id"], "message_seq": message_seq}, ensure_ascii=False),
                            _now(),
                        ),
                    )
            if handoff_enabled() and sender_type in {"ai", "customer_service"}:
                actor_label = "AI 客服" if sender_type == "ai" else "人工客服"
                self._append_case_action_on_connection(
                    db,
                    conv_id,
                    action_type="reply",
                    description=f"{actor_label}答复已发送",
                    source_ref=message["message_id"],
                    idempotency_key=f"reply:{message['message_id']}",
                    status="success",
                )
        message["message_seq"] = message_seq
        return message

    def sync_case_sub_requests(
        self,
        conv_id: str,
        sub_requests: list[Dict[str, Any]],
        *,
        source_ref: str,
        source_message_seq: Optional[int] = None,
    ) -> list[Dict[str, Any]]:
        """按本轮识别结果增量同步业务子诉求，保留已进入的生命周期状态。"""
        now = _now()
        normalized: list[Dict[str, Any]] = []
        with self._connect() as db:
            for index, item in enumerate(sub_requests, start=1):
                if not isinstance(item, dict):
                    continue
                intent_id = str(item.get("intent") or item.get("intent_id") or "").strip()
                if not intent_id or intent_id in {"human_handoff", "other"}:
                    continue
                request_key = str(item.get("id") or item.get("sub_request_key") or f"{intent_id}:{index}")
                existing = db.execute(
                    "SELECT sub_request_id,status FROM case_sub_requests WHERE conv_id=? AND sub_request_key=?",
                    (conv_id, request_key),
                ).fetchone()
                if existing is not None and str(existing["status"]) == "resolved":
                    # 同意图后来再次出现时另开一条诉求；原已解决记录保留审计历史。
                    occurrence = source_message_seq if source_message_seq is not None else source_ref
                    request_key = f"{request_key}@{occurrence}"
                    existing = db.execute(
                        "SELECT sub_request_id,status FROM case_sub_requests WHERE conv_id=? AND sub_request_key=?",
                        (conv_id, request_key),
                    ).fetchone()
                sub_request_id = (
                    str(existing["sub_request_id"])
                    if existing else uuid.uuid5(uuid.NAMESPACE_URL, f"shopcare:{conv_id}:{request_key}").hex
                )
                initial_status = (
                    str(item.get("status"))
                    if item.get("status") in {"open", "in_progress", "resolved", "blocked"}
                    else "open"
                )
                db.execute(
                    """INSERT INTO case_sub_requests(
                           sub_request_id,conv_id,sub_request_key,intent_id,status,source_ref,
                           source_message_seq,created_at,updated_at
                       ) VALUES (?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(conv_id,sub_request_key) DO UPDATE SET
                           intent_id=excluded.intent_id,source_ref=excluded.source_ref,
                           source_message_seq=excluded.source_message_seq,updated_at=excluded.updated_at""",
                    (
                        sub_request_id, conv_id, request_key, intent_id,
                        initial_status,
                        source_ref, source_message_seq, now, now,
                    ),
                )
                normalized.append({
                    "sub_request_id": sub_request_id,
                    "sub_request_key": request_key,
                    "intent_id": intent_id,
                    "status": str(existing["status"]) if existing else initial_status,
                    "source_ref": source_ref,
                    "source_message_seq": source_message_seq,
                })
            return normalized

    def set_case_sub_request_status(
        self, conv_id: str, sub_request_key: str, status: str, *, source_ref: str = ""
    ) -> Dict[str, Any]:
        """更新指定子诉求状态，只允许 open/in_progress/resolved/blocked。"""
        allowed = {"open", "in_progress", "resolved", "blocked"}
        if status not in allowed:
            raise ValueError(f"invalid sub-request status: {status}")
        with self._connect() as db:
            row = db.execute(
                "SELECT sub_request_id,intent_id FROM case_sub_requests WHERE conv_id=? AND sub_request_key=?",
                (conv_id, sub_request_key),
            ).fetchone()
            if row is None:
                raise ValueError(f"unknown sub-request: {sub_request_key}")
            db.execute(
                "UPDATE case_sub_requests SET status=?,source_ref=?,updated_at=? WHERE conv_id=? AND sub_request_key=?",
                (status, source_ref, _now(), conv_id, sub_request_key),
            )
            self._append_case_action_on_connection(
                db, conv_id, action_type="system",
                description=f"业务诉求状态更新为{status}", source_ref=source_ref,
                idempotency_key=f"subrequest:{row['sub_request_id']}:{status}:{source_ref}",
                status="success",
            )
            result = db.execute(
                "SELECT * FROM case_sub_requests WHERE conv_id=? AND sub_request_key=?",
                (conv_id, sub_request_key),
            ).fetchone()
        return dict(result)

    def list_case_sub_requests(self, conv_id: str) -> list[Dict[str, Any]]:
        """按创建顺序读取会话业务子诉求及状态。"""
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM case_sub_requests WHERE conv_id=? ORDER BY created_at,sub_request_id",
                (conv_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def resolve_active_case_sub_requests(self, conv_id: str, *, source_ref: str) -> int:
        """仅在消费者明确确认已解决后，关闭当前会话尚未结束的子诉求。"""
        now = _now()
        resolved = 0
        with self._connect() as db:
            rows = db.execute(
                """SELECT sub_request_id,sub_request_key FROM case_sub_requests
                   WHERE conv_id=? AND status IN ('open','in_progress','blocked')""",
                (conv_id,),
            ).fetchall()
            for row in rows:
                db.execute(
                    "UPDATE case_sub_requests SET status='resolved',source_ref=?,updated_at=? WHERE sub_request_id=?",
                    (source_ref, now, row["sub_request_id"]),
                )
                self._append_case_action_on_connection(
                    db, conv_id, action_type="system", description="消费者确认业务诉求已解决",
                    source_ref=source_ref,
                    idempotency_key=f"subrequest:{row['sub_request_id']}:resolved:{source_ref}",
                    status="success", sub_request_id=row["sub_request_id"],
                )
                resolved += 1
        return resolved

    @staticmethod
    def _append_case_action_on_connection(
        db: sqlite3.Connection,
        conv_id: str,
        *,
        action_type: str,
        description: str,
        source_ref: str,
        idempotency_key: str,
        status: str,
        handoff_id: Optional[str] = None,
        sub_request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """在调用方事务内幂等写入安全的案件动作描述。"""
        if not idempotency_key:
            raise ValueError("idempotency_key is required")
        now = _now()
        action_id = uuid.uuid5(uuid.NAMESPACE_URL, f"shopcare:action:{idempotency_key}").hex
        item = {
            "action_id": action_id, "conv_id": conv_id, "handoff_id": handoff_id,
            "sub_request_id": sub_request_id, "action_type": action_type,
            "description": _safe_action_description(description), "source_ref": str(source_ref or ""),
            "status": status, "idempotency_key": idempotency_key,
            "created_at": now, "updated_at": now,
        }
        existing = db.execute(
            "SELECT * FROM case_action_logs WHERE idempotency_key=?", (idempotency_key,)
        ).fetchone()
        if existing is None:
            db.execute(
                """INSERT INTO case_action_logs(
                       action_id,conv_id,handoff_id,sub_request_id,action_type,description,source_ref,
                       status,idempotency_key,created_at,updated_at
                   ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                tuple(item.values()),
            )
            db.execute(
                """INSERT INTO case_action_log_versions(conv_id,version) VALUES (?,1)
                   ON CONFLICT(conv_id) DO UPDATE SET version=version+1""",
                (conv_id,),
            )
        elif any(
            str(existing[field] or "") != str(item[field] or "")
            for field in ("handoff_id", "sub_request_id", "action_type", "description", "source_ref", "status")
        ):
            db.execute(
                """UPDATE case_action_logs SET handoff_id=?,sub_request_id=?,action_type=?,
                       description=?,source_ref=?,status=?,updated_at=? WHERE idempotency_key=?""",
                (
                    handoff_id, sub_request_id, action_type, item["description"],
                    item["source_ref"], status, now, idempotency_key,
                ),
            )
            db.execute(
                """INSERT INTO case_action_log_versions(conv_id,version) VALUES (?,1)
                   ON CONFLICT(conv_id) DO UPDATE SET version=version+1""",
                (conv_id,),
            )
        row = db.execute(
            "SELECT * FROM case_action_logs WHERE idempotency_key=?", (idempotency_key,)
        ).fetchone()
        return dict(row)

    def append_case_action(
        self,
        conv_id: str,
        *,
        action_type: str,
        description: str,
        source_ref: str,
        idempotency_key: str,
        status: str,
        handoff_id: Optional[str] = None,
        sub_request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """新增或更新一条具有幂等键的动作日志。"""
        with self._connect() as db:
            return self._append_case_action_on_connection(
                db, conv_id, action_type=action_type, description=description,
                source_ref=source_ref, idempotency_key=idempotency_key, status=status,
                handoff_id=handoff_id, sub_request_id=sub_request_id,
            )

    def list_case_action_logs(self, conv_id: str) -> list[Dict[str, Any]]:
        """读取会话动作台账；不含原始消息正文或工具载荷。"""
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM case_action_logs WHERE conv_id=? ORDER BY updated_at,rowid",
                (conv_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_case_action_log_version(self, conv_id: str) -> int:
        """返回可用于交接状态哈希的案件动作日志版本。"""
        with self._connect() as db:
            row = db.execute(
                "SELECT version FROM case_action_log_versions WHERE conv_id=?", (conv_id,)
            ).fetchone()
        return int(row[0] or 0)

    def get_message_sequence(self, conv_id: str, message_id: str) -> Optional[int]:
        """获取会话内 SQLite 消息序号，不暴露消息正文。"""
        with self._connect() as db:
            row = db.execute(
                "SELECT rowid FROM messages WHERE conv_id=? AND message_id=?",
                (conv_id, message_id),
            ).fetchone()
        return int(row[0]) if row else None

    def start_handoff(
        self,
        conv_id: str,
        *,
        user_id: str,
        message: str,
        idempotency_key: str,
        reason: str,
        temporary_summary: Dict[str, Any],
        temporary_priority: str,
        temporary_priority_score: int,
        temporary_priority_reasons: list[str],
        temporary_priority_rule_version: str = "",
        online_ack: str,
        unknown_presence_ack: str,
        callback_due_at: Optional[str],
        callback_reason: str = "客服在线状态未知，等待专员联系",
    ) -> Dict[str, Any]:
        """原子提交一次交接；返回时消费者消息、ACK、状态及事件均已提交。"""
        if not idempotency_key:
            raise ValueError("handoff idempotency_key is required")
        if temporary_priority not in {"P0", "P1", "P2", "P3"}:
            raise ValueError("invalid temporary priority")
        now = _now()
        with self._connect() as db:
            # Serialize the active-handoff check and insert so two API workers cannot
            # race into duplicate transfer records for one conversation.
            db.execute("BEGIN IMMEDIATE")
            session = db.execute(
                "SELECT user_id FROM sessions WHERE conv_id=?", (conv_id,)
            ).fetchone()
            if session is None:
                # Why：新会话的壳也必须与交接消息、状态和事件处于同一事务；
                # 如果后续任一写入失败，整个新会话都应回滚，不能留下孤立记录。
                db.execute(
                    "INSERT INTO sessions(conv_id,user_id,created_at,updated_at) VALUES (?,?,?,?)",
                    (conv_id, user_id, now, now),
                )
                db.execute(
                    "INSERT INTO case_states(conv_id,state_json,updated_at) VALUES (?,?,?)",
                    (conv_id, json.dumps({"version": 0}, ensure_ascii=False), now),
                )
                session = db.execute(
                    "SELECT user_id FROM sessions WHERE conv_id=?", (conv_id,)
                ).fetchone()
            if str(session["user_id"]) != user_id:
                raise ValueError("conversation owner does not match handoff requester")

            existing = db.execute(
                "SELECT * FROM handoff_records WHERE idempotency_key=?", (idempotency_key,)
            ).fetchone()
            if existing is None:
                existing = db.execute(
                    "SELECT * FROM handoff_records WHERE conv_id=? AND status IN ('open','in_progress') "
                    "ORDER BY created_at LIMIT 1",
                    (conv_id,),
                ).fetchone()
            if existing is not None:
                handoff_id = str(existing["handoff_id"])
                ack_message_id = uuid.uuid5(
                    uuid.NAMESPACE_URL, f"shopcare:handoff-ack:{handoff_id}"
                ).hex
                ack_message = db.execute(
                    "SELECT content FROM messages WHERE message_id=? AND conv_id=?",
                    (ack_message_id, conv_id),
                ).fetchone()
                if ack_message is None:
                    raise RuntimeError("active handoff has no persisted acknowledgment")
                state_row = db.execute(
                    "SELECT state_json FROM case_states WHERE conv_id=?", (conv_id,)
                ).fetchone()
                active_state = json.loads(state_row[0]) if state_row else {}
                temporary_urgency = active_state.get("temporary_urgency") or {}
                consumer_message = db.execute(
                    "SELECT message_id FROM messages WHERE conv_id=? AND rowid=?",
                    (conv_id, existing["source_message_seq"]),
                ).fetchone()
                consumer_id = uuid.uuid5(
                    uuid.NAMESPACE_URL, f"shopcare:handoff-consumer:{idempotency_key}"
                ).hex
                return {
                    "created": False,
                    "handoff_id": handoff_id,
                    "consumer_message_id": str(consumer_message["message_id"]) if consumer_message else consumer_id,
                    "ack_message_id": ack_message_id,
                    "ack_text": str(ack_message["content"]),
                    "source_message_seq": existing["source_message_seq"],
                    "temporary_summary": json.loads(existing["temporary_summary_json"] or "{}"),
                    "temporary_priority": str(existing["temporary_priority"]),
                    "temporary_priority_score": int(existing["temporary_priority_score"]),
                    "temporary_priority_reasons": json.loads(existing["temporary_priority_reasons_json"] or "[]"),
                    "temporary_priority_rule_version": str(temporary_urgency.get("rule_version") or ""),
                    "online_staff": False,
                }

            online = db.execute(
                "SELECT 1 FROM staff_presence WHERE status='online' AND expires_at IS NOT NULL "
                "AND expires_at>? LIMIT 1",
                (now,),
            ).fetchone() is not None
            ack_text = online_ack if online else unknown_presence_ack
            handoff_id = uuid.uuid5(
                uuid.NAMESPACE_URL, f"shopcare:handoff:{idempotency_key}"
            ).hex
            consumer_message_id = uuid.uuid5(
                uuid.NAMESPACE_URL, f"shopcare:handoff-consumer:{idempotency_key}"
            ).hex
            ack_message_id = uuid.uuid5(
                uuid.NAMESPACE_URL, f"shopcare:handoff-ack:{handoff_id}"
            ).hex
            for message_id, sender_type, sender_id, content in (
                (consumer_message_id, "consumer", user_id, message),
                (ack_message_id, "ai", "shopcare-ai", ack_text),
            ):
                db.execute(
                    "INSERT INTO messages(message_id,conv_id,sender_type,sender_id,content,visible_to_consumer,created_at) "
                    "VALUES (?,?,?,?,?,1,?)",
                    (message_id, conv_id, sender_type, sender_id, content, now),
                )
                db.execute(
                    "INSERT INTO memory_short_term(memory_id,conv_id,role,content,created_at) VALUES (?,?,?,?,?)",
                    (uuid.uuid4().hex, conv_id, sender_type, content, now),
                )
            db.execute(
                "DELETE FROM memory_short_term WHERE conv_id=? AND memory_id NOT IN "
                "(SELECT memory_id FROM memory_short_term WHERE conv_id=? ORDER BY created_at DESC LIMIT 10)",
                (conv_id, conv_id),
            )
            source_message_seq = int(
                db.execute("SELECT rowid FROM messages WHERE message_id=?", (consumer_message_id,)).fetchone()[0]
            )
            summary_payload = dict(temporary_summary or {})
            if not summary_payload.get("text"):
                raise ValueError("temporary handoff summary text is required")
            summary_payload["sources"] = sorted({
                int(item) for item in summary_payload.get("sources", [])
                if item is not None
            } | {source_message_seq})
            reasons = [str(item) for item in temporary_priority_reasons]
            db.execute(
                """INSERT INTO handoff_records(
                       handoff_id,conv_id,idempotency_key,status,reason,source_message_seq,
                       temporary_summary_json,temporary_priority,temporary_priority_score,
                       temporary_priority_reasons_json,state_hash,created_at,updated_at
                   ) VALUES (?,?,?,'open',?,?,?,?,?,?,?, ?,?)""",
                (
                    handoff_id, conv_id, idempotency_key, reason, source_message_seq,
                    json.dumps(summary_payload, ensure_ascii=False), temporary_priority,
                    int(temporary_priority_score), json.dumps(reasons, ensure_ascii=False),
                    "", now, now,
                ),
            )
            state_row = db.execute(
                "SELECT state_json FROM case_states WHERE conv_id=?", (conv_id,)
            ).fetchone()
            state = json.loads(state_row[0]) if state_row else {}
            state["version"] = int(state.get("version", 0) or 0) + 1
            state.update({
                "status": "waiting_human",
                "handoff_required": True,
                "handoff_id": handoff_id,
                "handoff_summary": {
                    "summary": str(summary_payload["text"]),
                    "temporary": True,
                    "urgency": temporary_priority,
                    "urgency_score": int(temporary_priority_score),
                    "urgency_reasons": reasons,
                    "urgency_rule_version": temporary_priority_rule_version,
                    "source_message_seq": source_message_seq,
                },
                "temporary_urgency": {
                    "value": temporary_priority,
                    "score": int(temporary_priority_score),
                    "reasons": reasons,
                    "rule_version": temporary_priority_rule_version,
                    "source_message_seq": source_message_seq,
                },
                "final_urgency": None,
                "urgency": temporary_priority,
            })
            initial_blocks = (
                ("summary", {
                    "text": str(summary_payload["text"]),
                    "temporary": True,
                    "sources": summary_payload["sources"],
                }),
                ("emotion_urgency", {
                    "temporary": {
                        "value": temporary_priority,
                        "score": int(temporary_priority_score),
                        "reasons": reasons,
                        "rule_version": temporary_priority_rule_version,
                        "source_message_seq": source_message_seq,
                    },
                    "final": None,
                }),
                ("next_steps", {"items": [], "temporary": True}),
            )
            for block_type, payload in initial_blocks:
                db.execute(
                    """INSERT INTO handoff_blocks(
                           handoff_id,block_type,status,state_hash,payload_json,
                           source_message_seq,updated_at
                       ) VALUES (?,?,'pending','',?,?,?)""",
                    (
                        handoff_id, block_type, json.dumps(payload, ensure_ascii=False),
                        source_message_seq, now,
                    ),
                )
            db.execute(
                "INSERT INTO case_states(conv_id,state_json,updated_at) VALUES (?,?,?) "
                "ON CONFLICT(conv_id) DO UPDATE SET state_json=excluded.state_json,updated_at=excluded.updated_at",
                (conv_id, json.dumps(state, ensure_ascii=False), now),
            )
            db.execute(
                "UPDATE sessions SET status='waiting_human',handler='human',updated_at=? WHERE conv_id=?",
                (now, conv_id),
            )
            if not online:
                callback_task_id = uuid.uuid5(
                    uuid.NAMESPACE_URL, f"shopcare:callback:{handoff_id}"
                ).hex
                db.execute(
                    """INSERT INTO callback_tasks(
                           callback_task_id,handoff_id,conv_id,status,due_at,reason,idempotency_key,created_at,updated_at
                       ) VALUES (?,?,?,'pending',?,?,?,?,?)""",
                    (
                        callback_task_id, handoff_id, conv_id, callback_due_at,
                        callback_reason, f"callback:{handoff_id}", now, now,
                    ),
                )
            event_data = {
                "handoff_id": handoff_id,
                "ack_message_id": ack_message_id,
                "delivery_status": "pending",
                "temporary_summary": summary_payload,
                "temporary_urgency": state["temporary_urgency"],
            }
            cursor = db.execute(
                "INSERT INTO handoff_events(handoff_id,conv_id,event_type,data_json,created_at) VALUES (?,?,?,?,?)",
                (handoff_id, conv_id, "handoff.started", json.dumps(event_data, ensure_ascii=False), now),
            )
            self._append_case_action_on_connection(
                db,
                conv_id,
                action_type="reply",
                description="AI 客服转人工确认已写入会话",
                source_ref=ack_message_id,
                idempotency_key=f"handoff-ack:{handoff_id}",
                status="success",
                handoff_id=handoff_id,
            )
            return {
                "created": True,
                "handoff_id": handoff_id,
                "consumer_message_id": consumer_message_id,
                "ack_message_id": ack_message_id,
                "ack_text": ack_text,
                "source_message_seq": source_message_seq,
                "temporary_summary": summary_payload,
                "temporary_priority": temporary_priority,
                "temporary_priority_score": int(temporary_priority_score),
                "temporary_priority_reasons": reasons,
                "temporary_priority_rule_version": temporary_priority_rule_version,
                "event_id": int(cursor.lastrowid),
                "online_staff": online,
            }

    def is_waiting_human(self, conv_id: str) -> bool:
        """True when a handoff has already taken ownership of the conversation."""
        with self._connect() as db:
            row = db.execute(
                "SELECT status FROM sessions WHERE conv_id=?", (conv_id,)
            ).fetchone()
        return bool(row and row["status"] == "waiting_human")

    def get_handoff_enrichment_input(self, handoff_id: str) -> Optional[Dict[str, Any]]:
        """Read only the latest transfer message plus three preceding turns and stored facts."""
        with self._connect() as db:
            handoff = db.execute(
                """SELECT h.handoff_id,h.conv_id,h.source_message_seq,
                          m.message_id,m.content,b.status AS emotion_block_status,
                          (SELECT status FROM handoff_blocks WHERE handoff_id=h.handoff_id AND block_type='summary') AS summary_block_status,
                          (SELECT status FROM handoff_blocks WHERE handoff_id=h.handoff_id AND block_type='next_steps') AS next_steps_block_status
                   FROM handoff_records h
                   JOIN messages m ON m.conv_id=h.conv_id AND m.rowid=h.source_message_seq
                   LEFT JOIN handoff_blocks b ON b.handoff_id=h.handoff_id AND b.block_type='emotion_urgency'
                   WHERE h.handoff_id=? AND h.status IN ('open','in_progress')""",
                (handoff_id,),
            ).fetchone()
            if handoff is None:
                return None
            prior = db.execute(
                """SELECT rowid AS message_seq,sender_type,content,message_id
                   FROM messages WHERE conv_id=? AND rowid<? AND sender_type IN ('consumer','ai','customer_service')
                   ORDER BY rowid DESC LIMIT 6""",
                (handoff["conv_id"], handoff["source_message_seq"]),
            ).fetchall()
            state_row = db.execute(
                "SELECT state_json FROM case_states WHERE conv_id=?", (handoff["conv_id"],)
            ).fetchone()
            active_intents = db.execute(
                """SELECT DISTINCT intent_id FROM case_sub_requests
                   WHERE conv_id=? AND status IN ('open','in_progress','blocked')
                   ORDER BY intent_id""",
                (handoff["conv_id"],),
            ).fetchall()
        state = json.loads(state_row[0]) if state_row else {}
        history = [
            {
                "message_seq": int(item["message_seq"]),
                "message_id": str(item["message_id"]),
                "role": "user" if item["sender_type"] == "consumer" else "assistant",
                "content": str(item["content"]),
            }
            for item in reversed(prior)
        ]
        return {
            "handoff_id": str(handoff["handoff_id"]),
            "conv_id": str(handoff["conv_id"]),
            "intent": str(state.get("intent") or ""),
            "intents": [str(row["intent_id"]) for row in active_intents],
            "source_message_seq": int(handoff["source_message_seq"]),
            "source_message_id": str(handoff["message_id"]),
            "message": str(handoff["content"]),
            "history": history,
            "emotion_block_status": str(handoff["emotion_block_status"] or "pending"),
            "summary_block_status": str(handoff["summary_block_status"] or "pending"),
            "next_steps_block_status": str(handoff["next_steps_block_status"] or "pending"),
            "intent_enrichment": state.get("handoff_intent_enrichment") or {},
        }

    def get_handoff_refresh_input(self, handoff_id: str) -> Optional[Dict[str, Any]]:
        """读取转人工后尚未处理的消费者消息及其之前的完整可见上下文。"""
        with self._connect() as db:
            handoff = db.execute(
                "SELECT handoff_id,conv_id,source_message_seq,status FROM handoff_records WHERE handoff_id=?",
                (handoff_id,),
            ).fetchone()
            if handoff is None or str(handoff["status"]) not in {"open", "in_progress"}:
                return None
            state_row = db.execute(
                "SELECT state_json FROM case_states WHERE conv_id=?", (handoff["conv_id"],)
            ).fetchone()
            state = json.loads(state_row[0]) if state_row else {}
            initial_cursor = int(
                (state.get("handoff_intent_enrichment") or {}).get("source_message_seq")
                or handoff["source_message_seq"]
                or 0
            )
            cursor = max(initial_cursor, int(state.get("handoff_refresh_message_seq") or 0))
            dialogue = db.execute(
                """SELECT rowid AS message_seq,message_id,sender_type,content
                   FROM messages WHERE conv_id=? AND visible_to_consumer=1
                     AND sender_type IN ('consumer','ai','customer_service')
                   ORDER BY rowid""",
                (handoff["conv_id"],),
            ).fetchall()

        history: list[dict[str, Any]] = []
        new_messages: list[dict[str, Any]] = []
        for row in dialogue:
            sender = str(row["sender_type"])
            item = {
                "message_seq": int(row["message_seq"]),
                "message_id": str(row["message_id"]),
                "role": "user" if sender == "consumer" else "assistant",
                "sender_type": sender,
                "content": str(row["content"]),
            }
            if sender == "consumer" and item["message_seq"] > cursor:
                new_messages.append({**item, "history": list(history)})
            history.append(item)
        result = {
            "handoff_id": str(handoff["handoff_id"]),
            "conv_id": str(handoff["conv_id"]),
            "source_message_seq": int(handoff["source_message_seq"] or 0),
            "last_processed_message_seq": cursor,
            "messages": new_messages,
        }
        return result

    def get_handoff_generation_input(self, handoff_id: str) -> Optional[Dict[str, Any]]:
        """读取完整可见对话与结构化案件事实，供客服交接内容生成使用。"""
        with self._connect() as db:
            handoff = db.execute(
                "SELECT handoff_id,conv_id,reason,source_message_seq,status FROM handoff_records WHERE handoff_id=?",
                (handoff_id,),
            ).fetchone()
            if handoff is None or str(handoff["status"]) not in {"open", "in_progress"}:
                return None
            conv_id = str(handoff["conv_id"])
            state_row = db.execute(
                "SELECT state_json FROM case_states WHERE conv_id=?", (conv_id,)
            ).fetchone()
            request_rows = db.execute(
                """SELECT sub_request_id,sub_request_key,intent_id,status,source_ref,source_message_seq
                   FROM case_sub_requests WHERE conv_id=? ORDER BY created_at,sub_request_id""",
                (conv_id,),
            ).fetchall()
            action_rows = db.execute(
                """SELECT a.action_id,a.sub_request_id,a.action_type,a.description,a.source_ref,a.status,
                          m.rowid AS source_message_seq
                   FROM case_action_logs a
                   LEFT JOIN messages m ON m.conv_id=a.conv_id AND m.message_id=a.source_ref
                   WHERE a.conv_id=? ORDER BY a.created_at,a.action_id""",
                (conv_id,),
            ).fetchall()
            conversation_rows = db.execute(
                """SELECT rowid AS message_seq,sender_type,content
                   FROM messages WHERE conv_id=? AND visible_to_consumer=1
                     AND sender_type IN ('consumer','ai','customer_service')
                   ORDER BY rowid""",
                (conv_id,),
            ).fetchall()
            slot_rows = db.execute(
                """SELECT v.slot_id,v.value,v.display_value,v.source_type,v.source_ref,v.confidence,
                          v.confirmation_status,m.rowid AS source_message_seq
                   FROM slot_values v
                   LEFT JOIN messages m ON m.conv_id=v.conv_id AND m.message_id=v.source_ref
                   WHERE v.conv_id=? ORDER BY v.created_at,v.rowid""",
                (conv_id,),
            ).fetchall()
            callback_row = db.execute(
                "SELECT status,due_at,reason FROM callback_tasks WHERE handoff_id=? ORDER BY created_at DESC LIMIT 1",
                (handoff_id,),
            ).fetchone()
            block_rows = db.execute(
                "SELECT block_type,status,state_hash,payload_json,error_code,latency_ms FROM handoff_blocks WHERE handoff_id=? AND block_type IN ('summary','emotion_urgency','next_steps')",
                (handoff_id,),
            ).fetchall()
        state = json.loads(state_row[0]) if state_row else {}
        slots: Dict[str, Dict[str, Any]] = {}
        for row in slot_rows:
            slots[str(row["slot_id"])] = {
                "value": str(row["value"]),
                "display_value": str(row["display_value"]),
                "source_type": str(row["source_type"]),
                "source_ref": str(row["source_ref"] or ""),
                "confidence": float(row["confidence"] or 0.0),
                "confirmation_status": str(row["confirmation_status"]),
                "source_message_seq": int(row["source_message_seq"]) if row["source_message_seq"] is not None else None,
            }
        ticket = state.get("ticket_draft") if isinstance(state.get("ticket_draft"), dict) else {}
        ticket = dict(ticket)
        ticket.setdefault("version", int(state.get("ticket_version", 0) or 0))
        conflicts = state.get("slot_conflicts") if isinstance(state.get("slot_conflicts"), list) else []
        return {
            "handoff_id": str(handoff["handoff_id"]),
            "conv_id": conv_id,
            "source_message_seq": int(handoff["source_message_seq"]) if handoff["source_message_seq"] is not None else None,
            "handoff_reason": str(handoff["reason"] or ""),
            "intent_enrichment": state.get("handoff_intent_enrichment") or {},
            "sub_requests": [dict(row) for row in request_rows],
            "action_logs": [dict(row) for row in action_rows],
            "conversation_messages": [
                {
                    "message_seq": int(row["message_seq"]),
                    "role": {
                        "consumer": "消费者", "ai": "AI客服", "customer_service": "人工客服",
                    }[str(row["sender_type"])],
                    "content": str(row["content"]),
                }
                for row in conversation_rows
            ],
            "slots": slots,
            "ticket": ticket,
            "case_facts": state.get("case_facts") or {},
            "callback_task": dict(callback_row) if callback_row else None,
            "content_blocks": {
                str(row["block_type"]): {
                    "status": str(row["status"]), "state_hash": str(row["state_hash"] or ""),
                    "payload": json.loads(row["payload_json"] or "{}"),
                    "error_code": str(row["error_code"] or ""), "latency_ms": row["latency_ms"],
                }
                for row in block_rows
            },
            "emotion_urgency": next(
                (json.loads(row["payload_json"] or "{}") for row in block_rows if str(row["block_type"]) == "emotion_urgency"),
                {},
            ),
            "urgency": {
                "value": state.get("urgency"),
                "temporary": state.get("temporary_urgency"),
                "final": state.get("final_urgency"),
            },
            "contradictions": [
                {
                    key: item[key]
                    for key in ("field", "field_id", "values", "source_message_seqs")
                    if key in item and isinstance(item[key], (str, int, float, list, tuple))
                }
                for item in conflicts if isinstance(item, dict)
            ],
            "action_log_version": self.get_case_action_log_version(conv_id),
        }

    def save_handoff_generated_block(
        self,
        handoff_id: str,
        *,
        block_type: str,
        status: str,
        state_hash: str,
        payload: Dict[str, Any],
        error_code: str = "",
        latency_ms: float = 0.0,
        source_message_seq: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Persist one independently generated handoff block and its event atomically."""
        if block_type not in {"summary", "next_steps"}:
            raise ValueError("handoff content block type must be summary or next_steps")
        if status not in {"ready", "degraded", "failed"}:
            raise ValueError("invalid generated handoff block status")
        now = _now()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            handoff = db.execute(
                "SELECT conv_id,status FROM handoff_records WHERE handoff_id=?", (handoff_id,)
            ).fetchone()
            if handoff is None or str(handoff["status"]) not in {"open", "in_progress"}:
                return {"status": "stale"}
            existing = db.execute(
                "SELECT status,state_hash,payload_json FROM handoff_blocks WHERE handoff_id=? AND block_type=?",
                (handoff_id, block_type),
            ).fetchone()
            if existing and str(existing["state_hash"] or "") == str(state_hash) and str(existing["status"]) in {"ready", "degraded"}:
                return {
                    "status": str(existing["status"]),
                    "payload": json.loads(existing["payload_json"] or "{}"),
                    "cached": True,
                }
            db.execute(
                """INSERT INTO handoff_blocks(
                       handoff_id,block_type,status,state_hash,payload_json,error_code,latency_ms,
                       source_message_seq,generated_at,updated_at
                   ) VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(handoff_id,block_type) DO UPDATE SET
                       status=excluded.status,state_hash=excluded.state_hash,payload_json=excluded.payload_json,
                       error_code=excluded.error_code,latency_ms=excluded.latency_ms,
                       source_message_seq=excluded.source_message_seq,generated_at=excluded.generated_at,
                       updated_at=excluded.updated_at""",
                (
                    handoff_id, block_type, status, state_hash,
                    json.dumps(payload, ensure_ascii=False), str(error_code or ""),
                    max(0.0, float(latency_ms)), source_message_seq, now, now,
                ),
            )
            event_payload = {
                "handoff_id": handoff_id,
                "block_type": block_type,
                "status": status,
                "state_hash": state_hash,
                "payload": payload,
                "error_code": str(error_code or ""),
                "latency_ms": max(0.0, float(latency_ms)),
            }
            db.execute(
                "INSERT INTO handoff_events(handoff_id,conv_id,event_type,data_json,created_at) VALUES (?,?,?,?,?)",
                (
                    handoff_id, str(handoff["conv_id"]), "handoff.block.updated",
                    json.dumps(event_payload, ensure_ascii=False), now,
                ),
            )
        return {"status": status, "payload": payload, "latency_ms": max(0.0, float(latency_ms))}

    def list_pending_handoff_enrichments(self) -> list[Dict[str, str]]:
        """Return durable open handoffs whose intent facts or emotion block need enrichment."""
        with self._connect() as db:
            rows = db.execute(
                """SELECT h.handoff_id,h.conv_id,b.status AS emotion_status,
                          (SELECT status FROM handoff_blocks WHERE handoff_id=h.handoff_id AND block_type='summary') AS summary_status,
                          (SELECT status FROM handoff_blocks WHERE handoff_id=h.handoff_id AND block_type='next_steps') AS next_steps_status,
                          s.state_json
                   FROM handoff_records h
                   JOIN case_states s ON s.conv_id=h.conv_id
                   LEFT JOIN handoff_blocks b ON b.handoff_id=h.handoff_id AND b.block_type='emotion_urgency'
                   WHERE h.status IN ('open','in_progress')"""
            ).fetchall()
        pending = []
        for row in rows:
            state = json.loads(row["state_json"] or "{}")
            intent_status = str((state.get("handoff_intent_enrichment") or {}).get("status") or "pending")
            if (
                str(row["emotion_status"] or "pending") == "pending"
                or intent_status == "pending"
                or str(row["summary_status"] or "pending") == "pending"
                or str(row["next_steps_status"] or "pending") == "pending"
            ):
                pending.append({"handoff_id": str(row["handoff_id"]), "conv_id": str(row["conv_id"])})
        return pending

    def save_handoff_intent_enrichment(
        self,
        handoff_id: str,
        *,
        source_message_seq: int,
        source_message_id: str,
        intent_id: str,
        confidence: float,
        intent_group: str,
        secondary_intents: list[str],
        entities: Dict[str, Any],
        sub_requests: list[Dict[str, Any]],
        contradictions: Optional[list[Dict[str, Any]]] = None,
        latency_ms: float = 0.0,
        diagnostics: Optional[Dict[str, Any]] = None,
        advance_refresh_cursor: bool = False,
    ) -> Dict[str, Any]:
        """Idempotently persist last-turn intent/slots as conversation facts, never business operations."""
        handoff = self._handoff_context_for_update(handoff_id, source_message_seq)
        if handoff is None:
            return {"status": "stale", "source_message_seq": source_message_seq}
        conv_id = str(handoff["conv_id"])
        source_ref = str(source_message_id)
        synced_sub_requests = self.sync_case_sub_requests(
            conv_id,
            sub_requests,
            source_ref=source_ref,
            source_message_seq=source_message_seq,
        )
        confidence_value = max(0.0, min(1.0, float(confidence or 0.0)))
        now = _now()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT state_json FROM case_states WHERE conv_id=?", (conv_id,)
            ).fetchone()
            state = json.loads(row[0]) if row else {}
            existing_enrichment = state.get("handoff_intent_enrichment") or {}
            if (
                existing_enrichment.get("status") in {"ready", "degraded"}
                and int(existing_enrichment.get("source_message_seq") or -1) == source_message_seq
            ):
                return {"status": str(existing_enrichment["status"]), "cached": True}
            case_facts = state.get("case_facts") if isinstance(state.get("case_facts"), dict) else {"schema_version": 1, "facts": {}}
            facts = case_facts.get("facts") if isinstance(case_facts.get("facts"), dict) else {}

            def append_fact(field: str, value: Any, source_type: str, fact_confidence: float) -> None:
                if value is None or value == "":
                    return
                records = facts.setdefault(field, [])
                fact = {
                    "value": value,
                    "source_type": source_type,
                    "source_ref": source_ref,
                    "confidence": max(0.0, min(1.0, float(fact_confidence))),
                    "confirmation_status": "unconfirmed",
                }
                if fact not in records:
                    records.append(fact)

            append_fact("intent", intent_id, "intent_recognizer", confidence_value)
            for slot_id, raw_values in (entities or {}).items():
                values = raw_values if isinstance(raw_values, (list, tuple, set)) else [raw_values]
                for index, raw_value in enumerate(values):
                    if raw_value is None or isinstance(raw_value, (dict, list, tuple, set)) or not str(raw_value).strip():
                        continue
                    value = str(raw_value).strip()
                    append_fact(str(slot_id), value, "message", confidence_value)
                    slot_idempotency = f"handoff-slot:{handoff_id}:{source_message_seq}:{slot_id}:{index}"
                    slot_value_id = uuid.uuid5(uuid.NAMESPACE_URL, slot_idempotency).hex
                    db.execute(
                        """INSERT INTO slot_values(
                               slot_value_id,conv_id,slot_id,value,display_value,source_type,source_ref,
                               confidence,confirmation_status,created_at
                           ) VALUES (?,?,?,?,?,'handoff_intent',?,?,?,?)
                           ON CONFLICT(slot_value_id) DO UPDATE SET value=excluded.value,
                               display_value=excluded.display_value,confidence=excluded.confidence,
                               confirmation_status=excluded.confirmation_status""",
                        (
                            slot_value_id, conv_id, str(slot_id), value, value, source_ref,
                            confidence_value, "unconfirmed", now,
                        ),
                    )
            case_facts["schema_version"] = int(case_facts.get("schema_version", 1) or 1)
            case_facts["facts"] = facts
            intent_payload = {
                "status": "ready",
                "intent": str(intent_id),
                "intent_group": str(intent_group or "other"),
                "confidence": confidence_value,
                "secondary_intents": list(dict.fromkeys(str(item) for item in secondary_intents if item)),
                "sub_requests": synced_sub_requests,
                "source_message_seq": int(source_message_seq),
                "source_message_id": source_ref,
                "latency_ms": max(0.0, float(latency_ms)),
                "diagnostics": dict(diagnostics or {}),
                "updated_at": now,
            }
            active_sub_requests = [
                dict(item)
                for item in db.execute(
                    "SELECT * FROM case_sub_requests WHERE conv_id=? AND status!='resolved' ORDER BY created_at,sub_request_id",
                    (conv_id,),
                ).fetchall()
            ]
            state.update({
                "secondary_intents": intent_payload["secondary_intents"],
                "business_sub_requests": active_sub_requests,
                "case_facts": case_facts,
                "handoff_intent_enrichment": intent_payload,
                "slot_conflicts": [
                    {
                        "field_id": str(item.get("field_id") or "")[:60],
                        "values": [str(value)[:80] for value in item.get("values", [])[:4] if isinstance(value, (str, int, float))],
                        "source_message_seqs": [int(value) for value in item.get("source_message_seqs", [])[:4] if isinstance(value, int)],
                    }
                    for item in (contradictions or []) if isinstance(item, dict)
                ],
            })
            if str(intent_id) not in {"human_handoff", "other"}:
                state["intent"] = str(intent_id)
                state["intent_group"] = str(intent_group or "other")
            if advance_refresh_cursor:
                state["handoff_refresh_message_seq"] = max(
                    int(state.get("handoff_refresh_message_seq") or 0), int(source_message_seq)
                )
            state["version"] = int(state.get("version", 0) or 0) + 1
            db.execute(
                "INSERT INTO case_states(conv_id,state_json,updated_at) VALUES (?,?,?) "
                "ON CONFLICT(conv_id) DO UPDATE SET state_json=excluded.state_json,updated_at=excluded.updated_at",
                (conv_id, json.dumps(state, ensure_ascii=False), now),
            )
            self._append_case_action_on_connection(
                db,
                conv_id,
                action_type="system",
                description="转人工后台已补齐本轮意图和字段来源",
                source_ref=source_ref,
                idempotency_key=f"handoff-intent:{handoff_id}:{source_message_seq}",
                status="success",
                handoff_id=handoff_id,
            )
        return {"status": "ready", "source_message_seq": source_message_seq}

    def save_handoff_intent_failure(
        self, handoff_id: str, *, source_message_seq: int, error_code: str = "intent_enrichment_failed"
    ) -> Dict[str, Any]:
        handoff = self._handoff_context_for_update(handoff_id, source_message_seq)
        if handoff is None:
            return {"status": "stale"}
        conv_id = str(handoff["conv_id"])
        source_ref = str(handoff["message_id"])
        now = _now()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT state_json FROM case_states WHERE conv_id=?", (conv_id,)).fetchone()
            state = json.loads(row[0]) if row else {}
            state["handoff_intent_enrichment"] = {
                "status": "failed", "error_code": str(error_code),
                "source_message_seq": int(source_message_seq), "updated_at": now,
            }
            state["version"] = int(state.get("version", 0) or 0) + 1
            db.execute(
                "INSERT INTO case_states(conv_id,state_json,updated_at) VALUES (?,?,?) "
                "ON CONFLICT(conv_id) DO UPDATE SET state_json=excluded.state_json,updated_at=excluded.updated_at",
                (conv_id, json.dumps(state, ensure_ascii=False), now),
            )
            self._append_case_action_on_connection(
                db, conv_id, action_type="system", description="转人工后台意图字段补齐失败",
                source_ref=source_ref, idempotency_key=f"handoff-intent:{handoff_id}:{source_message_seq}",
                status="failed", handoff_id=handoff_id,
            )
        return {"status": "failed", "error_code": str(error_code)}

    def save_handoff_emotion_enrichment(
        self,
        handoff_id: str,
        *,
        source_message_seq: int,
        emotion_result: Dict[str, Any],
        final_urgency: Dict[str, Any],
        state_hash: str,
        latency_ms: float,
    ) -> Dict[str, Any]:
        """Atomically persist the final emotion/urgency block and its push event."""
        handoff = self._handoff_context_for_update(handoff_id, source_message_seq)
        if handoff is None:
            return {"status": "stale"}
        conv_id = str(handoff["conv_id"])
        source_ref = str(handoff["message_id"])
        now = _now()
        block_status = "degraded" if emotion_result.get("degraded") else "ready"
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            prior_block = db.execute(
                "SELECT status,state_hash,payload_json FROM handoff_blocks WHERE handoff_id=? AND block_type='emotion_urgency'",
                (handoff_id,),
            ).fetchone()
            if prior_block and str(prior_block["state_hash"] or "") == state_hash and str(prior_block["status"]) in {"ready", "degraded"}:
                return {"status": str(prior_block["status"]), "cached": True}
            record = db.execute(
                "SELECT temporary_priority,temporary_priority_score,temporary_priority_reasons_json FROM handoff_records WHERE handoff_id=?",
                (handoff_id,),
            ).fetchone()
            if record is None:
                return {"status": "stale"}
            temporary = {
                "value": str(record["temporary_priority"]),
                "score": int(record["temporary_priority_score"]),
                "reasons": json.loads(record["temporary_priority_reasons_json"] or "[]"),
                "source_message_seq": int(source_message_seq),
            }
            final = {
                "value": str(final_urgency["priority"]),
                "score": int(final_urgency["score"]),
                "reasons": list(final_urgency.get("reasons") or []),
                "rule_version": str(final_urgency.get("rule_version") or ""),
                "source_message_seq": int(source_message_seq),
                "status": "final",
            }
            payload = {
                "emotion": {
                    "value": str(emotion_result.get("emotion") or "平静"),
                    "answers": emotion_result.get("answers") or {},
                    "evidence": emotion_result.get("evidence") or {},
                    "source": str(emotion_result.get("source") or "fallback_rules"),
                    "degraded": bool(emotion_result.get("degraded")),
                    "degradation_reason": str(emotion_result.get("degradation_reason") or ""),
                    "rule_version": str(emotion_result.get("rule_version") or ""),
                    "source_message_seq": int(source_message_seq),
                },
                "temporary": temporary,
                "final": final,
            }
            state_row = db.execute("SELECT state_json FROM case_states WHERE conv_id=?", (conv_id,)).fetchone()
            state = json.loads(state_row[0]) if state_row else {}
            case_facts = state.get("case_facts") if isinstance(state.get("case_facts"), dict) else {"schema_version": 1, "facts": {}}
            facts = case_facts.get("facts") if isinstance(case_facts.get("facts"), dict) else {}
            emotion = payload["emotion"]["value"]

            def append_fact(field: str, value: Any) -> None:
                if value is None or value == "":
                    return
                records = facts.setdefault(field, [])
                fact = {
                    "value": value,
                    "source_type": "emotion_llm" if emotion_result.get("source") == "llm" else "emotion_fallback",
                    "source_ref": source_ref,
                    "confidence": 1.0 if emotion_result.get("source") == "llm" else 0.5,
                    "confirmation_status": "inferred",
                }
                if fact not in records:
                    records.append(fact)

            append_fact("emotion", emotion)
            append_fact("emotion_answers", payload["emotion"]["answers"])
            for quote in dict.fromkeys(str(item) for item in payload["emotion"]["evidence"].values() if item):
                append_fact("emotion_evidence", quote)
            case_facts["schema_version"] = int(case_facts.get("schema_version", 1) or 1)
            case_facts["facts"] = facts
            summary = state.get("handoff_summary") if isinstance(state.get("handoff_summary"), dict) else {}
            summary.update({
                "emotion": emotion,
                "emotion_source": payload["emotion"]["source"],
                "emotion_evidence": list(payload["emotion"]["evidence"].values()),
                "urgency": final["value"],
                "urgency_reasons": final["reasons"],
                "final_urgency": final,
            })
            state.update({
                "case_facts": case_facts,
                "emotion": emotion,
                "emotion_llm": payload["emotion"],
                "emotion_source": payload["emotion"]["source"],
                "emotion_degraded": bool(emotion_result.get("degraded")),
                "temporary_urgency": state.get("temporary_urgency") or temporary,
                "final_urgency": final,
                "urgency": final["value"],
                "handoff_summary": summary,
            })
            state["version"] = int(state.get("version", 0) or 0) + 1
            db.execute(
                "INSERT INTO case_states(conv_id,state_json,updated_at) VALUES (?,?,?) "
                "ON CONFLICT(conv_id) DO UPDATE SET state_json=excluded.state_json,updated_at=excluded.updated_at",
                (conv_id, json.dumps(state, ensure_ascii=False), now),
            )
            db.execute(
                "UPDATE handoff_records SET final_priority=?,final_priority_score=?,final_priority_reasons_json=?,updated_at=? WHERE handoff_id=?",
                (final["value"], final["score"], json.dumps(final["reasons"], ensure_ascii=False), now, handoff_id),
            )
            db.execute(
                """UPDATE handoff_blocks SET status=?,state_hash=?,payload_json=?,error_code=?,latency_ms=?,
                       source_message_seq=?,generated_at=?,updated_at=? WHERE handoff_id=? AND block_type='emotion_urgency'""",
                (
                    block_status, state_hash, json.dumps(payload, ensure_ascii=False),
                    "emotion_fallback" if emotion_result.get("degraded") else "",
                    max(0.0, float(latency_ms)), source_message_seq, now, now, handoff_id,
                ),
            )
            event_data = {
                "handoff_id": handoff_id,
                "block_type": "emotion_urgency",
                "status": block_status,
                "state_hash": state_hash,
                "payload": payload,
                "error_code": "emotion_fallback" if emotion_result.get("degraded") else "",
                "latency_ms": max(0.0, float(latency_ms)),
            }
            db.execute(
                "INSERT INTO handoff_events(handoff_id,conv_id,event_type,data_json,created_at) VALUES (?,?,?,?,?)",
                (handoff_id, conv_id, "handoff.block.updated", json.dumps(event_data, ensure_ascii=False), now),
            )
            self._append_case_action_on_connection(
                db, conv_id, action_type="system",
                description="转人工后台已更新情绪与最终紧急度" if block_status == "ready" else "转人工情绪采用规则降级结果",
                source_ref=source_ref,
                idempotency_key=f"handoff-emotion:{handoff_id}:{state_hash}",
                status="success", handoff_id=handoff_id,
            )
        return {"status": block_status, "payload": payload, "latency_ms": max(0.0, float(latency_ms))}

    def save_handoff_block_failure(
        self,
        handoff_id: str,
        *,
        block_type: str,
        source_message_seq: int,
        state_hash: str,
        error_code: str,
        latency_ms: float,
    ) -> Dict[str, Any]:
        """Persist an independently failed block without erasing its temporary payload."""
        if block_type not in {"summary", "emotion_urgency", "next_steps"}:
            raise ValueError("invalid handoff block type")
        handoff = self._handoff_context_for_update(handoff_id, source_message_seq)
        if handoff is None:
            return {"status": "stale"}
        conv_id = str(handoff["conv_id"])
        now = _now()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT status,state_hash,payload_json FROM handoff_blocks WHERE handoff_id=? AND block_type=?",
                (handoff_id, block_type),
            ).fetchone()
            if existing and str(existing["status"]) in {"ready", "degraded"} and str(existing["state_hash"] or "") == state_hash:
                return {"status": str(existing["status"]), "payload": json.loads(existing["payload_json"] or "{}")}
            payload = json.loads(existing["payload_json"] or "{}") if existing else {}
            db.execute(
                """INSERT INTO handoff_blocks(
                       handoff_id,block_type,status,state_hash,payload_json,error_code,latency_ms,
                       source_message_seq,generated_at,updated_at
                   ) VALUES (?,?, 'failed', ?,?,?,?,?,?,?)
                   ON CONFLICT(handoff_id,block_type) DO UPDATE SET
                       status='failed',state_hash=excluded.state_hash,payload_json=excluded.payload_json,
                       error_code=excluded.error_code,latency_ms=excluded.latency_ms,
                       source_message_seq=excluded.source_message_seq,generated_at=excluded.generated_at,
                       updated_at=excluded.updated_at""",
                (
                    handoff_id, block_type, state_hash, json.dumps(payload, ensure_ascii=False),
                    str(error_code or "unknown"), max(0.0, float(latency_ms)), source_message_seq, now, now,
                ),
            )
            event_data = {
                "handoff_id": handoff_id, "block_type": block_type, "status": "failed",
                "state_hash": state_hash, "payload": payload, "error_code": str(error_code or "unknown"),
                "latency_ms": max(0.0, float(latency_ms)),
            }
            db.execute(
                "INSERT INTO handoff_events(handoff_id,conv_id,event_type,data_json,created_at) VALUES (?,?,?,?,?)",
                (handoff_id, conv_id, "handoff.block.updated", json.dumps(event_data, ensure_ascii=False), now),
            )
        return {"status": "failed", "payload": payload, "latency_ms": max(0.0, float(latency_ms))}

    def _handoff_context_for_update(self, handoff_id: str, source_message_seq: int) -> Optional[Dict[str, Any]]:
        with self._connect() as db:
            row = db.execute(
                """SELECT h.conv_id,h.status,h.source_message_seq,m.message_id
                   FROM handoff_records h JOIN messages m
                     ON m.conv_id=h.conv_id AND m.rowid=? AND m.sender_type='consumer'
                   WHERE h.handoff_id=?""",
                (source_message_seq, handoff_id),
            ).fetchone()
        if row is None:
            return None
        if str(row["status"]) not in {"open", "in_progress"}:
            return None
        return dict(row)

    def update_case_state(self, conv_id: str, updates: Dict[str, Any], expected_version: Optional[int] = None) -> Dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT state_json FROM case_states WHERE conv_id=?", (conv_id,)).fetchone()
            state = json.loads(row[0]) if row else {}
            current_version = int(state.get("version", 0) or 0)
            if expected_version is not None and current_version != int(expected_version):
                raise ValueError(f"case_state version conflict: expected {expected_version}, actual {current_version}")
            state.update(updates)
            state["version"] = current_version + 1
            db.execute(
                "INSERT INTO case_states(conv_id,state_json,updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(conv_id) DO UPDATE SET state_json=excluded.state_json,updated_at=excluded.updated_at",
                (conv_id, json.dumps(state, ensure_ascii=False), _now()),
            )
            if updates.get("status") == "waiting_human":
                db.execute(
                    "UPDATE sessions SET status='waiting_human',handler='human',updated_at=? WHERE conv_id=?",
                    (_now(), conv_id),
                )
        return state

    def update_emotion_analysis(self, conv_id: str, request_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """原子更新异步情绪结论、CaseFacts 与客服摘要，避免覆盖并发会话更新。"""
        with self._connect() as db:
            row = db.execute("SELECT state_json FROM case_states WHERE conv_id=?", (conv_id,)).fetchone()
            state = json.loads(row[0]) if row else {"version": 0}
            if state.get("emotion_analysis_request_id") != request_id:
                # 只允许最新一轮请求的异步结果更新客服侧当前情绪，避免旧任务晚返回覆盖新状态。
                return state
            case_facts = state.get("case_facts") if isinstance(state.get("case_facts"), dict) else {}
            facts = case_facts.get("facts") if isinstance(case_facts.get("facts"), dict) else {}

            def append_fact(field: str, value: Any) -> None:
                if value is None or value == "":
                    return
                records = facts.setdefault(field, [])
                record = {
                    "value": value,
                    "source_type": "emotion_llm" if result.get("source") == "llm" else "emotion_fallback",
                    "source_ref": request_id,
                    "confidence": 1.0 if result.get("source") == "llm" else 0.5,
                    "confirmation_status": "inferred",
                }
                if record not in records:
                    records.append(record)

            emotion = str(result.get("emotion") or state.get("emotion") or "平静")
            append_fact("emotion", emotion)
            answers = result.get("answers") if isinstance(result.get("answers"), dict) else {}
            append_fact("emotion_answers", answers)
            evidence = result.get("evidence") if isinstance(result.get("evidence"), dict) else {}
            evidence_values = [
                str(item.get("evidence", "") if isinstance(item, dict) else item).strip()
                for item in evidence.values()
                if (isinstance(item, dict) and item.get("evidence")) or (isinstance(item, str) and item.strip())
            ]
            for item in dict.fromkeys(evidence_values):
                append_fact("emotion_evidence", item)

            case_facts["schema_version"] = int(case_facts.get("schema_version", 1) or 1)
            case_facts["facts"] = facts
            state.update({
                "case_facts": case_facts,
                "emotion": emotion,
                "emotion_llm": result,
                "emotion_source": result.get("source", "fallback"),
                "emotion_degraded": bool(result.get("degraded", False)),
            })
            summary = state.get("handoff_summary")
            if isinstance(summary, dict):
                summary["emotion"] = emotion
                summary["emotion_source"] = state["emotion_source"]
                summary["emotion_evidence"] = list(dict.fromkeys(evidence_values))
                state["handoff_summary"] = summary
            state["version"] = int(state.get("version", 0) or 0) + 1
            db.execute(
                "INSERT INTO case_states(conv_id,state_json,updated_at) VALUES (?,?,?) "
                "ON CONFLICT(conv_id) DO UPDATE SET state_json=excluded.state_json,updated_at=excluded.updated_at",
                (conv_id, json.dumps(state, ensure_ascii=False), _now()),
            )
        return state

    def append_audit(self, conv_id: str, actor_id: str, action: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        item = {"audit_id": uuid.uuid4().hex, "conv_id": conv_id, "actor_id": actor_id, "action": action, "details": details or {}, "created_at": _now()}
        with self._connect() as db:
            db.execute("INSERT INTO audit_logs(audit_id,conv_id,actor_id,action,details_json,created_at) VALUES (?,?,?,?,?,?)", (item["audit_id"], conv_id, actor_id, action, json.dumps(item["details"], ensure_ascii=False), item["created_at"]))
        return item

    def record_slot_value(self, conv_id: str, slot_id: str, value: str, display_value: str, *, source_type: str, source_ref: str = "", confidence: float = 1.0, confirmation_status: str = "confirmed") -> Dict[str, Any]:
        item = {"slot_value_id": uuid.uuid4().hex, "conv_id": conv_id, "slot_id": slot_id, "value": value, "display_value": display_value, "source_type": source_type, "source_ref": source_ref, "confidence": float(confidence), "confirmation_status": confirmation_status, "created_at": _now()}
        with self._connect() as db:
            db.execute("INSERT INTO slot_values(slot_value_id,conv_id,slot_id,value,display_value,source_type,source_ref,confidence,confirmation_status,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)", tuple(item.values()))
        return item

    def record_risk_event(self, user_id: str, conv_id: str, level: str, risk_type: str, reason: str) -> Dict[str, Any]:
        item = {"risk_id": uuid.uuid4().hex, "user_id": user_id, "conv_id": conv_id, "level": level, "risk_type": risk_type, "reason": reason, "status": "new", "created_at": _now()}
        with self._connect() as db:
            db.execute("INSERT INTO risk_events(risk_id,user_id,conv_id,level,risk_type,reason,status,created_at) VALUES (?,?,?,?,?,?,?,?)", tuple(item.values()))
        return item

    def append_stream_event(self, conv_id: str, event_type: str, data: Dict[str, Any]) -> int:
        with self._connect() as db:
            cursor = db.execute(
                "INSERT INTO stream_events(conv_id,event_type,data_json,created_at) VALUES (?,?,?,?)",
                (conv_id, event_type, json.dumps(data, ensure_ascii=False), _now()),
            )
            return int(cursor.lastrowid)

    def list_stream_events(self, conv_id: str, after_id: int = 0) -> list[Dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT event_id,event_type,data_json FROM stream_events WHERE conv_id=? AND event_id>? ORDER BY event_id",
                (conv_id, int(after_id)),
            ).fetchall()
        return [{"id": int(row["event_id"]), "event": row["event_type"], "data": json.loads(row["data_json"])} for row in rows]

    def append_short_memory(self, conv_id: str, role: str, content: str) -> None:
        with self._connect() as db:
            db.execute("INSERT INTO memory_short_term(memory_id,conv_id,role,content,created_at) VALUES (?,?,?,?,?)", (uuid.uuid4().hex, conv_id, role, content, _now()))
            db.execute("DELETE FROM memory_short_term WHERE conv_id=? AND memory_id NOT IN (SELECT memory_id FROM memory_short_term WHERE conv_id=? ORDER BY created_at DESC LIMIT 10)", (conv_id, conv_id))

    def append_long_memory(self, user_id: str, conv_id: str, summary: Dict[str, Any]) -> None:
        with self._connect() as db:
            db.execute("INSERT INTO memory_long_term(memory_id,user_id,conv_id,summary,created_at) VALUES (?,?,?,?,?)", (uuid.uuid4().hex, user_id, conv_id, json.dumps(summary, ensure_ascii=False), _now()))

    def upsert_profile(self, user_id: str, profile: Dict[str, Any]) -> None:
        with self._connect() as db:
            db.execute("INSERT INTO user_profiles(user_id,profile_json,updated_at) VALUES (?,?,?) ON CONFLICT(user_id) DO UPDATE SET profile_json=excluded.profile_json,updated_at=excluded.updated_at", (user_id, json.dumps(profile, ensure_ascii=False), _now()))

    def record_observability(self, trace_id: str, conv_id: str, phases: Dict[str, Any], cache_hits: Dict[str, Any], degradation_reason: str = "") -> None:
        with self._connect() as db:
            db.execute("INSERT OR REPLACE INTO observability_events(trace_id,conv_id,phases_json,cache_hits_json,degradation_reason,created_at) VALUES (?,?,?,?,?,?)", (trace_id, conv_id, json.dumps(phases, ensure_ascii=False), json.dumps(cache_hits, ensure_ascii=False), degradation_reason, _now()))

    def get_conversation(
        self,
        conv_id: str,
        *,
        viewer_role: str,
        after_message_seq: Optional[int] = None,
    ) -> Dict[str, Any]:
        with self._connect() as db:
            session = db.execute("SELECT * FROM sessions WHERE conv_id=?", (conv_id,)).fetchone()
            if after_message_seq is not None and int(after_message_seq) < 0:
                raise ValueError("after_message_seq must be non-negative")
            if after_message_seq is None:
                message_rows = db.execute(
                    "SELECT rowid AS message_seq,* FROM messages WHERE conv_id=? ORDER BY rowid",
                    (conv_id,),
                ).fetchall()
            else:
                message_rows = db.execute(
                    "SELECT rowid AS message_seq,* FROM messages WHERE conv_id=? AND rowid>?"
                    " ORDER BY rowid LIMIT 100",
                    (conv_id, int(after_message_seq)),
                ).fetchall()
            messages = message_rows
            if viewer_role == "consumer":
                messages = [
                    row for row in message_rows
                    if bool(row["visible_to_consumer"])
                    and str(row["sender_type"]) in {"consumer", "ai", "customer_service"}
                ]
            state = db.execute("SELECT state_json FROM case_states WHERE conv_id=?", (conv_id,)).fetchone()
            handoff = db.execute(
                "SELECT handoff_id,reason,source_message_seq,status,created_at FROM handoff_records "
                "WHERE conv_id=? AND status IN ('open','in_progress') ORDER BY created_at DESC LIMIT 1",
                (conv_id,),
            ).fetchone()
            handoff_event_cursor = db.execute(
                "SELECT COALESCE(MAX(event_id),0) FROM handoff_events WHERE conv_id=?", (conv_id,)
            ).fetchone()[0]
            message_cursor = max(
                (int(row["message_seq"]) for row in message_rows),
                default=max(0, int(after_message_seq or 0)),
            )
            handoff_blocks = db.execute(
                "SELECT block_type,status,state_hash,payload_json,error_code,latency_ms,source_message_seq,updated_at "
                "FROM handoff_blocks WHERE handoff_id=?",
                (handoff["handoff_id"],),
            ).fetchall() if handoff else []
        result: Dict[str, Any] = {
            "conv_id": conv_id,
            "user_id": session["user_id"] if session else "",
            "status": session["status"] if session else "",
            "handler": session["handler"] if session else "ai",
            "messages": [dict(item) for item in messages],
        }
        if viewer_role == "agent":
            result["case_state"] = json.loads(state[0]) if state else {}
            result["handoff_event_cursor"] = int(handoff_event_cursor or 0)
            if handoff:
                result["handoff"] = {
                    "handoff_id": str(handoff["handoff_id"]),
                    "reason": str(handoff["reason"] or ""),
                    "status": str(handoff["status"]),
                    "source_message_seq": int(handoff["source_message_seq"] or 0),
                    "last_event_id": int(handoff_event_cursor or 0),
                    "blocks": {
                        str(row["block_type"]): {
                            "status": str(row["status"]),
                            "state_hash": str(row["state_hash"] or ""),
                            "payload": json.loads(row["payload_json"] or "{}"),
                            "error_code": str(row["error_code"] or ""),
                            "latency_ms": row["latency_ms"],
                            "source_message_seq": row["source_message_seq"],
                            "updated_at": row["updated_at"],
                        }
                        for row in handoff_blocks
                    },
                }
        elif viewer_role == "consumer":
            result["last_message_seq"] = int(message_cursor or 0)
        return result

    def list_handoff_events(self, conv_id: str, *, after_event_id: int = 0, limit: int = 100) -> list[Dict[str, Any]]:
        """Read ordered handoff events strictly after the caller's persisted SSE cursor."""
        if int(after_event_id) < 0:
            raise ValueError("after_event_id must be non-negative")
        bounded_limit = max(1, min(int(limit), 500))
        with self._connect() as db:
            rows = db.execute(
                "SELECT event_id,handoff_id,event_type,data_json,created_at FROM handoff_events "
                "WHERE conv_id=? AND event_id>? ORDER BY event_id LIMIT ?",
                (conv_id, int(after_event_id), bounded_limit),
            ).fetchall()
        return [
            {
                "event_id": int(row["event_id"]),
                "handoff_id": str(row["handoff_id"]),
                "event_type": str(row["event_type"]),
                "data": json.loads(row["data_json"] or "{}"),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]

    def create_ticket(
        self, user_id: str, conv_id: str, main_type: str, label: str, items: list[str]
    ) -> Dict[str, Any]:
        ticket = {
            "ticket_id": f"T-{uuid.uuid4().hex[:8]}",
            "user_id": user_id,
            "conv_id": conv_id,
            "main_type": main_type,
            "buyer_nickname": "",
            "shop": "ShopCare",
            "status": "draft",
            "items": list(dict.fromkeys([*items, main_type])),
            "created_at": _now(),
            "updated_at": _now(),
            "completed_at": None,
        }
        with self._connect() as db:
            user = db.execute("SELECT display_name FROM users WHERE user_id=?", (user_id,)).fetchone()
            ticket["buyer_nickname"] = user["display_name"] if user else user_id
            try:
                db.execute(
                    "INSERT INTO tickets(ticket_id,user_id,conv_id,buyer_nickname,shop,main_type,status,items_json,created_at,updated_at,completed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (ticket["ticket_id"], user_id, conv_id, ticket["buyer_nickname"], ticket["shop"], main_type, "draft", json.dumps(ticket["items"]), ticket["created_at"], ticket["updated_at"], ticket["completed_at"]),
                )
                if handoff_enabled():
                    self._append_case_action_on_connection(
                        db, conv_id, action_type="ticket", description="工单草稿已创建",
                        source_ref=ticket["ticket_id"],
                        idempotency_key=f"ticket:create:{ticket['ticket_id']}", status="success",
                    )
            except sqlite3.IntegrityError as exc:
                if "idx_one_open_ticket" not in str(exc) and "UNIQUE constraint failed: tickets.user_id" not in str(exc):
                    raise
                existing = db.execute(
                    "SELECT * FROM tickets WHERE user_id=? AND status NOT IN ('closed','done','rejected') ORDER BY created_at LIMIT 1",
                    (user_id,),
                ).fetchone()
                if existing is None:
                    raise
                merged = list(dict.fromkeys(json.loads(existing["items_json"]) + ticket["items"]))
                db.execute(
                    "UPDATE tickets SET items_json=?,updated_at=? WHERE ticket_id=?",
                    (json.dumps(merged), _now(), existing["ticket_id"]),
                )
                if handoff_enabled():
                    self._append_case_action_on_connection(
                        db, conv_id, action_type="ticket", description="相关业务诉求已合并到工单草稿",
                        source_ref=str(existing["ticket_id"]),
                        idempotency_key=f"ticket:merge:{existing['ticket_id']}:{main_type}", status="success",
                    )
                return {**dict(existing), "items": merged}
        return ticket

    def create_or_append_ticket(
        self, user_id: str, conv_id: str, item_type: str, label: str, items: list[str]
    ) -> Dict[str, Any]:
        del label
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM tickets WHERE user_id=? AND status NOT IN ('closed','done','rejected') ORDER BY created_at LIMIT 1",
                (user_id,),
            ).fetchone()
            if row is None:
                return self.create_ticket(user_id, conv_id, item_type, item_type, items)
            merged = list(dict.fromkeys(json.loads(row["items_json"]) + list(items) + [item_type]))
            db.execute(
                "UPDATE tickets SET items_json=?,updated_at=? WHERE ticket_id=?",
                (json.dumps(merged), _now(), row["ticket_id"]),
            )
            if handoff_enabled():
                self._append_case_action_on_connection(
                    db, conv_id, action_type="ticket", description="工单草稿已补充业务诉求",
                    source_ref=str(row["ticket_id"]),
                    idempotency_key=f"ticket:append:{row['ticket_id']}:{item_type}", status="success",
                )
            return {**dict(row), "items": merged}
