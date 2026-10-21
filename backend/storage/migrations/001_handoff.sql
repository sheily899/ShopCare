-- Additive schema for layered human handoff. Existing tables and rows are untouched.
BEGIN IMMEDIATE;

CREATE TABLE IF NOT EXISTS case_sub_requests (
    sub_request_id TEXT PRIMARY KEY,
    conv_id TEXT NOT NULL REFERENCES sessions(conv_id),
    sub_request_key TEXT NOT NULL,
    intent_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('open','in_progress','resolved','blocked')),
    source_ref TEXT NOT NULL DEFAULT '',
    source_message_seq INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(conv_id, sub_request_key)
);
CREATE INDEX IF NOT EXISTS idx_case_sub_requests_conv_status
    ON case_sub_requests(conv_id, status, updated_at);

CREATE TABLE IF NOT EXISTS handoff_records (
    handoff_id TEXT PRIMARY KEY,
    conv_id TEXT NOT NULL REFERENCES sessions(conv_id),
    idempotency_key TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open','in_progress','resolved','closed')),
    reason TEXT NOT NULL DEFAULT '',
    source_message_seq INTEGER,
    temporary_summary_json TEXT NOT NULL DEFAULT '{}',
    temporary_priority TEXT NOT NULL DEFAULT 'P3' CHECK(temporary_priority IN ('P0','P1','P2','P3')),
    temporary_priority_score INTEGER NOT NULL DEFAULT 0,
    temporary_priority_reasons_json TEXT NOT NULL DEFAULT '[]',
    final_priority TEXT CHECK(final_priority IS NULL OR final_priority IN ('P0','P1','P2','P3')),
    final_priority_score INTEGER,
    final_priority_reasons_json TEXT NOT NULL DEFAULT '[]',
    state_hash TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    closed_at TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_handoff_one_active_per_conversation
    ON handoff_records(conv_id) WHERE status IN ('open','in_progress');
CREATE INDEX IF NOT EXISTS idx_handoff_records_status_updated
    ON handoff_records(status, updated_at);

CREATE TABLE IF NOT EXISTS case_action_logs (
    action_id TEXT PRIMARY KEY,
    conv_id TEXT NOT NULL REFERENCES sessions(conv_id),
    handoff_id TEXT REFERENCES handoff_records(handoff_id),
    sub_request_id TEXT REFERENCES case_sub_requests(sub_request_id),
    action_type TEXT NOT NULL CHECK(action_type IN ('tool','rule','followup','ticket','reply','system')),
    description TEXT NOT NULL,
    source_ref TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL CHECK(status IN ('pending','success','failed')),
    idempotency_key TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_case_action_logs_conv_created
    ON case_action_logs(conv_id, created_at, action_id);

CREATE TABLE IF NOT EXISTS handoff_blocks (
    handoff_id TEXT NOT NULL REFERENCES handoff_records(handoff_id),
    block_type TEXT NOT NULL CHECK(block_type IN ('summary','emotion_urgency','next_steps')),
    status TEXT NOT NULL CHECK(status IN ('pending','ready','degraded','failed')),
    state_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    error_code TEXT NOT NULL DEFAULT '',
    latency_ms REAL,
    source_message_seq INTEGER,
    generated_at TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(handoff_id, block_type)
);

CREATE TABLE IF NOT EXISTS staff_presence (
    agent_id TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK(status IN ('online','offline','unknown')),
    observed_at TEXT NOT NULL,
    expires_at TEXT,
    source TEXT NOT NULL DEFAULT 'manual'
);

CREATE TABLE IF NOT EXISTS callback_tasks (
    callback_task_id TEXT PRIMARY KEY,
    handoff_id TEXT NOT NULL REFERENCES handoff_records(handoff_id),
    conv_id TEXT NOT NULL REFERENCES sessions(conv_id),
    status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','contacted','cancelled')),
    due_at TEXT,
    reason TEXT NOT NULL DEFAULT '',
    idempotency_key TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_callback_tasks_pending_due
    ON callback_tasks(status, due_at);

CREATE TABLE IF NOT EXISTS handoff_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    handoff_id TEXT NOT NULL REFERENCES handoff_records(handoff_id),
    conv_id TEXT NOT NULL REFERENCES sessions(conv_id),
    event_type TEXT NOT NULL,
    data_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_handoff_events_conv_event
    ON handoff_events(conv_id, event_id);

PRAGMA user_version = 1;
COMMIT;
