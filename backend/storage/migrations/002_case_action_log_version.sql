-- Monotonic action-ledger version used by handoff state hashes.
BEGIN IMMEDIATE;
CREATE TABLE IF NOT EXISTS case_action_log_versions (
    conv_id TEXT PRIMARY KEY REFERENCES sessions(conv_id),
    version INTEGER NOT NULL DEFAULT 0
);
PRAGMA user_version = 2;
COMMIT;
