-- Roll back only the additive handoff schema. Run only after backing up the database.
BEGIN IMMEDIATE;
DROP TABLE IF EXISTS handoff_events;
DROP TABLE IF EXISTS callback_tasks;
DROP TABLE IF EXISTS staff_presence;
DROP TABLE IF EXISTS handoff_blocks;
DROP TABLE IF EXISTS case_action_logs;
DROP TABLE IF EXISTS handoff_records;
DROP TABLE IF EXISTS case_sub_requests;
PRAGMA user_version = 0;
COMMIT;
