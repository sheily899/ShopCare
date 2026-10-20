-- Roll back only the additive action-log version table after backing up the database.
BEGIN IMMEDIATE;
DROP TABLE IF EXISTS case_action_log_versions;
PRAGMA user_version = 1;
COMMIT;
