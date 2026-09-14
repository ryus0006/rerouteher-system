-- Local-load finalize: turn FK/trigger enforcement back on for the app's sessions,
-- after the data has loaded (00_init_extensions_schemas.sql disabled it to allow the
-- db team's out-of-order data-only dump to load). Local-init concern only.
ALTER DATABASE rerouteher RESET session_replication_role;
