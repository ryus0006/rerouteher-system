-- E5 User Accounts & Saved Journey. First write path in the backend.
-- account_store is separate from the read-only rerouteher reference schema.
CREATE SCHEMA IF NOT EXISTS account_store;

CREATE TABLE IF NOT EXISTS account_store.app_user (
    username      text PRIMARY KEY,        -- normalised to lower-case; the login key
    password_hash text NOT NULL,           -- bcrypt; never plain text
    display_name  text NOT NULL,           -- shown on screen; defaults to username
    created_at    timestamptz NOT NULL DEFAULT now(),
    last_seen_at  timestamptz
);

CREATE TABLE IF NOT EXISTS account_store.saved_journey (
    username   text PRIMARY KEY
               REFERENCES account_store.app_user(username)
               ON UPDATE CASCADE ON DELETE CASCADE,
    plan_json  jsonb NOT NULL DEFAULT '{}'::jsonb,  -- the whole exportPlan()
    updated_at timestamptz NOT NULL DEFAULT now()
);
