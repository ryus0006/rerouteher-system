-- E5 User Accounts & Saved Journey - account model change (FOR DB-TEAM REVIEW).
--
-- The base schema (schema (2).sql) defines rerouteher.app_user on an external-identity
-- model (auth_subject / identity_provider). E5 (PM-agreed) uses accounts we store
-- ourselves: a username + a bcrypt password_hash, no email, no external identity
-- provider. saved_journey holds the whole guest plan as JSON, keyed by username.
--
-- This replaces the provided account tables with the E5 shape. The provided tables carry
-- NO data in the dump, so drop + create is safe here. REVIEW before running on any
-- environment that already holds account rows.
--
-- Apply AFTER the base schema + data.

BEGIN;

-- E5 recomputes employer matches and keeps the shortlist in plan_json, so the
-- separate shortlist_item table is not used.
DROP TABLE IF EXISTS rerouteher.shortlist_item CASCADE;
DROP TABLE IF EXISTS rerouteher.saved_journey CASCADE;
DROP TABLE IF EXISTS rerouteher.app_user CASCADE;

CREATE TABLE rerouteher.app_user (
    username      text PRIMARY KEY,        -- normalised to lower-case; the login key
    password_hash text NOT NULL,           -- bcrypt; never plain text
    display_name  text NOT NULL,           -- shown on screen; defaults to username
    created_at    timestamptz NOT NULL DEFAULT now(),
    last_seen_at  timestamptz
);

CREATE TABLE rerouteher.saved_journey (
    username   text PRIMARY KEY
               REFERENCES rerouteher.app_user(username) ON UPDATE CASCADE ON DELETE CASCADE,
    plan_json  jsonb NOT NULL DEFAULT '{}'::jsonb,   -- the whole exportPlan()
    updated_at timestamptz NOT NULL DEFAULT now()
);

COMMIT;
