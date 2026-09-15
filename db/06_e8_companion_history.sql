-- E8 AI Companion - conversation history (our data).
-- Guest-usable: keyed by a frontend-generated session_id; username is recorded
-- when signed in (nullable, no FK so guests work). Runs before finalize.
CREATE TABLE IF NOT EXISTS rerouteher.companion_message
(
    id         bigserial PRIMARY KEY,
    session_id text NOT NULL,
    username   text,
    role       text NOT NULL,
    content    text NOT NULL,
    tokens_in  integer NOT NULL DEFAULT 0,
    tokens_out integer NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS companion_message_session_idx
    ON rerouteher.companion_message (session_id, created_at);
