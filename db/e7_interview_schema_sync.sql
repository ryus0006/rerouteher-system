-- E7 interview schema sync.
-- Brings an EXISTING database (UAT, or any instance initialised before the E7
-- interview schema was finalised) up to the schema in 07_e7_interview_coach.sql.
-- Only the interview tables are touched. Seed DATA is not changed.
--
-- Safe and idempotent: re-runnable. Run manually against the target database,
-- it is NOT part of container init:
--   psql "<connection string>" -f db/e7_interview_schema_sync.sql
--
-- Caveats:
--   * The username NOT NULL and the (username, role_id, practice_focus) UNIQUE
--     will fail if UAT already holds rows that violate them. Clean those rows first.
--   * Foreign keys are assumed to exist from the original init and are not re-added.

BEGIN;

-- 1. interview_question : coaching columns.
-- Added NOT NULL with '' default so existing rows stay valid and future inserts
-- that omit them do not fail; the seed loader always supplies real values.
ALTER TABLE rerouteher.interview_question
    ADD COLUMN IF NOT EXISTS interview_method_sources text NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS authoring_method         text NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS answer_framework         text NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS answer_guidance          text NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS strong_evidence_signals  text NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS watch_out_for            text NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS follow_up_question       text NOT NULL DEFAULT '';

-- 2. interview_session : created_at / updated_at, status domain, username NOT NULL.
ALTER TABLE rerouteher.interview_session
    ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now(),
    ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT now();

-- Backfill the new timestamps from the legacy columns when they are still present.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_schema = 'rerouteher' AND table_name = 'interview_session'
                 AND column_name = 'started_at') THEN
        UPDATE rerouteher.interview_session SET created_at = started_at;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns
               WHERE table_schema = 'rerouteher' AND table_name = 'interview_session'
                 AND column_name = 'ended_at') THEN
        UPDATE rerouteher.interview_session SET updated_at = COALESCE(ended_at, updated_at);
    END IF;
END $$;

-- Align the status domain ('in_progress' -> 'active') and default.
UPDATE rerouteher.interview_session SET status = 'active' WHERE status = 'in_progress';
ALTER TABLE rerouteher.interview_session ALTER COLUMN status SET DEFAULT 'active';

-- Enforce NOT NULL on username only when no null rows remain.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM rerouteher.interview_session WHERE username IS NULL) THEN
        ALTER TABLE rerouteher.interview_session ALTER COLUMN username SET NOT NULL;
    END IF;
END $$;

-- Constraints and index (guarded so re-runs do not error).
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'interview_session_status_check') THEN
        ALTER TABLE rerouteher.interview_session
            ADD CONSTRAINT interview_session_status_check CHECK (status IN ('active', 'completed'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'interview_session_practice_focus_check') THEN
        ALTER TABLE rerouteher.interview_session
            ADD CONSTRAINT interview_session_practice_focus_check
            CHECK (practice_focus IN ('general', 'role_specific', 'mixed'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'interview_session_username_role_focus_key') THEN
        ALTER TABLE rerouteher.interview_session
            ADD CONSTRAINT interview_session_username_role_focus_key
            UNIQUE (username, role_id, practice_focus);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS interview_session_username_idx
    ON rerouteher.interview_session (username, created_at DESC);

-- Optional cleanup of the legacy columns (our code no longer uses them).
-- ALTER TABLE rerouteher.interview_session DROP COLUMN IF EXISTS started_at;
-- ALTER TABLE rerouteher.interview_session DROP COLUMN IF EXISTS ended_at;

-- 3. interview_response : processing / retention columns and jsonb feedback.
ALTER TABLE rerouteher.interview_response
    ADD COLUMN IF NOT EXISTS detected_language text,
    ADD COLUMN IF NOT EXISTS audio_duration_s  numeric(6, 2),
    ADD COLUMN IF NOT EXISTS feedback_status   text NOT NULL DEFAULT 'pending',
    ADD COLUMN IF NOT EXISTS error_code        text,
    ADD COLUMN IF NOT EXISTS processed_at      timestamptz,
    ADD COLUMN IF NOT EXISTS updated_at        timestamptz NOT NULL DEFAULT now(),
    ADD COLUMN IF NOT EXISTS content_purged_at timestamptz;

-- worked_well / what_to_improve : text -> jsonb array. NULL or '' becomes '[]'.
DO $$
BEGIN
    IF (SELECT data_type FROM information_schema.columns
        WHERE table_schema = 'rerouteher' AND table_name = 'interview_response'
          AND column_name = 'worked_well') = 'text' THEN
        ALTER TABLE rerouteher.interview_response
            ALTER COLUMN worked_well DROP DEFAULT,
            ALTER COLUMN worked_well TYPE jsonb
                USING CASE WHEN worked_well IS NULL OR worked_well = '' THEN '[]'::jsonb
                           ELSE worked_well::jsonb END,
            ALTER COLUMN worked_well SET DEFAULT '[]'::jsonb,
            ALTER COLUMN worked_well SET NOT NULL;
    END IF;
    IF (SELECT data_type FROM information_schema.columns
        WHERE table_schema = 'rerouteher' AND table_name = 'interview_response'
          AND column_name = 'what_to_improve') = 'text' THEN
        ALTER TABLE rerouteher.interview_response
            ALTER COLUMN what_to_improve DROP DEFAULT,
            ALTER COLUMN what_to_improve TYPE jsonb
                USING CASE WHEN what_to_improve IS NULL OR what_to_improve = '' THEN '[]'::jsonb
                           ELSE what_to_improve::jsonb END,
            ALTER COLUMN what_to_improve SET DEFAULT '[]'::jsonb,
            ALTER COLUMN what_to_improve SET NOT NULL;
    END IF;
END $$;

-- Check constraints (guarded).
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'interview_response_feedback_status_check') THEN
        ALTER TABLE rerouteher.interview_response
            ADD CONSTRAINT interview_response_feedback_status_check
            CHECK (feedback_status IN ('pending', 'processing', 'ready', 'error'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'interview_response_sequence_no_check') THEN
        ALTER TABLE rerouteher.interview_response
            ADD CONSTRAINT interview_response_sequence_no_check CHECK (sequence_no BETWEEN 1 AND 5);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'interview_response_attempt_no_check') THEN
        ALTER TABLE rerouteher.interview_response
            ADD CONSTRAINT interview_response_attempt_no_check CHECK (attempt_no >= 1);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS interview_response_retention_idx
    ON rerouteher.interview_response (created_at)
    WHERE content_purged_at IS NULL;

-- Optional cleanup of the legacy column (our code no longer uses it).
-- ALTER TABLE rerouteher.interview_response DROP COLUMN IF EXISTS answer_text;

COMMIT;
