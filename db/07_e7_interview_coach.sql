-- E7 AI Interview Coach - interview schema and authoritative seed datasets (our data).
--
-- Defines only the five interview-related tables. These are a focused re-draft of the
-- same five definitions sketched in it3_schema.sql (db-team ERD export, reference only,
-- not run): ai_evaluation_rubric, interview_question, interview_question_rubric,
-- interview_session, interview_response. it3_schema.sql itself is never executed; this
-- file is the one that actually runs, with the ownership, status, and retention rules
-- E7 needs (strict CHECK constraints, username cascade, JSON-array feedback fields,
-- transcription/processing metadata) that the ERD sketch did not carry.
--
-- Question and rubric content loads from the three authoritative CSVs via all-text
-- staging tables, so this script can be re-run as the CSVs are corrected. Staging
-- row counts are asserted before anything commits, so a bad CSV aborts the whole
-- container init rather than loading partial data silently.
--
-- Runs after 06_e8_companion_history.sql and before 08_finalize.sql, inside the
-- FK-disabled load window set up by 00_init_extensions_schemas.sql.

BEGIN;

CREATE TABLE IF NOT EXISTS rerouteher.ai_evaluation_rubric
(
    criterion_id             text NOT NULL,
    criterion                text NOT NULL,
    what_ai_checks           text NOT NULL,
    positive_feedback_tag    text NOT NULL,
    improvement_feedback_tag text NOT NULL,
    applies_to               text[] NOT NULL,
    prohibited_inference     text NOT NULL,
    CONSTRAINT ai_evaluation_rubric_pkey PRIMARY KEY (criterion_id),
    CONSTRAINT ai_evaluation_rubric_criterion_key UNIQUE (criterion)
);

CREATE TABLE IF NOT EXISTS rerouteher.interview_question
(
    question_id             text NOT NULL,
    role_id                 text,
    category                text NOT NULL,
    difficulty              text NOT NULL,
    question_text           text NOT NULL,
    anchor_json             jsonb NOT NULL,
    rubric_json             jsonb NOT NULL,
    review_status           text NOT NULL,
    interview_method_sources text NOT NULL,
    authoring_method          text NOT NULL,
    answer_framework          text NOT NULL,
    answer_guidance           text NOT NULL,
    strong_evidence_signals   text NOT NULL,
    watch_out_for             text NOT NULL,
    follow_up_question        text NOT NULL,
    CONSTRAINT interview_question_pkey PRIMARY KEY (question_id),
    CONSTRAINT interview_question_difficulty_check
        CHECK (difficulty IN ('foundation', 'intermediate', 'advanced')),
    CONSTRAINT interview_question_role_id_fkey FOREIGN KEY (role_id)
        REFERENCES rerouteher.roles (role_id) MATCH SIMPLE
        ON UPDATE NO ACTION ON DELETE NO ACTION
);

CREATE INDEX IF NOT EXISTS interview_question_role_difficulty_idx
    ON rerouteher.interview_question (role_id, difficulty);

CREATE TABLE IF NOT EXISTS rerouteher.interview_question_rubric
(
    question_id  text NOT NULL,
    criterion_id text NOT NULL,
    CONSTRAINT interview_question_rubric_pkey PRIMARY KEY (question_id, criterion_id),
    CONSTRAINT interview_question_rubric_question_id_fkey FOREIGN KEY (question_id)
        REFERENCES rerouteher.interview_question (question_id) MATCH SIMPLE
        ON UPDATE NO ACTION ON DELETE CASCADE,
    CONSTRAINT interview_question_rubric_criterion_id_fkey FOREIGN KEY (criterion_id)
        REFERENCES rerouteher.ai_evaluation_rubric (criterion_id) MATCH SIMPLE
        ON UPDATE NO ACTION ON DELETE NO ACTION
);

CREATE INDEX IF NOT EXISTS interview_question_rubric_criterion_idx
    ON rerouteher.interview_question_rubric (criterion_id);

CREATE TABLE IF NOT EXISTS rerouteher.interview_session
(
    session_id     text NOT NULL DEFAULT (gen_random_uuid())::text,
    username       text NOT NULL,
    role_id        text NOT NULL,
    practice_focus text NOT NULL,
    status         text NOT NULL DEFAULT 'active',
    created_at     timestamptz NOT NULL DEFAULT now(),
    updated_at     timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT interview_session_pkey PRIMARY KEY (session_id),
    CONSTRAINT interview_session_username_role_focus_key
        UNIQUE (username, role_id, practice_focus),
    CONSTRAINT interview_session_practice_focus_check
        CHECK (practice_focus IN ('general', 'role_specific', 'mixed')),
    CONSTRAINT interview_session_status_check
        CHECK (status IN ('active', 'completed')),
    CONSTRAINT interview_session_username_fkey FOREIGN KEY (username)
        REFERENCES rerouteher.app_user (username) MATCH SIMPLE
        ON UPDATE CASCADE ON DELETE CASCADE,
    CONSTRAINT interview_session_role_id_fkey FOREIGN KEY (role_id)
        REFERENCES rerouteher.roles (role_id) MATCH SIMPLE
        ON UPDATE NO ACTION ON DELETE NO ACTION
);

CREATE INDEX IF NOT EXISTS interview_session_username_idx
    ON rerouteher.interview_session (username, created_at DESC);

CREATE TABLE IF NOT EXISTS rerouteher.interview_response
(
    response_id          bigserial NOT NULL,
    session_id           text NOT NULL,
    question_id          text NOT NULL,
    sequence_no          integer NOT NULL,
    attempt_no           integer NOT NULL DEFAULT 1,
    transcript           text,
    detected_language    text,
    audio_duration_s     numeric(6, 2),
    feedback_status      text NOT NULL DEFAULT 'pending',
    feedback_summary     text,
    worked_well          jsonb NOT NULL DEFAULT '[]'::jsonb,
    what_to_improve      jsonb NOT NULL DEFAULT '[]'::jsonb,
    strength_tags        text[] NOT NULL DEFAULT '{}'::text[],
    improvement_tags     text[] NOT NULL DEFAULT '{}'::text[],
    model_metadata       jsonb NOT NULL DEFAULT '{}'::jsonb,
    error_code           text,
    processed_at         timestamptz,
    created_at           timestamptz NOT NULL DEFAULT now(),
    updated_at           timestamptz NOT NULL DEFAULT now(),
    content_purged_at    timestamptz,
    CONSTRAINT interview_response_pkey PRIMARY KEY (response_id),
    CONSTRAINT interview_response_session_sequence_attempt_key
        UNIQUE (session_id, sequence_no, attempt_no),
    CONSTRAINT interview_response_sequence_no_check
        CHECK (sequence_no BETWEEN 1 AND 5),
    CONSTRAINT interview_response_attempt_no_check
        CHECK (attempt_no >= 1),
    CONSTRAINT interview_response_feedback_status_check
        CHECK (feedback_status IN ('pending', 'processing', 'ready', 'error')),
    CONSTRAINT interview_response_session_id_fkey FOREIGN KEY (session_id)
        REFERENCES rerouteher.interview_session (session_id) MATCH SIMPLE
        ON UPDATE NO ACTION ON DELETE CASCADE,
    CONSTRAINT interview_response_question_id_fkey FOREIGN KEY (question_id)
        REFERENCES rerouteher.interview_question (question_id) MATCH SIMPLE
        ON UPDATE NO ACTION ON DELETE NO ACTION
);

CREATE INDEX IF NOT EXISTS interview_response_retention_idx
    ON rerouteher.interview_response (created_at)
    WHERE content_purged_at IS NULL;

-- Seed load: all-text staging tables COPYed from the mounted CSVs, converted on
-- insert, counted, then dropped. ON CONFLICT upserts the lookup rows and ignores
-- duplicate mapping pairs, so this script is safe to re-run.

CREATE TEMP TABLE staging_evaluation_rubric
(
    criterion_id             text,
    criterion                text,
    what_ai_checks           text,
    positive_feedback_tag    text,
    improvement_feedback_tag text,
    applies_to               text,
    prohibited_inference     text
) ON COMMIT DROP;

COPY staging_evaluation_rubric
    FROM '/docker-entrypoint-initdb.d/e7_interview_evaluation_rubric.csv'
    WITH (FORMAT csv, HEADER true);

CREATE TEMP TABLE staging_interview_question
(
    question_id              text,
    role_id                  text,
    category                 text,
    difficulty               text,
    question_text            text,
    anchor_json              text,
    rubric_json              text,
    review_status            text,
    interview_method_sources text,
    authoring_method         text,
    answer_framework         text,
    answer_guidance          text,
    strong_evidence_signals  text,
    watch_out_for            text,
    follow_up_question       text
) ON COMMIT DROP;

COPY staging_interview_question
    FROM '/docker-entrypoint-initdb.d/interview_question.csv'
    WITH (FORMAT csv, HEADER true);

CREATE TEMP TABLE staging_interview_question_rubric
(
    question_id  text,
    criterion_id text
) ON COMMIT DROP;

COPY staging_interview_question_rubric
    FROM '/docker-entrypoint-initdb.d/e7_interview_question_rubric.csv'
    WITH (FORMAT csv, HEADER true);

DO $$
DECLARE
    question_count int;
    criterion_count int;
    mapping_count int;
BEGIN
    SELECT count(*) INTO question_count FROM staging_interview_question;
    SELECT count(*) INTO criterion_count FROM staging_evaluation_rubric;
    SELECT count(*) INTO mapping_count FROM staging_interview_question_rubric;

    IF question_count != 7592 THEN
        RAISE EXCEPTION 'expected 7592 interview questions, found %', question_count;
    END IF;
    IF criterion_count != 10 THEN
        RAISE EXCEPTION 'expected 10 evaluation criteria, found %', criterion_count;
    END IF;
    IF mapping_count != 44902 THEN
        RAISE EXCEPTION 'expected 44902 question-criterion mappings, found %', mapping_count;
    END IF;
END $$;

INSERT INTO rerouteher.ai_evaluation_rubric
    (criterion_id, criterion, what_ai_checks, positive_feedback_tag,
     improvement_feedback_tag, applies_to, prohibited_inference)
SELECT
    criterion_id, criterion, what_ai_checks, positive_feedback_tag,
    improvement_feedback_tag, string_to_array(applies_to, '; '), prohibited_inference
FROM staging_evaluation_rubric
ON CONFLICT (criterion_id) DO UPDATE SET
    criterion = excluded.criterion,
    what_ai_checks = excluded.what_ai_checks,
    positive_feedback_tag = excluded.positive_feedback_tag,
    improvement_feedback_tag = excluded.improvement_feedback_tag,
    applies_to = excluded.applies_to,
    prohibited_inference = excluded.prohibited_inference;

INSERT INTO rerouteher.interview_question
    (question_id, role_id, category, difficulty, question_text,
     anchor_json, rubric_json, review_status,
     interview_method_sources, authoring_method, answer_framework, answer_guidance,
     strong_evidence_signals, watch_out_for, follow_up_question)
SELECT
    question_id, NULLIF(role_id, 'NULL'), category, difficulty, question_text,
    anchor_json::jsonb, rubric_json::jsonb, review_status,
    interview_method_sources, authoring_method, answer_framework, answer_guidance,
    strong_evidence_signals, watch_out_for, follow_up_question
FROM staging_interview_question
ON CONFLICT (question_id) DO UPDATE SET
    role_id = excluded.role_id,
    category = excluded.category,
    difficulty = excluded.difficulty,
    question_text = excluded.question_text,
    anchor_json = excluded.anchor_json,
    rubric_json = excluded.rubric_json,
    review_status = excluded.review_status,
    interview_method_sources = excluded.interview_method_sources,
    authoring_method = excluded.authoring_method,
    answer_framework = excluded.answer_framework,
    answer_guidance = excluded.answer_guidance,
    strong_evidence_signals = excluded.strong_evidence_signals,
    watch_out_for = excluded.watch_out_for,
    follow_up_question = excluded.follow_up_question;

INSERT INTO rerouteher.interview_question_rubric (question_id, criterion_id)
SELECT question_id, criterion_id
FROM staging_interview_question_rubric
ON CONFLICT (question_id, criterion_id) DO NOTHING;

COMMIT;
