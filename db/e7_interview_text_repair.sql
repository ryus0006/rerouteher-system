-- E7 interview text repair (live UAT).
-- Reverses UTF-8-decoded-as-Latin1 mojibake in the CSV-loaded interview text, e.g.
-- the smart quotes in question_text that render as:  â€œadminister ICT systemâ€
-- The fix re-encodes each affected value to its original bytes (Latin1) and reads
-- them back as UTF-8, restoring the real characters (" " ' - accents, etc.).
--
-- Idempotent and safe to re-run: the markers it keys on (A-circumflex, A-tilde,
-- a-circumflex) only exist in mojibake and are gone once a value is corrected, so a
-- second run matches nothing. Only rows that actually contain the markers are touched.
--
-- Run against the target database (e.g. UAT), inside one transaction:
--   psql "<uat connection string>" -f db/e7_interview_text_repair.sql
-- or paste into a pgAdmin query window and execute.

BEGIN;

-- interview_question : all CSV-loaded prose columns.
DO $$
DECLARE
    col   text;
    cols  text[] := ARRAY[
        'question_text', 'answer_framework', 'answer_guidance',
        'strong_evidence_signals', 'watch_out_for', 'follow_up_question',
        'interview_method_sources', 'authoring_method'
    ];
    fixed int;
BEGIN
    FOREACH col IN ARRAY cols LOOP
        EXECUTE format(
            'UPDATE rerouteher.interview_question
                SET %1$I = convert_from(convert_to(%1$I, ''LATIN1''), ''UTF8'')
              WHERE %1$I ~ ''[ÂÃâ]''',
            col
        );
        GET DIAGNOSTICS fixed = ROW_COUNT;
        RAISE NOTICE 'interview_question.% repaired % row(s)', col, fixed;
    END LOOP;
END $$;

-- ai_evaluation_rubric : CSV-loaded prose columns (10 rows).
DO $$
DECLARE
    col   text;
    cols  text[] := ARRAY[
        'criterion', 'what_ai_checks', 'positive_feedback_tag',
        'improvement_feedback_tag', 'prohibited_inference'
    ];
    fixed int;
BEGIN
    FOREACH col IN ARRAY cols LOOP
        EXECUTE format(
            'UPDATE rerouteher.ai_evaluation_rubric
                SET %1$I = convert_from(convert_to(%1$I, ''LATIN1''), ''UTF8'')
              WHERE %1$I ~ ''[ÂÃâ]''',
            col
        );
        GET DIAGNOSTICS fixed = ROW_COUNT;
        RAISE NOTICE 'ai_evaluation_rubric.% repaired % row(s)', col, fixed;
    END LOOP;
END $$;

COMMIT;
