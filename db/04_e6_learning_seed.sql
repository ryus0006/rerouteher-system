-- E6 Personalised Learning - our data fill (interim curated content).
--
-- The base schema ships learning_resource / learning_resource_skill empty; the db
-- team has not populated curated learning content yet. This fills a small set so the
-- E6 learning plan (and its wired e2e) show real curated resources instead of only
-- the YouTube search fallback, until the db team delivers the real catalogue.
--
-- Runs after our schema step (03) and before finalize (05), inside the FK-disabled
-- load window like the other data steps. Idempotent (ON CONFLICT DO NOTHING) with
-- namespaced resource ids (seed-e6-*) so it never collides with db-team rows.
-- References existing rows only: provider OPENLEARN (from the dump) and real
-- skill_taxonomy skill_ids.

-- JavaScript  = 3cd569a2-4f88-4c1e-9995-8dce8c5e51a7
-- SQL         = 598de5b0-5b58-4ea7-8058-a4bc4d18c742
-- (Project Management 7111b95d-... is left unseeded so the plan also exercises the
--  YouTube search fallback.)

INSERT INTO rerouteher.learning_resource
    (resource_id, provider_id, title, url, delivery_mode, duration, language, level, licence_note, last_verified_at)
VALUES
    ('seed-e6-js-basics', 'OPENLEARN', 'Introduction to JavaScript',
     'https://www.open.edu/openlearn/', 'Course', interval '3 hours', 'en', 'Beginner', 'CC-BY', now()),
    ('seed-e6-sql-basics', 'OPENLEARN', 'Learn SQL Fundamentals',
     'https://www.open.edu/openlearn/', 'Course', interval '90 minutes', 'en', 'Beginner', 'CC-BY', now())
ON CONFLICT (resource_id) DO NOTHING;

INSERT INTO rerouteher.learning_resource_skill
    (resource_id, skill_id, relevance, evidence_note)
VALUES
    ('seed-e6-js-basics', '3cd569a2-4f88-4c1e-9995-8dce8c5e51a7', 0.95,
     'Covers core JavaScript syntax and functions from scratch.'),
    ('seed-e6-sql-basics', '598de5b0-5b58-4ea7-8058-a4bc4d18c742', 0.90,
     'Builds the query-writing skills this gap needs.')
ON CONFLICT (resource_id, skill_id) DO NOTHING;
