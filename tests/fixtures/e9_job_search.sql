-- Deterministic E9 full-stack fixture. Safe to run repeatedly.
INSERT INTO rerouteher.job_search
    (role_id, search_title, status, searched_at, providers_attempted)
VALUES
    ('R03', 'Human Resources Officer', 'ready',
     TIMESTAMPTZ '2026-10-04 00:00:00+00', ARRAY['fixture']::text[])
ON CONFLICT (role_id) DO UPDATE SET
    search_title = EXCLUDED.search_title,
    status = EXCLUDED.status,
    searched_at = EXCLUDED.searched_at,
    providers_attempted = EXCLUDED.providers_attempted;

INSERT INTO rerouteher.job_opening
    (role_id, employer_id, canonical_url, title, location, provider,
     query_text, raw_payload, retrieved_at, first_seen_at, last_seen_at, status)
VALUES
    ('R03', '1023', 'https://example.test/jobs/hr-officer-r03',
     'Human Resources Officer', 'Kuala Lumpur', 'fixture',
     'Human Resources Officer Malaysia', '{}'::jsonb,
     TIMESTAMPTZ '2026-10-04 00:00:00+00',
     TIMESTAMPTZ '2026-10-04 00:00:00+00',
     TIMESTAMPTZ '2026-10-04 00:00:00+00', 'active')
ON CONFLICT (role_id, employer_id, canonical_url) DO UPDATE SET
    title = EXCLUDED.title,
    location = EXCLUDED.location,
    provider = EXCLUDED.provider,
    query_text = EXCLUDED.query_text,
    raw_payload = EXCLUDED.raw_payload,
    retrieved_at = EXCLUDED.retrieved_at,
    last_seen_at = EXCLUDED.last_seen_at,
    status = EXCLUDED.status;
