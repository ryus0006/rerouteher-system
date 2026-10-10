-- E9 role-level job-search cache and tracked-employer openings.

ALTER TABLE IF EXISTS rerouteher.roles
    ADD COLUMN IF NOT EXISTS search_title text;

UPDATE rerouteher.roles
SET search_title = COALESCE(
    NULLIF(
        btrim(
            regexp_replace(
                regexp_replace(role_title, '\s*\([^)]*\)', '', 'g'),
                '\s+(Grade\s+)?[A-Z]{1,3}[0-9]{1,3}\s*$',
                '',
                'i'
            )
        ),
        ''
    ),
    role_title
)
WHERE search_title IS NULL OR btrim(search_title) = '';

ALTER TABLE IF EXISTS rerouteher.roles
    ALTER COLUMN search_title SET NOT NULL;

ALTER TABLE IF EXISTS rerouteher.employers
    ADD COLUMN IF NOT EXISTS location text,
    ADD COLUMN IF NOT EXISTS website text,
    ADD COLUMN IF NOT EXISTS logo_url text;

-- Employer card logos, by image URL. Populate one row per employer; the card falls
-- back to an initials badge when logo_url is empty. Only sets rows not already set,
-- so manual overrides survive a re-run.
UPDATE rerouteher.employers AS e
SET logo_url = v.logo_url
FROM (VALUES
    ('1066', 'https://logo.clearbit.com/rhbgroup.com'),
    ('4677', 'https://logo.clearbit.com/ytl.com'),
    ('6012', 'https://logo.clearbit.com/maxis.com.my'),
    ('1155', 'https://logo.clearbit.com/maybank.com'),
    ('5183', 'https://logo.clearbit.com/petronas.com')
) AS v(employer_id, logo_url)
WHERE e.employer_id = v.employer_id
  AND COALESCE(btrim(e.logo_url), '') = '';

CREATE TABLE IF NOT EXISTS rerouteher.job_search
(
    role_id text NOT NULL,
    search_title text NOT NULL,
    status text NOT NULL,
    searched_at timestamp with time zone NOT NULL,
    providers_attempted text[] NOT NULL DEFAULT '{}',
    CONSTRAINT job_search_pkey PRIMARY KEY (role_id),
    CONSTRAINT job_search_role_id_fkey FOREIGN KEY (role_id)
        REFERENCES rerouteher.roles (role_id),
    CONSTRAINT job_search_status_check CHECK (
        status IN ('ready', 'empty', 'temporarily_unavailable')
    )
);

CREATE TABLE IF NOT EXISTS rerouteher.job_opening
(
    job_opening_id bigserial NOT NULL,
    role_id text NOT NULL,
    employer_id text NOT NULL,
    canonical_url text NOT NULL,
    title text NOT NULL,
    location text,
    provider text NOT NULL,
    query_text text,
    raw_payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    retrieved_at timestamp with time zone NOT NULL DEFAULT now(),
    first_seen_at timestamp with time zone NOT NULL DEFAULT now(),
    last_seen_at timestamp with time zone NOT NULL DEFAULT now(),
    status text NOT NULL DEFAULT 'active',
    CONSTRAINT job_opening_pkey PRIMARY KEY (job_opening_id),
    CONSTRAINT job_opening_role_id_employer_id_canonical_url_key
        UNIQUE (role_id, employer_id, canonical_url),
    CONSTRAINT job_opening_role_id_fkey FOREIGN KEY (role_id)
        REFERENCES rerouteher.roles (role_id),
    CONSTRAINT job_opening_employer_id_fkey FOREIGN KEY (employer_id)
        REFERENCES rerouteher.employers (employer_id),
    CONSTRAINT job_opening_status_check CHECK (status IN ('active', 'inactive'))
);

CREATE INDEX IF NOT EXISTS job_search_status_idx
    ON rerouteher.job_search(status);

CREATE INDEX IF NOT EXISTS job_opening_role_status_seen_idx
    ON rerouteher.job_opening(role_id, status, last_seen_at DESC);

CREATE INDEX IF NOT EXISTS job_opening_employer_idx
    ON rerouteher.job_opening(employer_id);
