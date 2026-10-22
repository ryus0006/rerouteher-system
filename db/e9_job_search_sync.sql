-- E9 job finder schema sync.
-- Brings an existing database (UAT, or a local instance initialised before E9) up
-- to the job-finder schema: roles.search_title, employers.location/website, and the
-- job_search and job_opening tables. Mirrors 08_e9_job_search.sql, which is the
-- fresh-init source; this file is the one to run manually against an existing DB.
--
-- Safe and idempotent: re-runnable. Run against the target database:
--   psql "<uat connection string>" -f db/e9_job_search_sync.sql
-- or paste into a pgAdmin query window.

BEGIN;

-- roles.search_title: strip civil-service grade codes and parentheticals so the
-- job search uses a clean title (e.g. "Radiographer Grade U29" -> "Radiographer").
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

-- employers display fields read by the matcher/UI (populated by the data team).
ALTER TABLE IF EXISTS rerouteher.employers
    ADD COLUMN IF NOT EXISTS location text,
    ADD COLUMN IF NOT EXISTS website text;

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

COMMIT;
