-- E9 employer card logos. Idempotent sync for an existing DB (local or UAT).
-- Adds employers.logo_url and seeds a starter set; override per employer as needed.
-- Mirrors the column added in 08_e9_job_search.sql for fresh builds.

ALTER TABLE IF EXISTS rerouteher.employers
    ADD COLUMN IF NOT EXISTS logo_url text;

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
