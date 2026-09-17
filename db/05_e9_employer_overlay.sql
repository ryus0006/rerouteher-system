-- E9 Workplace Needs & Employer Fit - curated employer overlay (our data).
--
-- The base schema's employer_* tables hold the db team's raw report extraction
-- (sparse, and without display fields or a clean 5-priority disclosure). This
-- overlay curates the employers the UI shows: display fields + the five workplace
-- priorities each employer's report discloses. The real report link is taken from
-- employer_report_evidence at query time (repository LEFT JOIN); the report_* here
-- is the fallback used when the DB has no row (e.g. Maybank) or no url.
--
-- Runs after our schema/data steps and before finalize. Standalone table (no FK),
-- idempotent. employer_id matches a real employer_report_evidence.employer_id where
-- one exists, so those rows surface the real DB report link.

CREATE TABLE IF NOT EXISTS rerouteher.employer_overlay
(
    employer_id  text PRIMARY KEY,
    name         text NOT NULL,
    industry     text,
    location     text,
    website      text,
    summary      text,
    logo_text    text,
    logo_bg      text,
    logo_fg      text,
    discloses    text[] NOT NULL DEFAULT '{}',
    report_label text,
    report_url   text,
    report_year  integer
);

INSERT INTO rerouteher.employer_overlay
    (employer_id, name, industry, location, website, summary,
     logo_text, logo_bg, logo_fg, discloses, report_label, report_url, report_year)
VALUES
    ('maybank', 'Maybank', 'Financial Services', 'Kuala Lumpur', 'https://www.maybank.com/',
     'Maybank reports flexible working arrangements, parental support and initiatives to promote workplace inclusion in its sustainability reporting.',
     'Maybank', '#ffcc00', '#1f2a44',
     ARRAY['flexible_work','childcare_support','inclusive_workplace'],
     'Sustainability Report 2024', 'https://www.maybank.com/en/sustainability.page', 2024),

    ('5183', 'PETRONAS', 'Energy', 'Kuala Lumpur', 'https://www.petronas.com/',
     'PETRONAS discloses parental leave provisions, a returner programme and workplace flexibility initiatives in its sustainability reporting.',
     'PETRONAS', '#00a19c', '#ffffff',
     ARRAY['flexible_work','parental_support','returning_to_work','inclusive_workplace'],
     'Sustainability Report 2023', 'https://www.petronas.com/sustainability/reports', 2023),

    ('1023', 'CIMB', 'Financial Services', 'Kuala Lumpur', 'https://www.cimb.com/',
     'CIMB reports flexible work arrangements and diversity and inclusion programmes in its sustainability reporting.',
     'CIMB', '#c4161c', '#ffffff',
     ARRAY['flexible_work','inclusive_workplace'],
     'Sustainability Report 2024', 'https://www.cimb.com/en/sustainability.html', 2024),

    ('4863', 'Telekom Malaysia', 'Telecommunications', 'Kuala Lumpur', 'https://www.tm.com.my/',
     'Telekom Malaysia reports childcare facilities, parental leave provisions and flexible working in its sustainability reporting.',
     'TM', '#ff6600', '#ffffff',
     ARRAY['flexible_work','childcare_support','parental_support'],
     'Sustainability Report 2024', 'https://www.tm.com.my/sustainability', 2024),

    ('5225', 'IHH Healthcare', 'Healthcare', 'Kuala Lumpur', 'https://www.ihhhealthcare.com/',
     'IHH Healthcare reports childcare provision, parental support and reintegration for returning staff in its sustainability reporting.',
     'IHH', '#6b2c91', '#ffffff',
     ARRAY['childcare_support','parental_support','returning_to_work'],
     'Sustainability Report 2024', 'https://www.ihhhealthcare.com/sustainability', 2024),

    ('4707', 'Nestle Malaysia', 'Consumer Goods', 'Petaling Jaya', 'https://www.nestle.com.my/',
     'Nestle Malaysia reports parental leave, a return-to-work policy and gender balance targets in its sustainability reporting.',
     'Nestle', '#00539f', '#ffffff',
     ARRAY['flexible_work','parental_support','returning_to_work','inclusive_workplace'],
     'Sustainability Report 2024', 'https://www.nestle.com.my/csv', 2024),

    ('6947', 'CelcomDigi', 'Telecommunications', 'Kuala Lumpur', 'https://www.celcomdigi.com/',
     'CelcomDigi reports hybrid working, a career reboot programme and diversity commitments in its sustainability reporting.',
     'CelcomDigi', '#e4007c', '#ffffff',
     ARRAY['flexible_work','returning_to_work','inclusive_workplace'],
     'Sustainability Report 2024', 'https://www.celcomdigi.com/sustainability', 2024),

    ('4197', 'Sime Darby', 'Industrial', 'Petaling Jaya', 'https://www.simedarby.com/',
     'Sime Darby reports parental leave provisions and women in leadership targets in its sustainability reporting.',
     'Sime Darby', '#003da5', '#ffffff',
     ARRAY['parental_support','inclusive_workplace'],
     'Sustainability Report 2024', 'https://www.simedarby.com/sustainability', 2024)
ON CONFLICT (employer_id) DO NOTHING;
