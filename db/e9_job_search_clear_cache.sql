-- E9 job finder cache reset.
-- Clears every cached role search and its stored openings so each role is searched
-- again with the job relevance check (openings found before it could be for the
-- wrong role, e.g. "Programme Coordinator" stored for Systems Programmer). The next
-- visit to a role's employer matches runs a fresh search. Nothing else references
-- these tables.
--
-- Safe and re-runnable. Run against the target database:
--   psql "<uat connection string>" -f db/e9_job_search_clear_cache.sql
-- or paste into a pgAdmin query window.

BEGIN;

DELETE FROM rerouteher.job_opening;
DELETE FROM rerouteher.job_search;

COMMIT;
