-- Local bootstrap that runs before the db-team schema import. The provided
-- schema (schema (2).sql) assumes the rerouteher schema and the pgvector extension
-- already exist (roles and skill_taxonomy use vector(384)), so create them here.
CREATE EXTENSION IF NOT EXISTS vector SCHEMA public;

CREATE SCHEMA IF NOT EXISTS rerouteher;

-- The db team's dump is data-only and its INSERT order does not follow FK
-- dependencies (e.g. caregiving_map before skill_taxonomy), while our split applies
-- all FK constraints up front. Load with FK/triggers disabled for the init sessions
-- that follow, then 04_finalize.sql turns enforcement back on for the app. This is a
-- local-load concern only; the db team restores data their own way.
ALTER DATABASE rerouteher SET session_replication_role = 'replica';
