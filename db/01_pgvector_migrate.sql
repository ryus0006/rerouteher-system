-- Run AFTER importing ReRouteHer_D1_D8_Full_PostgreSQL_Import.sql.
-- Adapts the No-pgvector compatibility columns (double precision[384]) to pgvector vector(384)
-- and adds cosine indexes so occupation and skill retrieval run inside Postgres.
--
-- Usage:
--   psql -v ON_ERROR_STOP=1 -d rerouteher -f db/01_pgvector_migrate.sql

BEGIN;

SET LOCAL search_path TO rerouteher, public;

-- Install into public so the vector type and its operators (<=>) are visible under any
-- search_path, not just rerouteher.
CREATE EXTENSION IF NOT EXISTS vector SCHEMA public;

-- The import defines inline CHECK (array_length(embedding, 1) = 384) constraints.
-- vector(384) enforces the dimension itself, and array_length has no vector overload,
-- so drop the now-redundant checks before converting the column type.
ALTER TABLE rerouteher.roles DROP CONSTRAINT IF EXISTS roles_role_embedding_check;
ALTER TABLE rerouteher.skill_taxonomy DROP CONSTRAINT IF EXISTS skill_taxonomy_embedding_check;

ALTER TABLE rerouteher.roles
    ALTER COLUMN role_embedding TYPE vector(384) USING role_embedding::vector;

ALTER TABLE rerouteher.skill_taxonomy
    ALTER COLUMN embedding TYPE vector(384) USING embedding::vector;

CREATE INDEX IF NOT EXISTS roles_role_embedding_hnsw
    ON rerouteher.roles USING hnsw (role_embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS skill_taxonomy_embedding_hnsw
    ON rerouteher.skill_taxonomy USING hnsw (embedding vector_cosine_ops);

COMMIT;
