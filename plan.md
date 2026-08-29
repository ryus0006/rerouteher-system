# ReRouteHer - FastAPI Backend Plan (Iteration 1)

Backend implementation plan for the ReRouteHer It1 guest journey, derived from the master backlog (`03 Design and Analysis Artefacts/User Story Mapping & Use Cases/epics-and-user-stories.md`) and the data-governance deliverables (`10 Data Governance/`).

This is an academic FIT5120 project, not a DOKU project: plain FastAPI, no DOKU AI Standard conventions, CORS is allowed (the Vite client needs it).

## 1. Scope

It1 backend is 3 stateless endpoints. No database writes (guests persist nothing; all compute is request-time). The service reads static reference tables the data team prepared.

| Endpoint | Story | Job |
|---|---|---|
| `POST /api/cv/parse` | E2 / US2.1 | PDF to structured CV (NLP extractor) |
| `POST /api/snapshot/generate` | E3 / US3.1-3.4 | skills + previous occupation + recommended roles |
| `POST /api/gap/compute` | E4 / US4.1-4.3 | readiness % + merged top-3 gap |

Out of It1 scope (loaded but not queried): Employer Fit (E4 / US4.4-4.5) is deferred to Iteration 2; the `employers` and `role_sector_map` tables are imported but not used yet.

## 2. Data layer (PostgreSQL + pgvector)

Source of truth for reference data is `10 Data Governance/ImprovedVersion/05_DATABASE_FILES/ReRouteHer_D1_D8_Full_PostgreSQL_Import.sql` (schema `rerouteher`).

Tables the It1 backend reads:

- `roles` - role_id, role_title, masco_code, task_summary, remote_possibility, ai_exposure (low/medium/high), flexible_role, role_embedding[384]. Drives occupation match and readiness AI weighting.
- `skill_taxonomy` - skill_id, canonical_name, aliases, skill_type (technical/soft/digital), embedding[384]. Doubles as the master skill dictionary for CV skill_mentions (rapidfuzz over canonical_name + aliases) and the semantic skill match.
- `role_skills` - role_id, skill_id, skill_name, skill_type, importance (0-100). The two-band gap source.
- `caregiving_map` - break_activity to reframed_label. The break reframer.
- `employers`, `role_sector_map` - loaded but not queried in It1 (Employer Fit deferred to It2).

The occupation classifier is not a table: it is trained offline (`10 Data Governance/ImprovedVersion/04_REPRODUCIBILITY/train_classifier.py`) from `D9_classifier_training_examples.csv` into a pickle the service loads at startup.

### pgvector adaptation

The shipped `.sql` is the No-pgvector compatibility build: embeddings are stored as `double precision[384]`. To run cosine search in Postgres, a one-time migration adapts the two embedding columns to `vector(384)`. Import data is unchanged; only the column type and index change.

`db/01_pgvector_migrate.sql` runs after the import:

- `CREATE EXTENSION IF NOT EXISTS vector;`
- `ALTER TABLE rerouteher.roles ALTER COLUMN role_embedding TYPE vector(384) USING role_embedding::vector;`
- `ALTER TABLE rerouteher.skill_taxonomy ALTER COLUMN embedding TYPE vector(384) USING embedding::vector;`
- `CREATE INDEX ON rerouteher.roles USING hnsw (role_embedding vector_cosine_ops);`
- `CREATE INDEX ON rerouteher.skill_taxonomy USING hnsw (embedding vector_cosine_ops);`

## 3. Endpoints

### 3.1 POST /api/cv/parse (E2 / US2.1) - NLP CV extractor

Request: multipart, `file` = PDF, up to 10MB.

Response 200: `{ "cv": { "raw_text": str, "experiences": [{ "title","organisation","start","end","description" }], "skill_mentions": [str] } }`

Response 400: `{ "error": str }`

Logic:

1. Validate content-type is PDF and size <= 10MB, else 400.
2. PyMuPDF (`fitz`) read text blocks, sort by bounding-box (x, y) so multi-column CVs keep reading order.
3. No text layer (scanned/image PDF) -> 400 "unreadable, please upload a text-based PDF" (no OCR).
4. spaCy `en_core_web_sm` NER + rule patterns (section headers, date-range regex) -> experiences[].
5. rapidfuzz fuzzy-match text against `skill_taxonomy` (canonical_name + aliases) -> skill_mentions[].
6. Never emit email/phone (no PII). In-memory only; nothing stored.

### 3.2 POST /api/snapshot/generate (E3 / US3.1-3.4)

Request: `{ "cv": {...}, "break": { "duration_years": number, "activities": [str] } }`

Response 200: `{ "professional_skills": [{ "skill","source":"experience","evidence" }], "reframed_skills": [{ "skill","source":"break","from_activity" }], "previous_occupation": { "role","confidence","method":"classifier"|"embedding" } | null, "recommended_roles": [{ "role","similarity" }] }`

Logic:

1. Skills (hybrid): exact = spaCy PhraseMatcher over `skill_taxonomy.aliases`; semantic = embed CV spans with `all-MiniLM-L6-v2` (CPU) and pgvector cosine vs `skill_taxonomy.embedding`; merge to canonical skill_id, dedupe -> professional_skills.
2. Reframed: map `break.activities` through `caregiving_map` -> reframed_skills (reframed_label).
3. Previous occupation (Tier 1): TF-IDF + Logistic Regression classifier (`char_wb`, 3-5) over profile text -> top role + confidence, method "classifier".
4. Fallback (Tier 2): if confidence < 0.65, pgvector cosine of the profile embedding vs `roles.role_embedding` -> top role, method "embedding".
5. Recommended roles: previous occupation pinned index 0 (default) + 2 nearest other roles by pgvector kNN over `roles.role_embedding`.
6. No confident match -> previous_occupation null, recommended_roles empty.

### 3.3 POST /api/gap/compute (E4 / US4.1-4.3)

Request: `{ "skills": [{ "skill","source" }], "target_role": str }`

Response 200: `{ "readiness": number, "skills_have": [str], "gaps": [{ "skill","band":"role"|"ai_digital","importance","uplift" }] }`

Logic:

1. Load `role_skills` for target_role, split band 1 (technical/soft) vs band 2 (digital), read `roles.ai_exposure`.
2. Per-band coverage = sum(importance of covered skills) / sum(importance of all required skills in band).
3. readiness % = ((1 - ai_exposure) * role_coverage + ai_exposure * ai_digital_coverage) * 100 (ai_exposure as 0-1 blend weight).
4. Gaps = required skills not covered; band-2 importance scaled by the {low 0.8, medium 1.0, high 1.2} factor for ranking.
5. uplift per gap = readiness recomputed with that skill covered, minus current readiness (drives the "+X% if learned" badge).
6. Merge both bands, rank by uplift desc (tie-break importance), return top-3 as focus areas (full per-band lists available underneath).

All three endpoints are stateless, computed at request time, nothing written to DB.

### Team still to finalise (placeholders in code, flagged)

- ai_exposure 0-1 blend weights (low/medium/high mapping).
- semantic skill cosine threshold.

## 4. Project structure

```
rerouteher-system/
  app/
    main.py                  # FastAPI app, lifespan (load models/vectors), router mounts, CORS
    config.py                # env-driven settings (DB url, thresholds, model paths)
    db.py                    # async engine + session (asyncpg), pgvector registration
    api/
      cv.py                  # POST /api/cv/parse
      snapshot.py            # POST /api/snapshot/generate
      gap.py                 # POST /api/gap/compute
    schemas/
      cv.py  snapshot.py  gap.py     # Pydantic v2 request/response models
    services/
      cv_extractor.py        # PyMuPDF + spaCy NER + rapidfuzz
      snapshot.py            # hybrid skills + occupation cascade + kNN
      gap.py                 # two-band readiness + uplift ranking
      embedder.py            # all-MiniLM-L6-v2 wrapper (load once)
      classifier.py          # loads occupation_clf.pkl, predict + confidence
    repositories/
      roles.py  skills.py  caregiving.py   # pgvector queries (cosine, kNN)
  ml/
    occupation_clf.pkl       # copied from data team train_classifier.py output
  db/
    ReRouteHer_..._Import.sql # the data-governance import (adapted to pgvector)
    01_pgvector_migrate.sql  # CREATE EXTENSION + ALTER columns to vector(384) + index
  tests/
    test_cv_parse.py  test_snapshot.py  test_gap.py
  requirements.txt
  .env.example
  README.md
```

### requirements.txt (proposed, to finalise pins)

```
fastapi
uvicorn[standard]
pydantic>=2
sqlalchemy[asyncio]>=2
asyncpg
pgvector
python-multipart
pymupdf
spacy
rapidfuzz
sentence-transformers
scikit-learn
numpy
python-dotenv
pytest
httpx
```

Post-install: `python3 -m spacy download en_core_web_sm`.

### Startup (FastAPI lifespan)

Load-once, held in app state so there is no per-request cold start:

1. Embedder: `SentenceTransformer("all-MiniLM-L6-v2")` (CPU).
2. Classifier: unpickle `ml/occupation_clf.pkl` plus its TF-IDF vectorizer.
3. spaCy pipeline + PhraseMatcher built from `skill_taxonomy.aliases`.
4. DB pool (asyncpg) with `register_vector` so `vector(384)` round-trips as numpy.

Reference tables stay in Postgres; embeddings are queried via pgvector cosine (`<=>`) with an hnsw index. Guest requests write nothing.
