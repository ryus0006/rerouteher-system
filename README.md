# ReRouteHer System (Backend)

FastAPI backend for the ReRouteHer Iteration 1 guest journey: CV parse, skill snapshot, and readiness/gap.

This is a Monash FIT5120 academic project. See `plan.md` for the full design derived from the user stories and data-governance deliverables.

## Endpoints (It1)

| Endpoint | Story | Job |
|---|---|---|
| `POST /api/cv/parse` | E2 / US2.1 | PDF to structured CV (NLP extractor) |
| `POST /api/snapshot/generate` | E3 / US3.1-3.4 | skills + previous occupation + recommended roles |
| `POST /api/gap/compute` | E4 / US4.1-4.3 | readiness % + merged top-3 gap |

All three are stateless and computed at request time. Guests persist nothing.

## Run with Docker (primary path)

The whole stack is containerized: a pgvector Postgres and the FastAPI API.

```bash
./scripts/local-start.sh     # build + start, waits for DB and API health, prints a banner
./scripts/local-stop.sh      # stop and reset the DB (add --keep-data to preserve it)
```

If port 5432 is taken on your host: `DB_PORT=5433 ./scripts/local-start.sh`.

Under the hood this is just:

```bash
docker compose up --build
```

On first boot the `db` service auto-loads `db/00_import.sql` (reference data) then `db/01_pgvector_migrate.sql` (converts the embedding columns to pgvector `vector(384)` and adds cosine indexes). The API waits for the DB healthcheck before starting.

- API: http://localhost:8080/docs (or `/api/health`)
- Postgres: localhost:5432 (db `rerouteher`, user/pass `postgres`)

The TF-IDF occupation model lives at `ml/tfidf_logreg.joblib` (produced by the data team) and is mounted read-only into the API container. The app boots without it (occupation falls back to embedding).

To re-run DB init from scratch: `docker compose down -v` (wipes the `pgdata` volume) then `up` again.

## Run locally without Docker (optional)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip3 install -r requirements.txt
python3 -m spacy download en_core_web_sm
cp .env.example .env   # edit DATABASE_URL to your local Postgres

psql -v ON_ERROR_STOP=1 -d rerouteher -f db/00_import.sql
psql -v ON_ERROR_STOP=1 -d rerouteher -f db/01_pgvector_migrate.sql

uvicorn app.main:app --reload --port 8080
```

## Test

```bash
pytest
```

## Status

All three It1 endpoints are implemented and unit-tested:

- **`/api/cv/parse`** - PyMuPDF column-aware text, regex + spaCy experience segmentation, rapidfuzz skill matching, PII redaction, no-OCR unreadable path.
- **`/api/snapshot/generate`** - hybrid skill extraction (exact alias + semantic embedding), two-tier occupation cascade (classifier -> embedding fallback), recommended roles (previous pinned first + 2 nearest by kNN), break reframing.
- **`/api/gap/compute`** - two-band readiness %, per-gap uplift, merged top-3 focus list.

The app boots even when models are absent (embedder/classifier optional) so the frontend can integrate against the schemas. `pytest` runs the fast suite without a DB or torch (repos/models faked for the snapshot); the real DB + model path is exercised by running the container.

## Endpoints (It3 addition: E7 - AI Interview Coach)

Backend only this iteration; a separate UI team integrates it later.

| Endpoint | Story | Job |
|---|---|---|
| `GET /api/interview/setup` | E7 / US7.1 | her selected + matched roles, focus choices |
| `GET /api/interview/sessions` | E7 / US7.2 | all her sessions, newest first |
| `POST /api/interview/sessions` | E7 / US7.1 | create (201) or return (200) a session for a role + focus |
| `GET /api/interview/sessions/{id}` | E7 / US7.2-7.4 | full session detail: 5 question slots, every attempt |
| `POST /api/interview/sessions/{id}/refresh` | E7 / US7.2 | replace with 5 new questions (201) |
| `DELETE /api/interview/sessions/{id}` | E7 / US7.2 | delete a session (204) |
| `POST /api/interview/sessions/{id}/questions/{n}/attempts` | E7 / US7.2-7.3 | upload a spoken answer (multipart), transcribe + feedback synchronously |
| `POST /api/interview/attempts/{id}/feedback` | E7 / US7.3 | retry feedback from the saved transcript, no re-recording |
| `GET /api/interview/areas` | E7 / US7.4 | recurring improvement areas + strengths across her sessions |

All routes require a signed-in session (401 otherwise); a missing or another user's resource returns 404 either way.

## Status (E7)

- **Setup** - saved journey role + matched `recommended_roles`, deduplicated in journey order, confirmed against `roles`; 409/422 if incomplete or out of set.
- **Sessions** - one persistent session per (role, focus); 5 questions (2 foundation / 2 intermediate / 1 advanced), general/role-specific/mixed pools, no repeats, refresh excludes the old 5 where alternatives exist.
- **Recording** - FFmpeg normalise to mono 16kHz WAV, local whisper.cpp (one model, one semaphore, multilingual auto-detect) transcribes synchronously; raw audio never persisted.
- **Privacy** - regex + spaCy redaction (email/phone/NRIC/address/PERSON/GPE/LOC) before anything is stored or sent to Gemini.
- **Feedback** - one forced `submit_interview_feedback` tool call, no numeric score, titles/tags derived from `ai_evaluation_rubric` not the model; Gemini failure preserves the transcript and returns 503 (retryable).
- **Retention** - 30-day background purge of transcript/feedback detail (tags + metadata kept); areas use latest-ready-per-question, sorted by frequency then title.
- **Health** - `/api/health` reports `interview.transcription_available` / `feedback_available`; `degraded` if either is false, API itself still 200.

The whisper.cpp model (`ggml-base.bin`) is downloaded + SHA-256-verified at image build time, not committed. `pytest` runs the full E7 suite without a DB, torch, or a real Gemini call (353/353 passing); `docker compose config` and the API image build both verified. A fresh Compose DB init (E7 table creation + the 7,592/10/44,902 seed counts against live Postgres) is not yet exercised here - `db/02_data.sql` (db team's base dump) is not present locally.

## Not in It1

Employer Fit (E4 / US4.4-4.5) is deferred to Iteration 2. The `employers` and `role_sector_map` tables import but are not queried yet.
