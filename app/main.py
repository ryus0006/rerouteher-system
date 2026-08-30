"""ReRouteHer backend. Guest journey: CV parse -> snapshot -> gap.

Models and reference-derived assets load once at startup (lifespan) and live on
app.state so requests have no cold start. Model loading is resilient: if an ML asset
is missing, the app still boots and the affected endpoint degrades at call time.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import cv, gap, snapshot
from app.config import get_settings
from app.core.logging import RequestLoggingMiddleware, configure_logging
from app.db import SessionLocal
from app.repositories import skills as skills_repo
from app.services.cv_extractor import CVExtractor
from app.services.embedder import Embedder
from app.services.gap import GapService
from app.services.occupation_matcher import EscoTfidfMatcher
from app.services.snapshot import SnapshotService

configure_logging()
logger = logging.getLogger("rerouteher")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # 1. Embedder (all-MiniLM-L6-v2, CPU). Optional so the app boots without the model cache.
    embedder = None
    try:
        embedder = Embedder(settings.embedding_model)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Embedder not loaded (%s); snapshot semantic match disabled", exc)

    # 2. ESCO TF-IDF matcher (Tier 1, logged). None if the artifact is absent.
    tfidf_matcher = EscoTfidfMatcher.load(settings.tfidf_model_path)
    if tfidf_matcher is None:
        logger.warning("TF-IDF matcher not found at %s; Tier 1 occupation logging disabled", settings.tfidf_model_path)

    # 3. spaCy pipeline + alias dictionary.
    nlp = None
    alias_pairs: list[tuple[str, str]] = []
    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
        async with SessionLocal() as session:
            alias_pairs = await skills_repo.load_alias_dictionary(session)
    except Exception as exc:  # noqa: BLE001
        logger.warning("spaCy/alias dictionary not loaded (%s); CV parsing degraded", exc)

    skill_dictionary = sorted({term for _, term in alias_pairs})

    # Wire services onto app state. The snapshot service loads its own skill lookup
    # lazily from the DB on first request and caches it.
    app.state.cv_extractor = CVExtractor(
        nlp=nlp, skill_dictionary=skill_dictionary, embedder=embedder
    )
    app.state.snapshot_service = SnapshotService(
        settings=settings, embedder=embedder, tfidf_matcher=tfidf_matcher
    )
    app.state.gap_service = GapService(settings=settings)

    logger.info(
        "startup: embedder=%s tfidf=%s spacy=%s skill_dict=%d",
        embedder is not None,
        tfidf_matcher is not None,
        nlp is not None,
        len(skill_dictionary),
    )

    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="ReRouteHer API", version="0.1.0", lifespan=lifespan)

    # Allow the browser client to call the API directly.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(RequestLoggingMiddleware)

    app.include_router(cv.router)
    app.include_router(snapshot.router)
    app.include_router(gap.router)

    @app.get("/api/health", tags=["meta"])
    async def health():
        return {"status": "ok"}

    return app


app = create_app()
