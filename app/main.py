"""ReRouteHer backend. Guest journey: CV parse -> snapshot -> gap.

Models and reference-derived assets load once at startup (lifespan) and live on
app.state so requests have no cold start. Model loading is resilient: if an ML asset
is missing, the app still boots and the affected endpoint degrades at call time.
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from app.api import account, cv, employers, gap, learning, snapshot
from app.config import get_settings
from app.core.logging import RequestLoggingMiddleware, configure_logging
from app.db import SessionLocal
from app.repositories import skills as skills_repo
from app.services.account import AccountService
from app.services.cv_extractor import CVExtractor
from app.services.embedder import Embedder
from app.services.employers import EmployerService
from app.services.gap import GapService
from app.services.learning import LearningService
from app.services.occupation_matcher import EscoTfidfMatcher
from app.services.reranker import CrossEncoderReranker
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
        logger.warning("Embedder not loaded (%s); snapshot semantic match disabled", exc, exc_info=True)

    # 2. ESCO TF-IDF matcher (Tier 1, logged). None if the artifact is absent.
    tfidf_matcher = EscoTfidfMatcher.load(settings.tfidf_model_path)
    if tfidf_matcher is None:
        logger.warning("TF-IDF matcher not found at %s; Tier 1 occupation logging disabled", settings.tfidf_model_path)

    # 2b. Occupation reranker (local cross-encoder). Downloaded from HF at first boot and
    # cached; degrades to None (today's ordering) if no model in the ladder loads.
    reranker = None
    if settings.rerank_enabled:
        reranker = CrossEncoderReranker.load(
            settings.rerank_model_id_list, settings.rerank_cache_dir
        )

    # 3. spaCy pipeline + alias dictionary.
    nlp = None
    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
    except Exception as exc:  # noqa: BLE001
        logger.warning("spaCy not loaded (%s); CV parsing degraded", exc, exc_info=True)

    # The database can still be finishing its own startup when the app boots, so retry
    # this one read rather than degrade to an empty alias dictionary on a transient miss.
    alias_pairs: list[tuple[str, str]] = []
    for attempt in range(1, 11):
        try:
            async with SessionLocal() as session:
                alias_pairs = await skills_repo.load_alias_dictionary(session)
            break
        except Exception as exc:  # noqa: BLE001
            if attempt == 10:
                logger.warning("alias dictionary not loaded after retries (%s); CV parsing degraded", exc)
            else:
                await asyncio.sleep(2)

    skill_dictionary = sorted({term for _, term in alias_pairs})

    # Wire services onto app state. The snapshot service loads its own skill lookup
    # lazily from the DB on first request and caches it.
    app.state.cv_extractor = CVExtractor(
        nlp=nlp, skill_dictionary=skill_dictionary, embedder=embedder
    )
    app.state.snapshot_service = SnapshotService(
        settings=settings, embedder=embedder, tfidf_matcher=tfidf_matcher, reranker=reranker
    )
    app.state.gap_service = GapService(settings=settings)

    logger.info(
        "startup: embedder=%s tfidf=%s reranker=%s spacy=%s skill_dict=%d",
        embedder is not None,
        tfidf_matcher is not None,
        reranker.model_id if reranker else None,
        nlp is not None,
        len(skill_dictionary),
    )

    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="ReRouteHer API", version="0.1.0", lifespan=lifespan)

    # Allow the browser client to call the API directly. Credentials are on so the
    # session cookie set by the account endpoints rides with subsequent requests.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Signed session cookie carries the signed-in username (E5).
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.session_secret,
        https_only=settings.session_https_only,
        same_site=settings.session_same_site,
    )

    app.add_middleware(RequestLoggingMiddleware)

    # Stateless service, so it is wired here rather than in lifespan.
    app.state.account_service = AccountService()
    app.state.learning_service = LearningService()
    app.state.employer_service = EmployerService()

    app.include_router(cv.router)
    app.include_router(snapshot.router)
    app.include_router(gap.router)
    app.include_router(account.router)
    app.include_router(learning.router)
    app.include_router(employers.router)

    @app.get("/api/health", tags=["meta"])
    async def health():
        return {"status": "ok"}

    return app


app = create_app()
