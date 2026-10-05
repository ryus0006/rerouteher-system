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

from app.api import account, companion, cv, employers, gap, interview, learning, snapshot
from app.config import get_settings
from app.core.logging import RequestLoggingMiddleware, configure_logging
from app.db import SessionLocal
from app.repositories import skills as skills_repo
from app.services.account import AccountService
from app.services.cv_extractor import CVExtractor
from app.services.cv_generation import CvGenerationService
from app.services.embedder import Embedder
from app.services.companion import CompanionService
from app.services.employers import EmployerService
from app.services.gap import GapService
from app.services.interview import InterviewService
from app.services.interview_feedback import InterviewFeedbackService
from app.services.interview_privacy import TranscriptRedactor
from app.services.job_search import JobSearchService
from app.services.job_sources import (
    FounditJobSource,
    JoobleJobSource,
    TavilyGeminiJobSource,
)
from app.services.learning import LearningService
from app.services.learning_fill import LearningFillService
from app.services.llm import GeminiClient
from app.services.profile_skills import ProfileSkillService
from app.services.tavily import TavilySearcher
from app.services.occupation_matcher import EscoTfidfMatcher
from app.services.reranker import CrossEncoderReranker
from app.services.snapshot import SnapshotService
from app.services.transcription import WhisperTranscriber

configure_logging()
logger = logging.getLogger("rerouteher")

_INTERVIEW_CLEANUP_INTERVAL_S = 24 * 60 * 60


async def _interview_cleanup_loop(interview_service: InterviewService) -> None:
    """Runs once immediately, then once every 24 hours. A failed run is logged (no
    secrets, prompts, or transcript content -- only a count and the exception type)
    and never stops the next scheduled run."""
    while True:
        try:
            async with SessionLocal() as session:
                purged = await interview_service.purge_expired_content(session)
                await session.commit()
            logger.info("interview retention cleanup: purged=%d", purged)
        except Exception as exc:  # noqa: BLE001
            logger.warning("interview retention cleanup failed (%s)", type(exc).__name__)
        await asyncio.sleep(_INTERVIEW_CLEANUP_INTERVAL_S)


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
    app.state.profile_skill_service = ProfileSkillService(
        embedder=embedder,
        gap_service=app.state.gap_service,
    )
    # One Gemini client is shared by the companion and the learning fill.
    llm = GeminiClient.from_settings(settings)
    app.state.cv_generation_service = CvGenerationService(llm)
    # Companion resolves a self-declared occupation to a role via the snapshot service
    # (same embedding + rerank path), so it is wired here where that service exists.
    app.state.companion_service = CompanionService(
        llm=llm,
        role_resolver=app.state.snapshot_service,
        profile_skill_service=app.state.profile_skill_service,
    )
    # E6 on-demand learning fill: Tavily search + Gemini pick, run in the background
    # after a gap is computed. Disabled (degrades to the YouTube fallback) if either
    # client is unavailable or learning_fill_enabled is false.
    tavily_searcher = TavilySearcher.from_settings(settings)
    app.state.learning_fill_service = LearningFillService(
        llm,
        tavily_searcher,
        enabled=settings.learning_fill_enabled,
        url_timeout_s=settings.learning_fill_url_timeout_s,
        candidates=settings.learning_fill_candidates,
    )

    job_sources = []
    jooble = JoobleJobSource.from_settings(settings)
    if jooble is not None:
        job_sources.append(("jooble", jooble))
    job_sources.append(("foundit", FounditJobSource(
        base_url=settings.foundit_base_url,
        timeout_s=settings.foundit_timeout_s,
    )))
    job_sources.append(("tavily", TavilyGeminiJobSource(tavily_searcher, llm)))
    app.state.job_search_service = JobSearchService(
        sources=job_sources,
        location=settings.job_search_location,
    )
    app.state.employer_service = EmployerService(
        job_search=app.state.job_search_service,
    )

    # E7 AI Interview Coach. A failed Whisper load degrades to an unavailable
    # transcriber (health reports degraded; endpoints return transcription_unavailable)
    # rather than failing startup. No audio/transcript content is ever logged here.
    transcriber = WhisperTranscriber.load(
        settings.whisper_model_path,
        threads=settings.whisper_threads,
        max_bytes=settings.interview_max_audio_bytes,
        max_seconds=settings.interview_max_audio_seconds,
        ffmpeg_timeout_s=settings.interview_ffmpeg_timeout_s,
        language=settings.interview_transcription_language,
    )
    # Reuses the same spaCy pipeline already loaded for CV parsing (PERSON/GPE/LOC
    # redaction) and the same shared Gemini client used by the companion.
    redactor = TranscriptRedactor(nlp=nlp, enabled=settings.interview_redaction_enabled)
    feedback_service = InterviewFeedbackService(llm)
    app.state.interview_service = InterviewService(
        transcriber=transcriber,
        redactor=redactor,
        feedback_service=feedback_service,
        retention_days=settings.interview_content_retention_days,
    )
    # Health reports only availability booleans -- never secrets, prompts, model
    # output, or user data.
    app.state.interview_health = {
        "transcription_available": transcriber.available,
        "feedback_available": llm is not None,
    }

    cleanup_task = asyncio.create_task(_interview_cleanup_loop(app.state.interview_service))

    logger.info(
        "startup: embedder=%s tfidf=%s reranker=%s spacy=%s skill_dict=%d learning_fill=%s "
        "whisper=%s feedback=%s",
        embedder is not None,
        tfidf_matcher is not None,
        reranker.model_id if reranker else None,
        nlp is not None,
        len(skill_dictionary),
        app.state.learning_fill_service.enabled,
        transcriber.available,
        llm is not None,
    )

    yield

    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass


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

    # Stateless services, so they are wired here rather than in lifespan.
    app.state.account_service = AccountService()
    app.state.learning_service = LearningService()
    app.state.employer_service = EmployerService()
    # companion_service is wired in the lifespan: it needs snapshot_service as its
    # role resolver, which is only built there.

    app.include_router(cv.router)
    app.include_router(snapshot.router)
    app.include_router(gap.router)
    app.include_router(account.router)
    app.include_router(learning.router)
    app.include_router(employers.router)
    app.include_router(companion.router)
    app.include_router(interview.router)

    @app.get("/api/health", tags=["meta"])
    async def health():
        # The API process itself is always 200 here; "status" reflects whether the E7
        # interview dependencies (local transcription, Gemini feedback) are available.
        interview_health = getattr(app.state, "interview_health", None)
        if interview_health is None:
            return {"status": "ok"}
        degraded = not (interview_health["transcription_available"] and interview_health["feedback_available"])
        return {
            "status": "degraded" if degraded else "ok",
            "interview": interview_health,
        }

    return app


app = create_app()
