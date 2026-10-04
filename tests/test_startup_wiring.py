"""The lifespan loads the reranker and the E7 interview dependencies, and the interview
retention cleanup loop runs once immediately, reschedules every 24 hours, survives a
failed run, and can be cancelled cleanly at shutdown.

Heavy model loads are stubbed so this runs without torch, the joblib artifact, the
real whisper model, or a DB.
"""
import asyncio

import pytest

import app.main as main_mod
from app.services.interview import InterviewService
from app.services.cv_generation import CvGenerationService
from app.services.reranker import CrossEncoderReranker
from app.services.transcription import WhisperTranscriber

pytestmark = pytest.mark.asyncio


class _SentinelReranker:
    model_id = "sentinel-model"


class _StubTranscriber:
    def __init__(self, available: bool) -> None:
        self.available = available


async def test_lifespan_wires_reranker_and_interview_dependencies(monkeypatch):
    sentinel = _SentinelReranker()
    stub_transcriber = _StubTranscriber(available=True)
    captured: dict = {}

    def _no_embedder(model_name):
        raise RuntimeError("skip embedder in test")

    monkeypatch.setattr(main_mod, "Embedder", _no_embedder)
    monkeypatch.setattr(main_mod.EscoTfidfMatcher, "load", staticmethod(lambda path: None))
    monkeypatch.setattr(
        CrossEncoderReranker, "load", classmethod(lambda cls, ids, cache: sentinel)
    )
    monkeypatch.setattr(
        WhisperTranscriber, "load", staticmethod(lambda *a, **k: stub_transcriber)
    )

    real_init = main_mod.SnapshotService.__init__

    def spy(self, **kwargs):
        captured.update(kwargs)
        real_init(self, **kwargs)

    monkeypatch.setattr(main_mod.SnapshotService, "__init__", spy)

    app = main_mod.create_app()
    async with main_mod.lifespan(app):
        assert isinstance(app.state.interview_service, InterviewService)
        assert isinstance(app.state.cv_generation_service, CvGenerationService)
        assert app.state.interview_health["transcription_available"] is True
        assert isinstance(app.state.interview_health["feedback_available"], bool)
        assert app.state.job_search_service is not None
        assert app.state.employer_service._job_search is app.state.job_search_service

    assert captured.get("reranker") is sentinel
    assert app.state.snapshot_service is not None


async def test_lifespan_degrades_when_whisper_unavailable(monkeypatch):
    stub_transcriber = _StubTranscriber(available=False)

    monkeypatch.setattr(main_mod, "Embedder", lambda model_name: (_ for _ in ()).throw(RuntimeError()))
    monkeypatch.setattr(main_mod.EscoTfidfMatcher, "load", staticmethod(lambda path: None))
    monkeypatch.setattr(CrossEncoderReranker, "load", classmethod(lambda cls, ids, cache: None))
    monkeypatch.setattr(
        WhisperTranscriber, "load", staticmethod(lambda *a, **k: stub_transcriber)
    )

    app = main_mod.create_app()
    async with main_mod.lifespan(app):
        # the app still boots; it degrades rather than failing startup
        assert app.state.interview_service is not None
        assert isinstance(app.state.cv_generation_service, CvGenerationService)
        assert app.state.interview_health["transcription_available"] is False


# -- retention cleanup loop -----------------------------------------------------------

class _FakeDbSession:
    def __init__(self):
        self.committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def commit(self):
        self.committed = True


class _FakeSessionLocal:
    def __init__(self):
        self.sessions: list[_FakeDbSession] = []

    def __call__(self):
        s = _FakeDbSession()
        self.sessions.append(s)
        return s


async def _run_loop_once_then_stop(monkeypatch, interview_service):
    """Lets _interview_cleanup_loop run its body once, then raises CancelledError from
    inside asyncio.sleep so the infinite loop exits after exactly one iteration."""
    sleep_calls = []

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)
        raise asyncio.CancelledError()

    monkeypatch.setattr(main_mod.asyncio, "sleep", fake_sleep)
    with pytest.raises(asyncio.CancelledError):
        await main_mod._interview_cleanup_loop(interview_service)
    return sleep_calls


async def test_cleanup_runs_immediately_and_then_schedules_once_every_24_hours(monkeypatch):
    fake_session_local = _FakeSessionLocal()
    monkeypatch.setattr(main_mod, "SessionLocal", fake_session_local)

    purge_calls = []

    class FakeInterview:
        async def purge_expired_content(self, session):
            purge_calls.append(session)
            return 3

    sleep_calls = await _run_loop_once_then_stop(monkeypatch, FakeInterview())

    assert len(purge_calls) == 1  # ran once, immediately
    assert sleep_calls == [main_mod._INTERVIEW_CLEANUP_INTERVAL_S]  # then scheduled for 24h
    assert fake_session_local.sessions[0].committed is True  # explicit commit per cleanup run


async def test_cleanup_failure_is_logged_and_does_not_stop_the_next_scheduled_run(monkeypatch, caplog):
    import logging

    monkeypatch.setattr(main_mod, "SessionLocal", _FakeSessionLocal())

    class BoomInterview:
        async def purge_expired_content(self, session):
            raise RuntimeError("db exploded")

    caplog.set_level(logging.WARNING, logger="rerouteher")
    sleep_calls = await _run_loop_once_then_stop(monkeypatch, BoomInterview())

    assert sleep_calls == [main_mod._INTERVIEW_CLEANUP_INTERVAL_S]  # still reached the next schedule
    assert any("interview retention cleanup failed" in r.getMessage() for r in caplog.records)
    # the log carries metadata only, never a stack trace string with internals
    assert all("db exploded" not in r.getMessage() for r in caplog.records)


async def test_cleanup_task_can_be_cancelled_and_awaited_cleanly(monkeypatch):
    monkeypatch.setattr(main_mod, "SessionLocal", _FakeSessionLocal())

    class SlowInterview:
        async def purge_expired_content(self, session):
            await asyncio.sleep(10)
            return 0

    task = asyncio.create_task(main_mod._interview_cleanup_loop(SlowInterview()))
    await asyncio.sleep(0)  # let the task start and reach its await point
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert task.done()
