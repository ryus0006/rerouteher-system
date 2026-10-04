"""E7 interview API: authentication, the full success/error status matrix, multipart
size enforcement, and commit/rollback behaviour. The interview service is faked;
app/services/interview.py and its own tests own the business-logic behaviour this
layer only needs to expose correctly over HTTP.
"""
from datetime import datetime, timezone

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

import app.api.interview as interview_api
from app.db import get_session
from app.main import create_app
from app.schemas.interview import (
    AreasOut,
    AttemptOut,
    FeedbackItemOut,
    InterviewSetupOut,
    QuestionSlotOut,
    RoleOut,
    SessionDetailOut,
    SessionSummaryOut,
)
from app.services.interview import InterviewError, RecordingOutcome

_NOW = datetime(2026, 10, 3, tzinfo=timezone.utc)


def _attempt_out(**over):
    base = dict(
        response_id=1,
        attempt_no=1, transcript="I led a team.", feedback_status="ready",
        feedback_summary="Clear and relevant.",
        strengths=[FeedbackItemOut(criterion_id="EVAL-01", title="Relevance To Question", detail="On topic.")],
        improvements=[], detected_language="en", duration_s=5.0, error_code=None,
        created_at=_NOW, updated_at=_NOW, content_expired=False,
    )
    base.update(over)
    return AttemptOut(**base)


def _session_detail(**over):
    base = dict(
        session_id="s1", role=RoleOut(role_id="R1", role_title="Data Analyst"),
        practice_focus="general", status="active", created_at=_NOW, updated_at=_NOW,
        questions=[
            QuestionSlotOut(
                sequence_no=1, question_id="GEN-001", question_text="Tell me about yourself.",
                category="introduction_and_background", difficulty="foundation",
                role_id=None, kind="general",
                attempts=[_attempt_out()],
            )
        ],
    )
    base.update(over)
    return SessionDetailOut(**base)


class FakeSession:
    def __init__(self):
        self.committed = False

    async def commit(self):
        self.committed = True


class FakeInterviewService:
    def __init__(self):
        self.setup_result = InterviewSetupOut(selected_role=RoleOut(role_id="R1", role_title="Data Analyst"))
        self.setup_error = None
        self.sessions_result = [
            SessionSummaryOut(
                session_id="s1", role=RoleOut(role_id="R1", role_title="Data Analyst"),
                practice_focus="general", status="active", progress=0,
                created_at=_NOW, updated_at=_NOW,
            )
        ]
        self.create_result = (_session_detail(), True)
        self.create_error = None
        self.get_result = _session_detail()
        self.get_error = None
        self.refresh_result = _session_detail(session_id="s2")
        self.refresh_error = None
        self.delete_error = None
        self.process_result = RecordingOutcome(status="ready", attempt=_attempt_out())
        self.process_error = None
        self.retry_result = RecordingOutcome(status="ready", attempt=_attempt_out(attempt_no=2))
        self.retry_error = None
        self.areas_result = AreasOut()
        self.calls: list[tuple] = []

    async def get_setup(self, session, username):
        self.calls.append(("get_setup", username))
        if self.setup_error:
            raise self.setup_error
        return self.setup_result

    async def list_sessions(self, session, username):
        self.calls.append(("list_sessions", username))
        return self.sessions_result

    async def create_session(self, session, username, role_id, practice_focus):
        self.calls.append(("create_session", username, role_id, practice_focus))
        if self.create_error:
            raise self.create_error
        return self.create_result

    async def get_session(self, session, username, session_id):
        self.calls.append(("get_session", username, session_id))
        if self.get_error:
            raise self.get_error
        return self.get_result

    async def refresh_session(self, session, username, session_id):
        self.calls.append(("refresh_session", username, session_id))
        if self.refresh_error:
            raise self.refresh_error
        return self.refresh_result

    async def delete_session(self, session, username, session_id):
        self.calls.append(("delete_session", username, session_id))
        if self.delete_error:
            raise self.delete_error

    async def process_recording(self, session, username, session_id, sequence_no, audio):
        self.calls.append(("process_recording", username, session_id, sequence_no, audio))
        if self.process_error:
            raise self.process_error
        return self.process_result

    async def retry_feedback(self, session, username, response_id):
        self.calls.append(("retry_feedback", username, response_id))
        if self.retry_error:
            raise self.retry_error
        return self.retry_result

    async def get_areas(self, session, username):
        self.calls.append(("get_areas", username))
        return self.areas_result


@pytest.fixture
def app_client():
    app = create_app()
    fake_service = FakeInterviewService()
    app.state.interview_service = fake_service
    fake_sessions: list[FakeSession] = []

    async def _fake_session():
        s = FakeSession()
        fake_sessions.append(s)
        yield s

    app.dependency_overrides[get_session] = _fake_session

    # Test-only route: signs in via the real SessionMiddleware without needing the
    # account service, so cookies behave exactly as they do in production.
    @app.post("/_test/sign-in")
    async def _sign_in(request: Request):
        body = await request.json()
        request.session["username"] = body["username"]
        return {"ok": True}

    client = TestClient(app)
    return client, fake_service, fake_sessions


def _sign_in(client, username="aisha"):
    client.post("/_test/sign-in", json={"username": username})


_ENDPOINTS = [
    ("GET", "/api/interview/setup", None),
    ("GET", "/api/interview/sessions", None),
    ("POST", "/api/interview/sessions", {"role_id": "R1", "practice_focus": "general"}),
    ("GET", "/api/interview/sessions/s1", None),
    ("POST", "/api/interview/sessions/s1/refresh", None),
    ("DELETE", "/api/interview/sessions/s1", None),
    ("POST", "/api/interview/attempts/1/feedback", None),
    ("GET", "/api/interview/areas", None),
]


@pytest.mark.parametrize("method,path,body", _ENDPOINTS)
def test_every_endpoint_requires_authentication(app_client, method, path, body):
    client, fake_service, _ = app_client
    resp = client.request(method, path, json=body)
    assert resp.status_code == 401
    assert resp.json() == {"error": "authentication_required"}
    assert fake_service.calls == []


def test_record_attempt_requires_authentication(app_client):
    client, fake_service, _ = app_client
    resp = client.post(
        "/api/interview/sessions/s1/questions/1/attempts",
        files={"file": ("a.wav", b"fake-audio", "audio/wav")},
    )
    assert resp.status_code == 401
    assert fake_service.calls == []


# -- success paths --------------------------------------------------------------------

def test_get_setup_success(app_client):
    client, _, _ = app_client
    _sign_in(client)
    resp = client.get("/api/interview/setup")
    assert resp.status_code == 200
    assert resp.json()["selected_role"]["role_id"] == "R1"


def test_list_sessions_success(app_client):
    client, _, _ = app_client
    _sign_in(client)
    resp = client.get("/api/interview/sessions")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_create_session_returns_201_when_new(app_client):
    client, fake_service, sessions = app_client
    _sign_in(client)
    fake_service.create_result = (_session_detail(), True)
    resp = client.post("/api/interview/sessions", json={"role_id": "R1", "practice_focus": "general"})
    assert resp.status_code == 201
    assert resp.json()["session_id"] == "s1"
    assert sessions[-1].committed is True


def test_create_session_returns_200_when_existing(app_client):
    client, fake_service, sessions = app_client
    _sign_in(client)
    fake_service.create_result = (_session_detail(), False)
    resp = client.post("/api/interview/sessions", json={"role_id": "R1", "practice_focus": "general"})
    assert resp.status_code == 200
    assert sessions[-1].committed is True


def test_get_session_detail_success(app_client):
    client, _, _ = app_client
    _sign_in(client)
    resp = client.get("/api/interview/sessions/s1")
    assert resp.status_code == 200
    assert resp.json()["questions"][0]["attempts"][0]["feedback_status"] == "ready"


_QUESTION_DESIGN_CONTEXT_FIELDS = [
    "interview_method_sources", "authoring_method", "answer_framework",
    "answer_guidance", "strong_evidence_signals", "watch_out_for", "follow_up_question",
]


def test_session_detail_never_exposes_question_design_context(app_client):
    client, _, _ = app_client
    _sign_in(client)
    resp = client.get("/api/interview/sessions/s1")
    blob = resp.text
    for field_name in _QUESTION_DESIGN_CONTEXT_FIELDS:
        assert field_name not in blob


def test_record_attempt_response_never_exposes_question_design_context(app_client):
    client, _, _ = app_client
    _sign_in(client)
    resp = client.post(
        "/api/interview/sessions/s1/questions/1/attempts",
        files={"file": ("a.wav", b"fake-audio", "audio/wav")},
    )
    blob = resp.text
    for field_name in _QUESTION_DESIGN_CONTEXT_FIELDS:
        assert field_name not in blob


def test_refresh_session_returns_201(app_client):
    client, _, sessions = app_client
    _sign_in(client)
    resp = client.post("/api/interview/sessions/s1/refresh")
    assert resp.status_code == 201
    assert resp.json()["session_id"] == "s2"
    assert sessions[-1].committed is True


def test_delete_session_returns_204(app_client):
    client, _, sessions = app_client
    _sign_in(client)
    resp = client.delete("/api/interview/sessions/s1")
    assert resp.status_code == 204
    assert resp.content == b""
    assert sessions[-1].committed is True


def test_record_attempt_success(app_client):
    client, fake_service, sessions = app_client
    _sign_in(client)
    resp = client.post(
        "/api/interview/sessions/s1/questions/1/attempts",
        files={"file": ("a.wav", b"fake-audio", "audio/wav")},
    )
    assert resp.status_code == 200
    assert resp.json()["feedback_status"] == "ready"
    assert "error" not in resp.json()
    assert sessions[-1].committed is True
    call = fake_service.calls[0]
    assert call[0] == "process_recording" and call[1] == "aisha" and call[2] == "s1" and call[3] == 1


def test_retry_feedback_success(app_client):
    client, fake_service, sessions = app_client
    _sign_in(client)
    resp = client.post("/api/interview/attempts/7/feedback")
    assert resp.status_code == 200
    assert resp.json()["attempt_no"] == 2
    assert sessions[-1].committed is True
    assert fake_service.calls[0] == ("retry_feedback", "aisha", 7)


def test_get_areas_success(app_client):
    client, _, _ = app_client
    _sign_in(client)
    resp = client.get("/api/interview/areas")
    assert resp.status_code == 200
    assert resp.json() == {"improvements": [], "strengths": []}


# -- 503 feedback outcome carries the saved attempt and still commits -----------------

def test_record_attempt_feedback_error_returns_503_with_saved_attempt(app_client):
    client, fake_service, sessions = app_client
    _sign_in(client)
    fake_service.process_result = RecordingOutcome(
        status="feedback_error", attempt=_attempt_out(feedback_status="error", error_code="feedback_service_unavailable")
    )
    resp = client.post(
        "/api/interview/sessions/s1/questions/1/attempts",
        files={"file": ("a.wav", b"fake-audio", "audio/wav")},
    )
    assert resp.status_code == 503
    body = resp.json()
    assert body["error"] == "feedback_service_unavailable"
    assert body["attempt"]["transcript"] == "I led a team."  # the saved transcript is returned
    assert sessions[-1].committed is True  # preserved transcript is committed, not rolled back


def test_retry_feedback_error_returns_503_with_saved_attempt(app_client):
    client, fake_service, sessions = app_client
    _sign_in(client)
    fake_service.retry_result = RecordingOutcome(
        status="feedback_error", attempt=_attempt_out(feedback_status="error", error_code="feedback_service_unavailable")
    )
    resp = client.post("/api/interview/attempts/7/feedback")
    assert resp.status_code == 503
    assert resp.json()["error"] == "feedback_service_unavailable"
    assert sessions[-1].committed is True


# -- error contract mapping + rollback behaviour ---------------------------------------

_ERROR_CASES = [
    ("journey_prerequisite_incomplete", 409),
    ("invalid_setup", 422),
    ("interview_resource_not_found", 404),
    ("unsupported_audio_type", 415),
    ("recording_too_large", 413),
    ("invalid_audio", 422),
    ("recording_too_long", 422),
    ("no_speech_detected", 422),
    ("transcription_unavailable", 503),
]


@pytest.mark.parametrize("code,status", _ERROR_CASES)
def test_create_session_error_contract_and_no_commit(app_client, code, status):
    client, fake_service, sessions = app_client
    _sign_in(client)
    fake_service.create_error = InterviewError(code)
    resp = client.post("/api/interview/sessions", json={"role_id": "R1", "practice_focus": "general"})
    assert resp.status_code == status
    assert resp.json() == {"error": code}
    assert sessions[-1].committed is False


def test_unmapped_error_code_falls_back_to_400_without_leaking_internals(app_client):
    client, fake_service, _ = app_client
    _sign_in(client)
    fake_service.get_error = InterviewError("something_unexpected")
    resp = client.get("/api/interview/sessions/s1")
    assert resp.status_code == 400
    assert resp.json() == {"error": "something_unexpected"}
    assert "Traceback" not in resp.text


def test_refresh_session_not_found_does_not_commit(app_client):
    client, fake_service, sessions = app_client
    _sign_in(client)
    fake_service.refresh_error = InterviewError("interview_resource_not_found")
    resp = client.post("/api/interview/sessions/ghost/refresh")
    assert resp.status_code == 404
    assert sessions[-1].committed is False


def test_delete_session_not_found(app_client):
    client, fake_service, sessions = app_client
    _sign_in(client)
    fake_service.delete_error = InterviewError("interview_resource_not_found")
    resp = client.delete("/api/interview/sessions/ghost")
    assert resp.status_code == 404
    assert sessions[-1].committed is False


def test_record_attempt_transcription_error_does_not_commit(app_client):
    client, fake_service, sessions = app_client
    _sign_in(client)
    fake_service.process_error = InterviewError("no_speech_detected")
    resp = client.post(
        "/api/interview/sessions/s1/questions/1/attempts",
        files={"file": ("a.wav", b"fake-audio", "audio/wav")},
    )
    assert resp.status_code == 422
    assert resp.json() == {"error": "no_speech_detected"}
    assert sessions[-1].committed is False


# -- multipart size enforcement ---------------------------------------------------------

class _TinyLimitSettings:
    interview_max_audio_bytes = 10


def test_oversized_upload_rejected_before_reaching_the_service(app_client, monkeypatch):
    client, fake_service, sessions = app_client
    _sign_in(client)
    monkeypatch.setattr(interview_api, "get_settings", lambda: _TinyLimitSettings())

    resp = client.post(
        "/api/interview/sessions/s1/questions/1/attempts",
        files={"file": ("a.wav", b"x" * 11, "audio/wav")},
    )

    assert resp.status_code == 413
    assert resp.json() == {"error": "recording_too_large"}
    assert fake_service.calls == []  # rejected before the service is ever called
    assert sessions[-1].committed is False


def test_upload_within_limit_reaches_the_service(app_client, monkeypatch):
    client, fake_service, _ = app_client
    _sign_in(client)
    monkeypatch.setattr(interview_api, "get_settings", lambda: _TinyLimitSettings())

    resp = client.post(
        "/api/interview/sessions/s1/questions/1/attempts",
        files={"file": ("a.wav", b"x" * 10, "audio/wav")},
    )

    assert resp.status_code == 200
    assert fake_service.calls[0][0] == "process_recording"
