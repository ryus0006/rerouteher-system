"""Smoke test: the app builds and /api/health responds, plus the E7 interview
dependency-availability contract (ok vs degraded, and no secret/prompt leakage)."""
from fastapi.testclient import TestClient

from app.main import create_app


def test_health_smoke_with_real_lifespan():
    # build the app without pre-seeding app.state (lifespan runs for real here)
    app = create_app()
    with TestClient(app, raise_server_exceptions=True) as client:
        # lifespan runs here; if models/DB are absent it logs warnings but still boots
        resp = client.get("/api/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("ok", "degraded")
        assert set(body["interview"].keys()) == {"transcription_available", "feedback_available"}


def test_health_reports_ok_when_interview_dependencies_available():
    app = create_app()  # lifespan not triggered without `with`, so state is set manually
    app.state.interview_health = {"transcription_available": True, "feedback_available": True}
    client = TestClient(app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {
        "status": "ok",
        "interview": {"transcription_available": True, "feedback_available": True},
    }


def test_health_reports_degraded_when_transcription_unavailable():
    app = create_app()
    app.state.interview_health = {"transcription_available": False, "feedback_available": True}
    client = TestClient(app)
    assert client.get("/api/health").json()["status"] == "degraded"


def test_health_reports_degraded_when_feedback_unavailable():
    app = create_app()
    app.state.interview_health = {"transcription_available": True, "feedback_available": False}
    client = TestClient(app)
    assert client.get("/api/health").json()["status"] == "degraded"


def test_health_never_exposes_secrets_prompts_or_user_data():
    app = create_app()
    app.state.interview_health = {"transcription_available": True, "feedback_available": True}
    client = TestClient(app)
    blob = str(client.get("/api/health").json()).lower()
    for forbidden in ("secret", "prompt", "password", "username"):
        assert forbidden not in blob
