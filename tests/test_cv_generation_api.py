from datetime import datetime, timezone

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from app.db import get_session
from app.main import create_app
from app.services.cv_generation import (
    CvGenerationError,
    CvGenerateResult,
    CvImproveResult,
)


class FakeSession:
    def __init__(self):
        self.commit_count = 0

    async def commit(self):
        self.commit_count += 1


class FakeCvGenerationService:
    def __init__(self):
        self.generate_calls = []
        self.improve_calls = []
        self.generate_error = None
        self.improve_error = None
        self.generate_result = CvGenerateResult(
            role_id="role-project",
            status="generated",
            generated_at=datetime(2026, 10, 4, tzinfo=timezone.utc),
            draft={
                "version": 3,
                "roleId": "role-project",
                "personal": {"name": "", "email": "", "phone": "", "location": ""},
                "summary": "Operations professional.",
                "skills": ["Project coordination"],
                "experiences": [
                    {
                        "title": "Operations Coordinator",
                        "organisation": "Acme Sdn Bhd",
                        "start": "2018",
                        "end": "2021",
                        "description": "- Coordinated delivery.",
                    }
                ],
                "careerBreak": None,
            },
        )
        self.improve_result = CvImproveResult(
            section="experience",
            experience_index=0,
            suggestion="Coordinated cross-team delivery.",
            evidence="Coordinated delivery.",
        )

    async def generate(self, session, username, *, role_id=None, regenerate=False):
        self.generate_calls.append((session, username, role_id, regenerate))
        if self.generate_error:
            raise self.generate_error
        return self.generate_result

    async def improve(
        self,
        session,
        username,
        *,
        role_id,
        section,
        experience_index=None,
        current_text="",
        previous_suggestions=None,
    ):
        self.improve_calls.append(
            (
                session,
                username,
                role_id,
                section,
                experience_index,
                current_text,
                previous_suggestions,
            )
        )
        if self.improve_error:
            raise self.improve_error
        return self.improve_result


@pytest.fixture
def app_client():
    app = create_app()
    service = FakeCvGenerationService()
    app.state.cv_generation_service = service
    sessions = []

    async def _fake_session():
        session = FakeSession()
        sessions.append(session)
        yield session

    app.dependency_overrides[get_session] = _fake_session

    @app.post("/_test/sign-in")
    async def _sign_in(request: Request):
        request.session["username"] = (await request.json())["username"]
        return {"ok": True}

    return TestClient(app), service, sessions


def _sign_in(client, username="aisha"):
    client.post("/_test/sign-in", json={"username": username})


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/api/cv/generate", {}),
        (
            "/api/cv/improve",
            {"roleId": "role-project", "section": "summary"},
        ),
    ],
)
def test_cv_generation_endpoints_require_authentication(app_client, path, body):
    client, service, _ = app_client

    response = client.post(path, json=body)

    assert response.status_code == 401
    assert response.json() == {"error": "authentication_required"}
    assert service.generate_calls == []
    assert service.improve_calls == []


def test_generate_uses_session_username_returns_draft_and_commits(app_client):
    client, service, sessions = app_client
    _sign_in(client)

    response = client.post(
        "/api/cv/generate",
        json={"roleId": "role-project", "regenerate": True, "username": "attacker"},
    )

    assert response.status_code == 200
    assert response.json()["role_id"] == "role-project"
    assert response.json()["draft"]["roleId"] == "role-project"
    assert service.generate_calls[0][1:] == ("aisha", "role-project", True)
    assert sessions[-1].commit_count == 1


def test_improve_returns_suggestion_without_committing(app_client):
    client, service, sessions = app_client
    _sign_in(client)

    response = client.post(
        "/api/cv/improve",
        json={
            "roleId": "role-project",
            "section": "experience",
            "experienceIndex": 0,
            "currentText": "- Coordinated delivery.",
            "previousSuggestions": ["Earlier wording."],
        },
    )

    assert response.status_code == 200
    assert response.json()["suggestion"] == "Coordinated cross-team delivery."
    assert service.improve_calls[0][1] == "aisha"
    assert service.improve_calls[0][6] == ["Earlier wording."]
    assert sessions[-1].commit_count == 0


@pytest.mark.parametrize(
    ("code", "status"),
    [
        ("journey_prerequisite_incomplete", 409),
        ("invalid_cv_setup", 422),
        ("invalid_cv_content", 422),
        ("cv_generation_unavailable", 503),
    ],
)
def test_generate_maps_service_errors_to_stable_codes(app_client, code, status):
    client, service, _ = app_client
    service.generate_error = CvGenerationError(code)
    _sign_in(client)

    response = client.post("/api/cv/generate", json={})

    assert response.status_code == status
    assert response.json() == {"error": code}
