import pytest
from fastapi.testclient import TestClient

from app.db import get_session
from app.main import create_app
from app.services.account import AccountError
from app.services.profile_skills import ProfileSkillError, ProfileSkillMutation


class FakeSession:
    def __init__(self):
        self.commits = 0

    async def commit(self):
        self.commits += 1


class FakeAccountService:
    async def create(self, session, *, username, password, display_name, plan):
        return {
            "username": username.strip().lower(),
            "display_name": display_name or username.strip().lower(),
        }

    async def sign_in(self, session, *, username, password):
        raise AccountError("Wrong username or password.", "auth")

    async def save_plan(self, session, *, username, plan):
        return None


class FakeProfileSkillService:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    async def add_skill(self, session, username, skill_id):
        self.calls.append(("add", username, skill_id))
        if self.error:
            raise self.error
        return _mutation("added", skill_id)

    async def remove_skill(self, session, username, skill_id):
        self.calls.append(("remove", username, skill_id))
        if self.error:
            raise self.error
        return _mutation("removed", skill_id)


def _mutation(status, skill_id):
    snapshot = {"professional_skills": [{"skill_id": skill_id, "skill": "SQL"}]}
    return ProfileSkillMutation(
        status=status,
        skill_id=skill_id,
        skill="SQL",
        definition="Working with structured data.",
        plan={"snapshot": snapshot},
        snapshot=snapshot,
        gap_result=None,
        learned_skills=[],
    )


@pytest.fixture
def client():
    app = create_app()
    app.state.account_service = FakeAccountService()
    profile = FakeProfileSkillService()
    app.state.profile_skill_service = profile
    session = FakeSession()

    async def _fake_session():
        yield session

    app.dependency_overrides[get_session] = _fake_session
    test_client = TestClient(app)
    test_client.profile = profile
    test_client.db_session = session
    return test_client


def sign_in(client):
    client.post(
        "/api/account/create",
        json={"username": "Aisha", "password": "password1", "plan": {}},
    )


def test_skill_mutations_require_signed_in_session(client):
    assert client.put("/api/account/professional-skills/s1").status_code == 401
    assert client.delete("/api/account/professional-skills/s1").status_code == 401
    assert client.profile.calls == []


def test_add_and_remove_skill_use_session_identity_and_commit(client):
    sign_in(client)

    added = client.put("/api/account/professional-skills/s1")
    removed = client.delete("/api/account/professional-skills/s1")

    assert added.status_code == 200
    assert added.json()["status"] == "added"
    assert removed.status_code == 200
    assert removed.json()["status"] == "removed"
    assert client.profile.calls == [
        ("add", "aisha", "s1"),
        ("remove", "aisha", "s1"),
    ]
    assert client.db_session.commits == 3


@pytest.mark.parametrize(
    ("kind", "status", "message"),
    [
        ("not_found", 404, "That skill was not found."),
        ("journey", 409, "Complete your skill snapshot first."),
        ("internal", 500, "Profile skill is temporarily unavailable."),
    ],
)
def test_profile_skill_errors_map_to_stable_statuses(client, kind, status, message):
    sign_in(client)
    client.profile.error = ProfileSkillError(message, kind)
    before = client.db_session.commits

    response = client.put("/api/account/professional-skills/s1")

    assert response.status_code == status
    assert response.json() == {"error": message}
    assert client.db_session.commits == before
