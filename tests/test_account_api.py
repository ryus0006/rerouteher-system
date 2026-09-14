import pytest
from fastapi.testclient import TestClient

from app.db import get_session
from app.main import create_app
from app.services.account import AccountError


class FakeSession:
    async def commit(self):
        pass


class FakeAccountService:
    def __init__(self):
        self.accounts = {}  # username -> {display_name, plan}

    async def create(self, session, *, username, password, display_name, plan):
        u = username.strip().lower()
        if u in self.accounts:
            raise AccountError("That username is taken.", "conflict")
        self.accounts[u] = {"display_name": display_name or u, "plan": plan}
        return {"username": u, "display_name": display_name or u}

    async def sign_in(self, session, *, username, password):
        u = username.strip().lower()
        acct = self.accounts.get(u)
        if acct is None:
            raise AccountError("Wrong username or password.", "auth")
        return {"username": u, "display_name": acct["display_name"], "plan": acct["plan"]}

    async def save_plan(self, session, *, username, plan):
        self.accounts.setdefault(username, {"display_name": username, "plan": {}})["plan"] = plan


@pytest.fixture
def client():
    app = create_app()
    app.state.account_service = FakeAccountService()

    async def _fake_session():
        yield FakeSession()

    app.dependency_overrides[get_session] = _fake_session
    return TestClient(app)


def test_create_sets_cookie_and_returns_identity(client):
    r = client.post("/api/account/create", json={"username": "Aisha", "password": "password1", "plan": {}})
    assert r.status_code == 200
    assert r.json() == {"username": "aisha", "display_name": "aisha"}
    assert any("session" in c for c in client.cookies.keys())


def test_duplicate_username_returns_409_error_shape(client):
    client.post("/api/account/create", json={"username": "aisha", "password": "password1"})
    r = client.post("/api/account/create", json={"username": "aisha", "password": "password1"})
    assert r.status_code == 409
    assert r.json() == {"error": "That username is taken."}


def test_save_plan_requires_session(client):
    r = client.post("/api/account/plan", json={"plan": {"a": 1}})
    assert r.status_code == 401
    assert r.json() == {"error": "Not signed in."}


def test_plan_works_after_sign_in(client):
    client.post("/api/account/create", json={"username": "aisha", "password": "password1"})
    r = client.post("/api/account/plan", json={"plan": {"a": 1}})
    assert r.status_code == 200
    assert r.json() == {"status": "saved"}


def test_bad_credentials_generic_401(client):
    r = client.post("/api/account/sign-in", json={"username": "ghost", "password": "password1"})
    assert r.status_code == 401
    assert r.json() == {"error": "Wrong username or password."}


def test_sign_out_clears_session(client):
    client.post("/api/account/create", json={"username": "aisha", "password": "password1"})
    assert client.post("/api/account/plan", json={"plan": {}}).status_code == 200
    assert client.post("/api/account/sign-out").json() == {"status": "signed_out"}
    assert client.post("/api/account/plan", json={"plan": {}}).status_code == 401
