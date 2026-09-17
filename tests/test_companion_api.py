from fastapi.testclient import TestClient

from app.api.companion import router as companion_router
from app.db import get_session
from app.schemas.companion import AskResponse


class FakeService:
    def __init__(self):
        self.seen = None

    async def ask(self, req, session, username):
        self.seen = {"req": req, "username": username}
        return AskResponse(answer="hello", sources=[], journey_update=None)


class FakeSession:
    def __init__(self):
        self.committed = False

    async def commit(self):
        self.committed = True


def _client(service, session):
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(companion_router)
    app.state.companion_service = service

    async def _fake_session():
        yield session

    app.dependency_overrides[get_session] = _fake_session
    return TestClient(app)


def test_ask_returns_answer_guest_no_username():
    service, session = FakeService(), FakeSession()
    client = _client(service, session)
    resp = client.post("/api/companion/ask", json={"question": "hi", "session_id": "s1"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "hello"
    assert service.seen["username"] is None
    assert session.committed is True


def test_ask_requires_question_and_session_id():
    client = _client(FakeService(), FakeSession())
    assert client.post("/api/companion/ask", json={"question": "hi"}).status_code == 422
