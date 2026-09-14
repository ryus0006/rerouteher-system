from fastapi.testclient import TestClient

from app.api.employers import router as employers_router
from app.db import get_session
from app.schemas.employers import EmployerMatchOut, EmployerMatchResponse


class FakeEmployerService:
    def __init__(self):
        self.seen = None

    async def match(self, req, session):
        self.seen = req
        return EmployerMatchResponse(
            employers=[
                EmployerMatchOut(
                    id="nestle",
                    name="Nestle Malaysia",
                    discloses=["flexible_work"],
                    met=["flexible_work"],
                    unmet=[],
                )
            ]
        )


def _client(service):
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(employers_router)
    app.state.employer_service = service

    async def _fake_session():
        yield object()

    app.dependency_overrides[get_session] = _fake_session
    return TestClient(app)


def test_match_returns_employers_and_passes_priorities():
    service = FakeEmployerService()
    client = _client(service)
    resp = client.post(
        "/api/employers/match",
        json={"priorities": ["flexible_work"], "target_role_id": "role_dev"},
    )
    assert resp.status_code == 200
    assert resp.json()["employers"][0]["id"] == "nestle"
    assert service.seen.priorities == ["flexible_work"]


def test_match_works_without_target_role_id():
    client = _client(FakeEmployerService())
    resp = client.post("/api/employers/match", json={"priorities": ["flexible_work"]})
    assert resp.status_code == 200
