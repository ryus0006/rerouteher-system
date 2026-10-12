from fastapi.testclient import TestClient

from app.api.employers import router as employers_router
from app.db import get_session
from app.schemas.employers import (
    EmployerMatchOut,
    EmployerMatchResponse,
    JobRefreshResponse,
    JobSearchOut,
)


class FakeEmployerService:
    def __init__(self, refresh_result=None):
        self.seen = None
        self.refreshed = None
        self.refresh_result = refresh_result

    async def match(self, req, session):
        self.seen = req
        return EmployerMatchResponse(
            job_search=JobSearchOut(status="empty", searched_at=None),
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

    async def refresh(self, role_id, session):
        self.refreshed = role_id
        return self.refresh_result or JobRefreshResponse(
            role_id=role_id, status="ready", opening_count=1,
            searched_at="2026-10-04T00:00:00Z",
        )


def _client(service):
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(employers_router)
    app.state.employer_service = service

    class FakeSession:
        async def commit(self):
            return None

    async def _fake_session():
        yield FakeSession()

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
    assert resp.status_code == 422


def test_refresh_returns_job_search_state():
    service = FakeEmployerService()
    client = _client(service)
    resp = client.post("/api/employers/jobs/refresh/role_dev")
    assert resp.status_code == 200
    assert resp.json()["opening_count"] == 1
    assert service.refreshed == "role_dev"


def test_refresh_returns_top_level_temporary_unavailable_status():
    service = FakeEmployerService(
        refresh_result=JobRefreshResponse(
            role_id="role_dev", status="temporarily_unavailable",
            opening_count=0, searched_at="2026-10-04T00:00:00Z",
        )
    )
    client = _client(service)
    resp = client.post("/api/employers/jobs/refresh/role_dev")
    assert resp.status_code == 503
    assert resp.json()["status"] == "temporarily_unavailable"
