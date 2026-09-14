from fastapi.testclient import TestClient

from app.api.learning import router as learning_router
from app.db import get_session
from app.schemas.learning import LearningGroupOut, LearningResourceOut, LearningResponse


class FakeLearningService:
    def __init__(self):
        self.seen = None

    async def recommend(self, req, session):
        self.seen = req
        return LearningResponse(
            groups=[LearningGroupOut(skill_id="s1", skill="User research")],
            resources=[
                LearningResourceOut(
                    id="search-s1",
                    skill_id="s1",
                    title="User research tutorials",
                    provider="YouTube",
                    logo="youtube",
                    format="Video",
                    minutes=None,
                    cost="Free",
                    free=True,
                    url="https://www.youtube.com/results?search_query=x",
                    why="A starting point.",
                )
            ],
        )


def _client(service):
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(learning_router)
    app.state.learning_service = service

    async def _fake_session():
        yield object()

    app.dependency_overrides[get_session] = _fake_session
    return TestClient(app)


def test_recommend_returns_plan_and_passes_skill_ids():
    service = FakeLearningService()
    client = _client(service)
    resp = client.post(
        "/api/learning/recommend",
        json={"skill_ids": ["s1"], "target_role_id": "R1", "target_role": "UX Designer"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["groups"][0]["skill_id"] == "s1"
    assert body["resources"][0]["provider"] == "YouTube"
    assert service.seen.skill_ids == ["s1"]


def test_recommend_requires_target_role_id():
    client = _client(FakeLearningService())
    resp = client.post("/api/learning/recommend", json={"skill_ids": ["s1"]})
    assert resp.status_code == 422
