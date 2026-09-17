from app.schemas.learning import (
    LearningGroupOut,
    LearningRequest,
    LearningResourceOut,
    LearningResponse,
)


def test_request_defaults_skill_ids_to_empty():
    req = LearningRequest(target_role_id="R1")
    assert req.skill_ids == []
    assert req.target_role is None


def test_response_round_trips_groups_and_resources():
    resp = LearningResponse(
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
                why="A starting point for building user research.",
            )
        ],
    )
    dumped = resp.model_dump()
    assert dumped["groups"][0]["icon"] is None
    assert dumped["resources"][0]["minutes"] is None
    assert dumped["resources"][0]["free"] is True
