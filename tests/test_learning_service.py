import pytest

from app.repositories.learning import CuratedResource, SkillLabel
from app.schemas.learning import LearningRequest
from app.services.learning import LearningService

pytestmark = pytest.mark.asyncio


class FakeRepo:
    def __init__(self, labels, curated):
        self._labels = labels
        self._curated = curated
        self.label_ids = None
        self.curated_ids = None

    async def get_skill_labels(self, session, skill_ids):
        self.label_ids = skill_ids
        return {k: v for k, v in self._labels.items() if k in skill_ids}

    async def get_curated_resources(self, session, skill_ids):
        self.curated_ids = skill_ids
        return [c for c in self._curated if c.skill_id in skill_ids]


def _labels():
    return {
        "s1": SkillLabel("s1", "User research", "Talking to users to learn what they need."),
        "s2": SkillLabel("s2", "Prototyping", None),
    }


async def test_curated_resource_is_mapped_with_evidence_as_why():
    curated = [
        CuratedResource(
            skill_id="s1",
            resource_id="r1",
            title="Intro to UX research",
            url="https://open.edu/x",
            delivery_mode="Course",
            duration_minutes=90,
            licence_note="CC-BY",
            evidence_note="Covers the interview basics you are missing.",
            provider_name="OpenLearn",
        )
    ]
    svc = LearningService(repo=FakeRepo(_labels(), curated))
    resp = await svc.recommend(LearningRequest(skill_ids=["s1"], target_role_id="R1"), session=object())
    assert [g.skill_id for g in resp.groups] == ["s1"]
    assert resp.groups[0].skill == "User research"
    assert resp.groups[0].blurb  # from definition
    res = resp.resources[0]
    assert res.skill_id == "s1"
    assert res.provider == "OpenLearn"
    assert res.minutes == 90
    assert res.why == "Covers the interview basics you are missing."
    assert res.free is True and res.cost == "Free"


async def test_uncovered_skill_gets_one_youtube_search_resource():
    svc = LearningService(repo=FakeRepo(_labels(), curated=[]))
    resp = await svc.recommend(LearningRequest(skill_ids=["s2"], target_role_id="R1"), session=object())
    assert len(resp.resources) == 1
    res = resp.resources[0]
    assert res.skill_id == "s2"
    assert res.provider == "YouTube"
    assert res.logo == "youtube"
    assert res.minutes is None  # AC 6.2.4
    assert res.free is True and res.cost == "Free"  # AC 6.2.5
    assert "youtube.com/results?search_query=" in res.url
    assert "Prototyping" in res.url or "prototyping" in res.url.lower()
    assert res.why  # relevance explanation present (AC 6.2.2)


async def test_unresolved_skill_id_is_skipped():
    svc = LearningService(repo=FakeRepo(_labels(), curated=[]))
    resp = await svc.recommend(
        LearningRequest(skill_ids=["s1", "ghost"], target_role_id="R1"), session=object()
    )
    assert [g.skill_id for g in resp.groups] == ["s1"]  # ghost dropped, no broken entry
    assert all(r.skill_id == "s1" for r in resp.resources)


async def test_no_pii_only_skill_ids_passed_to_repo():
    repo = FakeRepo(_labels(), curated=[])
    svc = LearningService(repo=repo)
    await svc.recommend(
        LearningRequest(skill_ids=["s1"], target_role_id="R1", target_role="UX Designer"),
        session=object(),
    )
    assert repo.label_ids == ["s1"] and repo.curated_ids == ["s1"]


async def test_empty_skill_ids_returns_empty_plan():
    svc = LearningService(repo=FakeRepo(_labels(), curated=[]))
    resp = await svc.recommend(LearningRequest(skill_ids=[], target_role_id="R1"), session=object())
    assert resp.groups == [] and resp.resources == []
