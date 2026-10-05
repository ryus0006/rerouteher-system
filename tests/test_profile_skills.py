import copy

import pytest

from app.repositories.roles import RoleSkillRow, RoleWithSkills
from app.repositories.skills import SkillMatch, SkillRow
from app.schemas.gap import Gap, GapResponse
from app.services import profile_skills as profile_skills_mod
from app.services.profile_skills import ProfileSkillError, ProfileSkillService

pytestmark = pytest.mark.asyncio


class FakeAccounts:
    def __init__(self, plan, fail=False):
        self.plan = plan
        self.fail = fail
        self.saved = []

    async def get_plan(self, session, username):
        return copy.deepcopy(self.plan)

    async def upsert_plan(self, session, username, plan):
        if self.fail:
            raise RuntimeError("database unavailable")
        self.plan = copy.deepcopy(plan)
        self.saved.append((username, copy.deepcopy(plan)))


class FakeSkills:
    def __init__(self):
        self.by_id = {
            "s-sql": SkillRow("s-sql", "SQL", "technical", "Working with structured data."),
            "s-ux": SkillRow("s-ux", "User research", "technical", "Understanding users."),
            "s-copy": SkillRow("s-copy", "Copywriting", "technical", "Writing clear copy."),
        }
        self.exact = []
        self.embedding = []

    async def get_skill_by_id(self, session, skill_id):
        return self.by_id.get(skill_id)

    async def find_exact_skills(self, session, query, limit):
        return self.exact[:limit]

    async def match_by_embedding(self, session, query_vec, k, threshold):
        return self.embedding[:k]


class FakeEmbedder:
    def encode_one(self, query):
        return [0.1, 0.2]


class FakeGapService:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def compute(self, request, session):
        self.calls.append(request)
        return self.response


def make_plan():
    return {
        "snapshot": {
            "professional_skills": [{"skill_id": "s-ux", "skill": "User research"}],
            "reframed_skills": [],
        },
        "selectedRole": {"role_id": "role-1", "role": "Analyst"},
        "gapResult": {
            "readiness": 55,
            "skills_have": ["User research"],
            "gaps": [
                {
                    "skill_id": "s-sql",
                    "skill": "SQL",
                    "band": "role",
                    "importance": 0.8,
                    "uplift": 12,
                    "definition": "Working with structured data.",
                },
                {
                    "skill_id": "s-copy",
                    "skill": "Copywriting",
                    "band": "role",
                    "importance": 0.4,
                    "uplift": 5,
                    "definition": "Writing clear copy.",
                },
            ],
        },
        "learningProgress": {},
        "learnedSkills": [],
        "unrelated": {"keep": True},
    }


@pytest.fixture
def fakes(monkeypatch):
    accounts = FakeAccounts(make_plan())
    skills = FakeSkills()
    monkeypatch.setattr(profile_skills_mod, "accounts_repo", accounts)
    monkeypatch.setattr(profile_skills_mod, "skills_repo", skills)
    class FakeRoles:
        @staticmethod
        async def get_role_with_skills_by_id(session, role_id):
            return _role_with("s-sql")

    monkeypatch.setattr(profile_skills_mod, "roles_repo", FakeRoles)
    return accounts, skills


def _role_with(*skill_ids):
    return RoleWithSkills(
        "role-1",
        "Analyst",
        "low",
        [
            RoleSkillRow(skill_id, skill_id, "technical", 0.8, None)
            for skill_id in skill_ids
        ],
    )


async def test_match_prefers_exact_alias_and_limits_embedding_candidates(fakes):
    _accounts, skills = fakes
    skills.exact = [
        SkillRow("s-sql", "SQL", "technical", "Working with structured data.")
    ]
    skills.embedding = [
        SkillMatch("s-ux", "User research", 0.88, "Understanding users."),
        SkillMatch("s-copy", "Copywriting", 0.84, "Writing clear copy."),
        SkillMatch("extra", "Extra", 0.81, "Extra."),
    ]

    service = ProfileSkillService(embedder=FakeEmbedder())
    matches = await service.match_skills(object(), "database querying", limit=3)

    assert [match.skill_id for match in matches] == ["s-sql"]


async def test_match_uses_embedding_fallback_and_caps_results(fakes):
    _accounts, skills = fakes
    skills.embedding = [
        SkillMatch("s-ux", "User research", 0.88, "Understanding users."),
        SkillMatch("s-copy", "Copywriting", 0.84, "Writing clear copy."),
        SkillMatch("extra", "Extra", 0.81, "Extra."),
        SkillMatch("fourth", "Fourth", 0.79, "Fourth."),
    ]

    service = ProfileSkillService(embedder=FakeEmbedder())
    matches = await service.match_skills(object(), "writing for customers", limit=3)

    assert [match.skill_id for match in matches] == ["s-ux", "s-copy", "extra"]


async def test_add_current_gap_skill_applies_uplift_and_leaves_learned_skills(fakes):
    accounts, _skills = fakes
    service = ProfileSkillService()

    result = await service.add_skill(object(), "aisha", "s-sql")

    assert result.status == "added"
    assert result.skill_id == "s-sql"
    assert result.snapshot["professional_skills"][-1]["skill"] == "SQL"
    assert result.gap_result["readiness"] == 67
    assert result.gap_result["skills_have"][-1] == "SQL"
    assert [gap["skill_id"] for gap in result.gap_result["gaps"]] == ["s-copy"]
    assert result.learned_skills == []
    assert result.plan["unrelated"] == {"keep": True}
    assert len(accounts.saved) == 1


async def test_add_is_idempotent_and_does_not_apply_uplift_twice(fakes):
    accounts, _skills = fakes
    service = ProfileSkillService()

    first = await service.add_skill(object(), "aisha", "s-sql")
    second = await service.add_skill(object(), "aisha", "s-sql")

    assert first.status == "added"
    assert second.status == "already_present"
    assert second.gap_result["readiness"] == 67
    assert len(second.snapshot["professional_skills"]) == 2
    assert len(accounts.saved) == 1


async def test_remove_required_skill_recomputes_readiness(fakes):
    accounts, _skills = fakes
    recomputed = GapResponse(
        readiness=43,
        skills_have=[],
        gaps=[Gap(skill_id="s-sql", skill="SQL", band="role", importance=0.8, uplift=12)],
    )
    gap_service = FakeGapService(recomputed)
    service = ProfileSkillService(gap_service=gap_service)
    await service.add_skill(object(), "aisha", "s-sql")

    result = await service.remove_skill(object(), "aisha", "s-sql")

    assert result.status == "removed"
    assert result.snapshot["professional_skills"] == [{"skill_id": "s-ux", "skill": "User research"}]
    assert result.gap_result["readiness"] == 43
    assert gap_service.calls
    assert len(accounts.saved) == 2


async def test_remove_unrelated_skill_keeps_current_gap_result(fakes):
    accounts, _skills = fakes
    service = ProfileSkillService()
    await service.add_skill(object(), "aisha", "s-ux")

    result = await service.remove_skill(object(), "aisha", "s-ux")

    assert result.status == "removed"
    assert result.gap_result["readiness"] == 55
    assert result.gap_result["gaps"][0]["skill_id"] == "s-sql"
    assert len(accounts.saved) == 1


async def test_remove_absent_skill_is_idempotent(fakes):
    accounts, _skills = fakes
    service = ProfileSkillService()

    result = await service.remove_skill(object(), "aisha", "s-copy")

    assert result.status == "not_present"
    assert result.snapshot["professional_skills"] == [
        {"skill_id": "s-ux", "skill": "User research"}
    ]
    assert len(accounts.saved) == 0


async def test_mutation_rejects_missing_snapshot(fakes):
    accounts, _skills = fakes
    accounts.plan = {"learningProgress": {}}
    service = ProfileSkillService()

    with pytest.raises(ProfileSkillError, match="skill snapshot"):
        await service.add_skill(object(), "aisha", "s-sql")


async def test_mutation_rejects_unknown_taxonomy_skill(fakes):
    _accounts, _skills = fakes
    service = ProfileSkillService()

    with pytest.raises(ProfileSkillError, match="not found"):
        await service.add_skill(object(), "aisha", "missing")


async def test_mutation_maps_persistence_failure_to_internal_error(fakes):
    accounts, _skills = fakes
    accounts.fail = True
    service = ProfileSkillService()

    with pytest.raises(ProfileSkillError, match="temporarily unavailable"):
        await service.add_skill(object(), "aisha", "s-sql")
