"""Unit tests for the deterministic readiness/gap math (no DB, no models)."""
from app.config import Settings
from app.repositories.roles import RoleSkillRow, RoleWithSkills
from app.schemas.gap import GapRequest
from app.services import gap as gap_module
from app.services.gap import GapService


def _service() -> GapService:
    return GapService(Settings(ai_exposure_medium=0.4))


def _role() -> RoleWithSkills:
    return RoleWithSkills(
        role_id="R1",
        role_title="UX/UI Designer",
        ai_exposure="medium",
        skills=[
            RoleSkillRow(
                "s1",
                "User research",
                "technical",
                80,
                "Understanding users and their needs.",
            ),
            RoleSkillRow("s2", "Prototyping", "technical", 70, " "),
            RoleSkillRow("s3", "Coordination", "soft", 60),
            RoleSkillRow("s4", "Use AI design tools", "ai_usage", 90),
        ],
    )


def test_full_coverage_is_100():
    svc = _service()
    role = _role()
    cov = {s.skill_id: 1.0 for s in role.skills}
    assert round(svc._readiness(role.skills, cov, 0.4), 1) == 100.0


def test_zero_coverage_is_0():
    svc = _service()
    role = _role()
    cov = {s.skill_id: 0.0 for s in role.skills}
    assert round(svc._readiness(role.skills, cov, 0.4), 1) == 0.0


def test_partial_coverage_counts():
    svc = _service()
    role = _role()
    # partial credit on a role-band skill lifts readiness above zero
    cov = {"s1": 0.5, "s2": 0.0, "s3": 0.0}
    r = svc._readiness(role.skills, cov, 0.4)
    assert 0.0 < r < 100.0


def test_uplift_is_positive_for_missing_skill():
    svc = _service()
    role = _role()
    cov = {s.skill_id: 0.0 for s in role.skills}
    base = svc._readiness(role.skills, cov, 0.4)
    up = svc._uplift(role.skills, cov, 0.4, base, "s4")
    assert up > 0


def test_soft_skills_are_not_listed_as_gaps():
    svc = _service()
    role = _role()
    cov = {s.skill_id: 0.0 for s in role.skills}
    base = svc._readiness(role.skills, cov, 0.4)
    gaps = {g.skill for g in svc._rank_gaps(role.skills, cov, 0.4, base)}
    assert "Coordination" not in gaps  # soft counts toward readiness but is never a gap
    assert "User research" in gaps


def test_ai_usage_skill_is_labelled_ai_usage():
    svc = _service()
    role = _role()
    cov = {s.skill_id: 0.0 for s in role.skills}
    base = svc._readiness(role.skills, cov, 0.4)
    bands = {g.skill: g.band for g in svc._rank_gaps(role.skills, cov, 0.4, base)}
    assert bands["Use AI design tools"] == "ai_usage"
    assert bands["User research"] == "role"


def test_ranked_gaps_carry_skill_id():
    svc = _service()
    role = _role()
    # s1 (User research) uncovered -> must appear as a gap carrying its skill_id
    cov = {
        s.skill_id: (0.0 if s.skill_id in {"s1", "s2"} else 1.0)
        for s in role.skills
    }
    base = svc._readiness(role.skills, cov, 0.4)
    gaps = svc._rank_gaps(role.skills, cov, 0.4, base)
    by_id = {g.skill_id: g for g in gaps}
    assert "s1" in by_id
    assert by_id["s1"].skill == "User research"
    assert by_id["s1"].definition == "Understanding users and their needs."
    assert by_id["s2"].definition is None


async def test_held_skills_carry_definition(monkeypatch):
    # A met soft skill (Coordination) must carry its ESCO definition so the UI can
    # show it on hover without a name match against the user's own skills.
    role = RoleWithSkills(
        role_id="R1",
        role_title="UX/UI Designer",
        ai_exposure="medium",
        skills=[
            RoleSkillRow("s1", "User research", "technical", 80, "Understanding users."),
            RoleSkillRow("s3", "Coordination", "soft", 60, "Adjusting actions to others."),
        ],
    )

    async def fake_get_role(session, role_id):
        return role

    async def fake_sims(session, role_id, have_ids):
        return {}

    monkeypatch.setattr(gap_module.roles_repo, "get_role_with_skills_by_id", fake_get_role)
    monkeypatch.setattr(gap_module.roles_repo, "best_similarity_for_role", fake_sims)

    # The user holds both skills exactly, so both land in skills_have.
    req = GapRequest(skill_ids=["s1", "s3"], target_role_id="R1")
    resp = await _service().compute(req, session=None)

    held = {h.skill_id: h for h in resp.skills_have}
    assert held["s3"].skill == "Coordination"
    assert held["s3"].definition == "Adjusting actions to others."
    assert held["s1"].definition == "Understanding users."
