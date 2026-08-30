"""Unit tests for the deterministic readiness/gap math (no DB, no models)."""
from app.config import Settings
from app.repositories.roles import RoleSkillRow, RoleWithSkills
from app.services.gap import GapService


def _service() -> GapService:
    return GapService(Settings(ai_exposure_medium=0.4))


def _role() -> RoleWithSkills:
    return RoleWithSkills(
        role_id="R1",
        role_title="UX/UI Designer",
        ai_exposure="medium",
        skills=[
            RoleSkillRow("s1", "User research", "technical", 80),
            RoleSkillRow("s2", "Prototyping", "technical", 70),
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
