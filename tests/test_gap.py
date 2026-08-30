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
            RoleSkillRow("s3", "AI design tools", "digital", 90),
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
    up = svc._uplift(role.skills, cov, 0.4, base, "s3")
    assert up > 0
