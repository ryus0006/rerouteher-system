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
    have = {"user research", "prototyping", "ai design tools"}
    readiness = svc._readiness(
        have,
        [s for s in role.skills if s.skill_type in ("technical", "soft")],
        [s for s in role.skills if s.skill_type == "digital"],
        0.4,
    )
    assert round(readiness, 1) == 100.0


def test_uplift_is_positive_for_missing_skill():
    svc = _service()
    role = _role()
    role_band = [s for s in role.skills if s.skill_type in ("technical", "soft")]
    ai_band = [s for s in role.skills if s.skill_type == "digital"]
    have: set[str] = set()
    base = svc._readiness(have, role_band, ai_band, 0.4)
    up = svc._uplift(have, role_band, ai_band, 0.4, base, "AI design tools")
    assert up > 0
