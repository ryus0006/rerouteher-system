"""Two-band skill-gap engine.

Readiness % blended by the role's ai_exposure, per-gap uplift, and a merged top-3
focus list. Computed at request time, nothing stored.
"""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.repositories import roles as roles_repo
from app.schemas.gap import Gap, GapRequest, GapResponse


class GapService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def compute(self, req: GapRequest, session: AsyncSession) -> GapResponse:
        role = await roles_repo.get_role_with_skills(session, req.target_role)
        if role is None:
            # unknown target role: nothing to score against
            return GapResponse(readiness=0.0, skills_have=[], gaps=[])

        have = {s.skill.lower() for s in req.skills}
        role_band = [rs for rs in role.skills if rs.skill_type in ("technical", "soft")]
        ai_band = [rs for rs in role.skills if rs.skill_type == "digital"]
        exposure_w = self._settings.ai_exposure_weight(role.ai_exposure)

        readiness = self._readiness(have, role_band, ai_band, exposure_w)
        skills_have = sorted({rs.skill_name for rs in role.skills if rs.skill_name.lower() in have})
        gaps = self._rank_gaps(have, role_band, ai_band, exposure_w, readiness)
        return GapResponse(readiness=round(readiness, 1), skills_have=skills_have, gaps=gaps)

    def _coverage(self, have: set[str], band: list) -> float:
        total = sum(float(rs.importance) for rs in band)
        if total == 0:
            return 1.0
        covered = sum(float(rs.importance) for rs in band if rs.skill_name.lower() in have)
        return covered / total

    def _readiness(self, have, role_band, ai_band, exposure_w: float) -> float:
        role_cov = self._coverage(have, role_band)
        ai_cov = self._coverage(have, ai_band)
        return ((1 - exposure_w) * role_cov + exposure_w * ai_cov) * 100

    def _rank_gaps(self, have, role_band, ai_band, exposure_w: float, base: float) -> list[Gap]:
        gaps: list[Gap] = []
        for rs in role_band:
            if rs.skill_name.lower() in have:
                continue
            uplift = self._uplift(have, role_band, ai_band, exposure_w, base, rs.skill_name)
            gaps.append(Gap(skill=rs.skill_name, band="role", importance=float(rs.importance), uplift=uplift))
        for rs in ai_band:
            if rs.skill_name.lower() in have:
                continue
            uplift = self._uplift(have, role_band, ai_band, exposure_w, base, rs.skill_name)
            # TODO: apply the band-2 {low 0.8, medium 1.0, high 1.2} emphasis factor to
            # importance before ranking.
            gaps.append(Gap(skill=rs.skill_name, band="ai_digital", importance=float(rs.importance), uplift=uplift))
        # merge both bands, rank by uplift desc, tie-break importance; top-3 are the focus areas
        gaps.sort(key=lambda g: (g.uplift, g.importance), reverse=True)
        return gaps[:3]

    def _uplift(self, have, role_band, ai_band, exposure_w, base: float, skill_name: str) -> float:
        # marginal readiness gain if this skill were covered
        boosted = have | {skill_name.lower()}
        return round(self._readiness(boosted, role_band, ai_band, exposure_w) - base, 1)
