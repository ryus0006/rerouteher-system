"""Two-band skill-gap engine (exact skill_id + bounded embedding match).

A role skill is covered when the user has the exact ESCO skill (same skill_id) or a skill
whose embedding is at least gap_cosine_threshold similar - cosine is computed only between
this role's skills and the user's skills, never the whole taxonomy, so a concrete CV skill
(Java (Computer Programming)) covers the role's parent skill (Computer Programming).
Readiness blends the two bands (role skills, AI/digital) by the role's AI-exposure, and
the focus list is the top-3 uncovered skills by readiness uplift. Nothing is stored.
"""
from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.repositories import roles as roles_repo
from app.schemas.gap import Gap, GapRequest, GapResponse

logger = logging.getLogger("rerouteher")


class GapService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def compute(self, req: GapRequest, session: AsyncSession) -> GapResponse:
        role = await roles_repo.get_role_with_skills_by_id(session, req.target_role_id)
        if role is None:
            # unknown target role: nothing to score against
            logger.info(
                "gap: target_id=%s (%r) resolved=None -> readiness=0.0",
                req.target_role_id, req.target_role,
            )
            return GapResponse(readiness=0.0, skills_have=[], gaps=[])

        have_ids = {sid for sid in req.skill_ids if sid}
        sims = await roles_repo.best_similarity_for_role(session, role.role_id, list(have_ids))
        threshold = self._settings.gap_cosine_threshold

        cov: dict[str, float] = {}
        exact = embed = 0
        for rs in role.skills:
            if rs.skill_id in have_ids:
                cov[rs.skill_id] = 1.0
                exact += 1
            elif sims.get(rs.skill_id, 0.0) >= threshold:
                cov[rs.skill_id] = 1.0
                embed += 1
            else:
                cov[rs.skill_id] = 0.0

        exposure_w = self._settings.ai_exposure_weight(role.ai_exposure)
        readiness = self._readiness(role.skills, cov, exposure_w)
        skills_have = sorted({rs.skill_name for rs in role.skills if cov[rs.skill_id] >= 1.0})
        gaps = self._rank_gaps(role.skills, cov, exposure_w, readiness)
        logger.info(
            "gap: target_id=%s (%r) -> role_id=%s role_skills=%d have_ids=%d exact=%d embed=%d readiness=%.1f",
            req.target_role_id, req.target_role, role.role_id, len(role.skills), len(have_ids),
            exact, embed, readiness,
        )
        logger.info(
            "gap sims: %s",
            [(rs.skill_name, round(sims.get(rs.skill_id, 0.0), 3)) for rs in role.skills],
        )
        return GapResponse(readiness=round(readiness, 1), skills_have=skills_have, gaps=gaps)

    # --------------------------------------------------------------- readiness
    def _readiness(self, role_skills, cov: dict[str, float], exposure_w: float) -> float:
        role_band = [rs for rs in role_skills if rs.skill_type in ("technical", "soft")]
        ai_band = [rs for rs in role_skills if rs.skill_type == "digital"]
        role_cov = self._band_coverage(role_band, cov)
        ai_cov = self._band_coverage(ai_band, cov)
        return ((1 - exposure_w) * role_cov + exposure_w * ai_cov) * 100

    @staticmethod
    def _band_coverage(band, cov: dict[str, float]) -> float:
        total = sum(float(rs.importance) for rs in band)
        if total == 0:
            return 1.0
        covered = sum(float(rs.importance) * cov.get(rs.skill_id, 0.0) for rs in band)
        return covered / total

    # -------------------------------------------------------------------- gaps
    def _rank_gaps(self, role_skills, cov: dict[str, float], exposure_w: float, base: float) -> list[Gap]:
        gaps: list[Gap] = []
        for rs in role_skills:
            if cov.get(rs.skill_id, 0.0) >= 1.0:
                continue
            uplift = self._uplift(role_skills, cov, exposure_w, base, rs.skill_id)
            band = "ai_digital" if rs.skill_type == "digital" else "role"
            gaps.append(
                Gap(skill=rs.skill_name, band=band, importance=float(rs.importance), uplift=uplift)
            )
        gaps.sort(key=lambda g: (g.uplift, g.importance), reverse=True)
        return gaps[:3]

    def _uplift(self, role_skills, cov: dict[str, float], exposure_w, base: float, skill_id: str) -> float:
        # marginal readiness gain if this skill were covered
        boosted = dict(cov)
        boosted[skill_id] = 1.0
        return round(self._readiness(role_skills, boosted, exposure_w) - base, 1)
