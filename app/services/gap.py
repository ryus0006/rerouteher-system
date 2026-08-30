"""Two-band skill-gap engine.

Each required role skill gets a coverage in 0..1: 1.0 when the user has the exact
skill (same ESCO skill_id), otherwise the best cosine similarity between the role
skill and the user's skills (career-break skills counted at a discount). Readiness
blends the two bands (role skills, AI/digital skills) by the role's AI-exposure, and
the focus list is the top-3 uncovered skills by readiness uplift. Nothing is stored.
"""
from __future__ import annotations

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.repositories import roles as roles_repo
from app.repositories import skills as skills_repo
from app.schemas.gap import Gap, GapRequest, GapResponse

# a role skill at or above this coverage is treated as "already have" (not a gap)
_HAVE_CUTOFF = 0.8


class GapService:
    def __init__(self, settings: Settings, embedder=None) -> None:
        self._settings = settings
        self._embedder = embedder  # used to embed break/unlinked skills that carry no skill_id

    async def compute(self, req: GapRequest, session: AsyncSession) -> GapResponse:
        role = await roles_repo.get_role_with_skills(session, req.target_role)
        if role is None:
            # unknown target role: nothing to score against
            return GapResponse(readiness=0.0, skills_have=[], gaps=[])

        ids = {rs.skill_id for rs in role.skills}
        ids |= {s.skill_id for s in req.skills if s.skill_id}
        emb = await skills_repo.get_embeddings_by_ids(session, list(ids))

        user_vecs = self._user_vectors(req.skills, emb)
        exact_ids = {s.skill_id for s in req.skills if s.skill_id and s.source != "break"}
        cov = self._coverage_map(role.skills, emb, user_vecs, exact_ids)

        exposure_w = self._settings.ai_exposure_weight(role.ai_exposure)
        readiness = self._readiness(role.skills, cov, exposure_w)
        skills_have = sorted(
            {rs.skill_name for rs in role.skills if cov.get(rs.skill_id, 0.0) >= _HAVE_CUTOFF}
        )
        gaps = self._rank_gaps(role.skills, cov, exposure_w, readiness)
        return GapResponse(readiness=round(readiness, 1), skills_have=skills_have, gaps=gaps)

    # ----------------------------------------------------------------- vectors
    def _user_vectors(self, skills, emb) -> list[tuple[np.ndarray, float]]:
        """(vector, weight) per user skill; break skills weighted down."""
        out: list[tuple[np.ndarray, float]] = []
        for s in skills:
            weight = self._settings.break_skill_weight if s.source == "break" else 1.0
            vec = None
            if s.skill_id and s.skill_id in emb:
                vec = emb[s.skill_id]
            elif self._embedder is not None and s.skill:
                vec = self._normalize(self._embedder.encode_one(s.skill))
            if vec is not None:
                out.append((vec, weight))
        return out

    def _coverage_map(self, role_skills, emb, user_vecs, exact_ids) -> dict[str, float]:
        floor = self._settings.skill_cosine_threshold
        cov: dict[str, float] = {}
        for rs in role_skills:
            if rs.skill_id in exact_ids:
                cov[rs.skill_id] = 1.0
                continue
            rv = emb.get(rs.skill_id)
            best = 0.0
            if rv is not None:
                for uv, weight in user_vecs:
                    c = float(rv @ uv)
                    if c >= floor:
                        val = c * weight
                        if val > best:
                            best = val
            cov[rs.skill_id] = min(best, 1.0)
        return cov

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
            if cov.get(rs.skill_id, 0.0) >= _HAVE_CUTOFF:
                continue
            uplift = self._uplift(role_skills, cov, exposure_w, base, rs.skill_id)
            band = "ai_digital" if rs.skill_type == "digital" else "role"
            gaps.append(
                Gap(skill=rs.skill_name, band=band, importance=float(rs.importance), uplift=uplift)
            )
        gaps.sort(key=lambda g: (g.uplift, g.importance), reverse=True)
        return gaps[:3]

    def _uplift(self, role_skills, cov: dict[str, float], exposure_w, base: float, skill_id: str) -> float:
        # marginal readiness gain if this skill were fully covered
        boosted = dict(cov)
        boosted[skill_id] = 1.0
        return round(self._readiness(role_skills, boosted, exposure_w) - base, 1)

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        v = np.asarray(vec, dtype="float32")
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v
