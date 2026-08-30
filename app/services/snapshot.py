"""Skill snapshot + occupation match.

Hybrid skill extraction (exact alias + semantic embedding); occupation matching runs a
TF-IDF matcher (Tier 1) whose MASCO code is mapped to a role in the role table, falling
back to an embedding match over the role pool (Tier 2) when no MASCO code resolves.
Recommended roles = the matched role pinned first + the next resolved roles (or nearest
by embedding kNN in the fallback). Query text is alias-normalized before both tiers.
Computed at request time, nothing stored.
"""
from __future__ import annotations

import logging
import re

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.repositories import caregiving as caregiving_repo
from app.repositories import roles as roles_repo
from app.repositories import skills as skills_repo
from app.schemas.snapshot import (
    PreviousOccupation,
    ProfessionalSkill,
    RecommendedRole,
    ReframedSkill,
    SnapshotRequest,
    SnapshotResponse,
)
from app.services.occupation_matcher import normalize_text

logger = logging.getLogger("rerouteher")

_SENTENCE_SPLIT_RE = re.compile(r"[.;•\n]+")
_WORD_RE = re.compile(r"[a-z][a-z0-9+.#-]*")
_MAX_SEMANTIC_SPANS = 20
_MAX_PHRASE_WORDS = 4
_MAX_SEMANTIC_SKILLS = 10


def _phrase_set(text_lower: str, max_n: int = _MAX_PHRASE_WORDS) -> set[str]:
    """Word/phrase tokens (1..max_n grams) so short skill terms match on word
    boundaries, not as substrings inside larger words."""
    words = _WORD_RE.findall(text_lower)
    phrases: set[str] = set()
    for n in range(1, max_n + 1):
        for i in range(len(words) - n + 1):
            phrases.add(" ".join(words[i : i + n]))
    return phrases


class SnapshotService:
    def __init__(self, settings: Settings, embedder, tfidf_matcher) -> None:
        self._settings = settings
        self._embedder = embedder  # may be None if the model is unavailable
        self._tfidf = tfidf_matcher  # ESCO TF-IDF matcher (Tier 1), may be None if the artifact is absent
        # lazily loaded skill lookup (cached on this singleton instance)
        self._term_to_skill: dict[str, str] | None = None
        self._skill_to_canonical: dict[str, str] = {}

    async def generate(self, req: SnapshotRequest, session: AsyncSession) -> SnapshotResponse:
        await self._ensure_skill_lookup(session)
        professional = await self._extract_skills(req, session)
        reframed = await self._reframe_break(req, session)
        previous, recommended = await self._match_occupation(req, professional, session)
        return SnapshotResponse(
            professional_skills=professional,
            reframed_skills=reframed,
            previous_occupation=previous,
            recommended_roles=recommended,
        )

    # ------------------------------------------------------------- skill lookup
    async def _ensure_skill_lookup(self, session: AsyncSession) -> None:
        if self._term_to_skill is not None:
            return
        for row in await skills_repo.list_skills(session):
            self._skill_to_canonical[row.skill_id] = row.canonical_name
        term_to_skill: dict[str, str] = {}
        for skill_id, term in await skills_repo.load_alias_dictionary(session):
            key = term.strip().lower()
            if key:
                term_to_skill.setdefault(key, skill_id)
        self._term_to_skill = term_to_skill

    # ----------------------------------------------------------------- skills
    async def _extract_skills(
        self, req: SnapshotRequest, session: AsyncSession
    ) -> list[ProfessionalSkill]:
        cv = req.cv
        assert self._term_to_skill is not None
        found: dict[str, ProfessionalSkill] = {}

        # 1. exact alias pass over the CV text (PhraseMatcher-equivalent, word-boundary safe)
        phrases = _phrase_set(cv.raw_text.lower())
        for term, skill_id in self._term_to_skill.items():
            if term in phrases:
                found.setdefault(skill_id, self._professional(skill_id, term))

        # 2. the extractor's fuzzy skill_mentions (surface terms already matched)
        for mention in cv.skill_mentions:
            skill_id = self._term_to_skill.get(mention.lower())
            if skill_id:
                found.setdefault(skill_id, self._professional(skill_id, mention))

        # 3. semantic pass: catch skills phrased differently in the experience text
        if self._embedder is not None:
            for match in await self._semantic_skills(cv, session):
                found.setdefault(
                    match.skill_id,
                    ProfessionalSkill(skill=match.canonical_name, source="experience", evidence="semantic match"),
                )

        return list(found.values())

    def _professional(self, skill_id: str, evidence: str) -> ProfessionalSkill:
        return ProfessionalSkill(
            skill=self._skill_to_canonical.get(skill_id, skill_id),
            source="experience",
            evidence=evidence,
        )

    async def _semantic_skills(self, cv, session: AsyncSession) -> list[skills_repo.SkillMatch]:
        spans: list[str] = []
        for exp in cv.experiences:
            if exp.title:
                spans.append(exp.title)
            if exp.description:
                spans.extend(s.strip() for s in _SENTENCE_SPLIT_RE.split(exp.description) if s.strip())
        spans = list(dict.fromkeys(spans))[:_MAX_SEMANTIC_SPANS]
        if not spans:
            return []

        vectors = self._embedder.encode(spans)
        best: dict[str, skills_repo.SkillMatch] = {}
        for vec in vectors:
            matches = await skills_repo.match_by_embedding(
                session, vec, k=3, threshold=self._settings.skill_cosine_threshold
            )
            for m in matches:
                if m.skill_id not in best or m.similarity > best[m.skill_id].similarity:
                    best[m.skill_id] = m
        # keep only the strongest semantic matches so the dense taxonomy does not flood the list
        ranked = sorted(best.values(), key=lambda m: m.similarity, reverse=True)
        return ranked[:_MAX_SEMANTIC_SKILLS]

    # ------------------------------------------------------------- break reframe
    async def _reframe_break(
        self, req: SnapshotRequest, session: AsyncSession
    ) -> list[ReframedSkill]:
        rows = await caregiving_repo.reframe(session, req.break_.activities)
        seen: set[tuple[str, str]] = set()
        reframed: list[ReframedSkill] = []
        for row in rows:
            key = (row.reframed_label, row.activity_id)
            if key in seen:
                continue
            seen.add(key)
            reframed.append(
                ReframedSkill(skill=row.reframed_label, source="break", from_activity=row.activity_id)
            )
        return reframed

    # ------------------------------------------------------------- occupation
    async def _match_occupation(
        self, req: SnapshotRequest, professional: list[ProfessionalSkill], session: AsyncSession
    ) -> tuple[PreviousOccupation | None, list[RecommendedRole]]:
        # alias-normalize the query inputs once so both tiers see the same normalized text
        job_title = normalize_text(next((e.title for e in req.cv.experiences if e.title), ""))
        skill_names = [normalize_text(p.skill) for p in professional] or [
            normalize_text(m) for m in req.cv.skill_mentions
        ]

        # profile embedding: tops up the classifier recommendations and drives the Tier 2 fallback
        emb_text = self._embedding_text(job_title, skill_names)
        profile_vec = (
            self._embedder.encode_one(emb_text) if (self._embedder is not None and emb_text) else None
        )

        # Tier 1: TF-IDF matcher -> map each match's MASCO code to a role in the role table
        if self._tfidf is not None and (job_title or skill_names):
            tier1 = []
            try:
                tier1 = self._tfidf.predict(job_title=job_title, skills=skill_names, top_k=5)
                logger.info("occupation Tier 1 (tfidf): %s", tier1)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Tier 1 occupation matcher failed: %s", exc)
            resolved = await self._roles_from_matches(tier1, session)
            if resolved:
                top_role, top_score = resolved[0]
                previous = PreviousOccupation(
                    role=top_role.role_title, confidence=round(top_score, 3), method="classifier"
                )
                recommended = [
                    RecommendedRole(role=role.role_title, similarity=1.0 if i == 0 else round(score, 3))
                    for i, (role, score) in enumerate(resolved[:3])
                ]
                # top up to 3 with the nearest roles by embedding, excluding ones already listed
                if len(recommended) < 3 and profile_vec is not None:
                    seen_ids = {role.role_id for role, _ in resolved}
                    seen_titles = {r.role for r in recommended}
                    for r in await roles_repo.nearest_by_embedding(session, profile_vec, k=6):
                        if len(recommended) >= 3:
                            break
                        if r.role_id not in seen_ids and r.role_title not in seen_titles:
                            recommended.append(RecommendedRole(role=r.role_title, similarity=round(r.similarity, 3)))
                            seen_titles.add(r.role_title)
                return previous, recommended

        # Tier 2 fallback: embedding match over the role pool
        if profile_vec is None:
            return None, []
        nearest = await roles_repo.nearest_by_embedding(session, profile_vec, k=1)
        if not nearest:
            return None, []
        top = nearest[0]
        previous = PreviousOccupation(
            role=top.role_title, confidence=round(top.similarity, 3), method="embedding"
        )
        recommended = [RecommendedRole(role=previous.role, similarity=1.0)]
        for r in await roles_repo.nearest_by_embedding(session, profile_vec, k=4):
            if len(recommended) >= 3:
                break
            if r.role_title != previous.role:
                recommended.append(RecommendedRole(role=r.role_title, similarity=round(r.similarity, 3)))
        return previous, recommended

    @staticmethod
    async def _roles_from_matches(matches, session: AsyncSession):
        """Resolve matches to roles by ESCO code (precise), then MASCO code as fallback.

        Deduped, input order preserved.
        """
        resolved: list[tuple] = []
        seen: set[str] = set()
        for m in matches:
            role = None
            if getattr(m, "esco_code", None):
                role = await roles_repo.get_by_esco_code(session, m.esco_code)
            if role is None and m.masco_code:
                role = await roles_repo.get_by_masco_code(session, m.masco_code)
            if role and role.role_id not in seen:
                seen.add(role.role_id)
                resolved.append((role, m.score))
        return resolved

    @staticmethod
    def _embedding_text(job_title: str, skill_names: list[str]) -> str:
        parts = [job_title, *skill_names]
        return " ".join(p for p in parts if p).strip()
