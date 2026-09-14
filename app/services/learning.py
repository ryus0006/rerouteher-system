"""E6 learning recommendation service.

Deterministic id-driven lookup: curated resources first (from the reference
tables), then a single YouTube search resource for any gap with no curated hit,
so every gap she was shown always has at least one place to start. No PII is
read or emitted; only skill ids (resolved to names server-side) and the role.
"""
import logging
from urllib.parse import quote_plus

from app.repositories import learning as learning_repo
from app.schemas.learning import (
    LearningGroupOut,
    LearningRequest,
    LearningResourceOut,
    LearningResponse,
)

logger = logging.getLogger("rerouteher")

MAX_BLURB_CHARS = 140


def _blurb(definition: str | None) -> str | None:
    if not definition:
        return None
    text = definition.strip()
    if len(text) <= MAX_BLURB_CHARS:
        return text
    return text[:MAX_BLURB_CHARS].rstrip() + "..."


def _youtube_resource(skill_id: str, name: str) -> LearningResourceOut:
    query = quote_plus(f"{name} tutorial")
    return LearningResourceOut(
        id=f"search-{skill_id}",
        skill_id=skill_id,
        title=f"{name} tutorials",
        provider="YouTube",
        logo="youtube",
        format="Video",
        minutes=None,
        cost="Free",
        free=True,
        url=f"https://www.youtube.com/results?search_query={query}",
        why=f"A starting point for building {name}, from a search of free video lessons.",
    )


# A handful of provider names map to a logo the page already ships; everything
# else falls back to the provider's initials in the UI.
_LOGO_BY_PROVIDER = {
    "youtube": "youtube",
    "figma": "figma",
    "figma learn": "figma",
    "nielsen norman group": "nngroup",
    "openai": "openai",
}


def _curated_resource(row: learning_repo.CuratedResource) -> LearningResourceOut:
    why = (row.evidence_note or "").strip() or (
        f"A curated resource from {row.provider_name} for this skill."
    )
    return LearningResourceOut(
        id=row.resource_id,
        skill_id=row.skill_id,
        title=row.title or row.provider_name,
        provider=row.provider_name,
        logo=_LOGO_BY_PROVIDER.get(row.provider_name.lower().strip()),
        format=(row.delivery_mode or "Course"),
        minutes=row.duration_minutes,
        cost="Free",
        free=True,
        url=row.url,
        why=why,
    )


class LearningService:
    def __init__(self, repo=learning_repo) -> None:
        # repo is injectable so the service is testable without a DB.
        self._repo = repo

    async def recommend(self, req: LearningRequest, session) -> LearningResponse:
        if not req.skill_ids:
            return LearningResponse(groups=[], resources=[])

        labels = await self._repo.get_skill_labels(session, req.skill_ids)
        curated_rows = await self._repo.get_curated_resources(session, req.skill_ids)

        by_skill: dict[str, list[learning_repo.CuratedResource]] = {}
        for row in curated_rows:
            by_skill.setdefault(row.skill_id, []).append(row)

        groups: list[LearningGroupOut] = []
        resources: list[LearningResourceOut] = []
        # Preserve the request order (the gap's uplift order) so the most
        # valuable focus area is first on the page.
        for skill_id in req.skill_ids:
            label = labels.get(skill_id)
            if label is None:
                continue  # unresolved id: no group, no broken entry (AC 6.1.4)
            groups.append(
                LearningGroupOut(
                    skill_id=skill_id,
                    skill=label.name,
                    icon=None,
                    blurb=_blurb(label.definition),
                )
            )
            hits = by_skill.get(skill_id, [])
            if hits:
                resources.extend(_curated_resource(row) for row in hits)
            else:
                resources.append(_youtube_resource(skill_id, label.name))

        logger.info(
            "learning: role=%s skills=%d groups=%d curated=%d resources=%d",
            req.target_role_id,
            len(req.skill_ids),
            len(groups),
            len(curated_rows),
            len(resources),
        )
        return LearningResponse(groups=groups, resources=resources)
