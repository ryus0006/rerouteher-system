"""Read-only queries for E6 learning recommendations.

Reference tables live in the rerouteher schema; the connection search_path
resolves the unqualified names. Requests write nothing.
"""
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class SkillLabel:
    skill_id: str
    name: str
    definition: str | None


@dataclass
class CuratedResource:
    skill_id: str
    resource_id: str
    title: str
    url: str
    delivery_mode: str | None
    duration_minutes: int | None
    licence_note: str | None
    evidence_note: str | None
    provider_name: str


async def get_skill_labels(session: AsyncSession, skill_ids: list[str]) -> dict[str, SkillLabel]:
    """skill_id -> canonical name + definition, for the given ids only."""
    if not skill_ids:
        return {}
    rows = (
        await session.execute(
            text(
                "SELECT skill_id, canonical_name, definition "
                "FROM skill_taxonomy WHERE skill_id = ANY(:ids)"
            ),
            {"ids": skill_ids},
        )
    ).all()
    return {
        str(r.skill_id): SkillLabel(str(r.skill_id), r.canonical_name, r.definition)
        for r in rows
    }


def _to_minutes(duration) -> int | None:
    if duration is None:
        return None
    if isinstance(duration, timedelta):
        return int(duration.total_seconds() // 60)
    return None


async def get_curated_resources(
    session: AsyncSession, skill_ids: list[str]
) -> list[CuratedResource]:
    """Curated resources for the given skills, best relevance first per skill."""
    if not skill_ids:
        return []
    rows = (
        await session.execute(
            text(
                "SELECT lrs.skill_id, lr.resource_id, lr.title, lr.url, "
                "lr.delivery_mode, lr.duration, lr.licence_note, "
                "lrs.evidence_note, p.provider_name "
                "FROM learning_resource_skill lrs "
                "JOIN learning_resource lr ON lr.resource_id = lrs.resource_id "
                "JOIN provider p ON p.provider_id = lr.provider_id "
                "WHERE lrs.skill_id = ANY(:ids) "
                "ORDER BY lrs.skill_id, lrs.relevance DESC NULLS LAST"
            ),
            {"ids": skill_ids},
        )
    ).all()
    return [
        CuratedResource(
            skill_id=str(r.skill_id),
            resource_id=str(r.resource_id),
            title=r.title or "",
            url=r.url or "",
            delivery_mode=r.delivery_mode,
            duration_minutes=_to_minutes(r.duration),
            licence_note=r.licence_note,
            evidence_note=r.evidence_note,
            provider_name=r.provider_name,
        )
        for r in rows
    ]
