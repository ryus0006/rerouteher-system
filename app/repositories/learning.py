"""Read-only queries for E6 learning recommendations.

Reference tables live in the rerouteher schema; the connection search_path
resolves the unqualified names. Requests write nothing.
"""
import re
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


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _provider_slug(name: str) -> str:
    slug = _SLUG_RE.sub("-", (name or "").lower()).strip("-")
    return f"web-{slug}" if slug else "web"


async def skills_missing_resources(session: AsyncSession, skill_ids: list[str]) -> list[str]:
    """Of the given skill ids, those with no curated learning_resource_skill row.

    Input order is preserved so the fill honours the gap's uplift ordering.
    """
    if not skill_ids:
        return []
    rows = (
        await session.execute(
            text("SELECT skill_id FROM learning_resource_skill WHERE skill_id = ANY(:ids)"),
            {"ids": skill_ids},
        )
    ).all()
    have = {str(r.skill_id) for r in rows}
    return [s for s in skill_ids if s not in have]


async def upsert_filled_resource(session: AsyncSession, chosen) -> None:
    """Persist one AI-filled resource: provider + learning_resource + link.

    Idempotent per skill (resource_id = 'ai-<skill_id>', ON CONFLICT DO NOTHING).
    The provider row is ensured by name, then the actual provider_id is read back
    so the FK always points at an existing row (an existing provider may use a
    different id than our slug). Caller owns the transaction (commit).
    """
    slug = _provider_slug(chosen.provider_name)
    await session.execute(
        text(
            "INSERT INTO provider (provider_id, provider_name) "
            "VALUES (:pid, :pname) ON CONFLICT DO NOTHING"
        ),
        {"pid": slug, "pname": chosen.provider_name},
    )
    row = (
        await session.execute(
            text("SELECT provider_id FROM provider WHERE provider_name = :pname"),
            {"pname": chosen.provider_name},
        )
    ).first()
    provider_id = str(row.provider_id) if row else slug

    resource_id = f"ai-{chosen.skill_id}"
    duration = f"{chosen.duration_minutes} minutes" if chosen.duration_minutes else None
    await session.execute(
        text(
            "INSERT INTO learning_resource "
            "(resource_id, provider_id, title, url, delivery_mode, duration, "
            " language, level, licence_note, last_verified_at) "
            "VALUES (:rid, :pid, :title, :url, :mode, CAST(:duration AS interval), "
            " 'en', :level, :licence, now()) "
            "ON CONFLICT (resource_id) DO NOTHING"
        ),
        {
            "rid": resource_id, "pid": provider_id, "title": chosen.title,
            "url": chosen.url, "mode": chosen.delivery_mode, "duration": duration,
            "level": chosen.level, "licence": chosen.licence_note,
        },
    )
    await session.execute(
        text(
            "INSERT INTO learning_resource_skill "
            "(resource_id, skill_id, relevance, evidence_note) "
            "VALUES (:rid, :sid, :rel, :note) "
            "ON CONFLICT (resource_id, skill_id) DO NOTHING"
        ),
        {"rid": resource_id, "sid": chosen.skill_id,
         "rel": chosen.relevance, "note": chosen.evidence_note},
    )
