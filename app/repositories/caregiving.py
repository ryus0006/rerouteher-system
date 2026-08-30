"""Caregiving map queries: break activity -> reframed professional label."""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class ReframedRow:
    activity_id: str
    reframed_label: str


async def reframe(session: AsyncSession, activities: list[str]) -> list[ReframedRow]:
    """Reframe break activities. `activities` are the activity ids the UI sends."""
    if not activities:
        return []
    rows = (
        await session.execute(
            text(
                "SELECT DISTINCT activity_id, reframed_label "
                "FROM caregiving_map WHERE activity_id = ANY(:acts)"
            ),
            {"acts": activities},
        )
    ).all()
    return [ReframedRow(r.activity_id, r.reframed_label) for r in rows]
