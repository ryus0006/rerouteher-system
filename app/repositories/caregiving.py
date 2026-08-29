"""Caregiving map queries: break activity -> reframed professional label."""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class ReframedRow:
    break_activity: str
    reframed_label: str


async def reframe(session: AsyncSession, activities: list[str]) -> list[ReframedRow]:
    if not activities:
        return []
    rows = (
        await session.execute(
            text(
                "SELECT DISTINCT break_activity, reframed_label "
                "FROM caregiving_map WHERE break_activity = ANY(:acts)"
            ),
            {"acts": activities},
        )
    ).all()
    return [ReframedRow(r.break_activity, r.reframed_label) for r in rows]
