"""Caregiving map queries: break activity -> reframed professional label."""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class ReframedRow:
    activity_id: str
    reframed_label: str
    skill_id: str | None = None


# Cached once per process: whether caregiving_map carries the (curated) skill_id column
# that links a reframed label to a concrete skill_taxonomy skill. Until the data team adds
# it, reframe still works and returns skill_id=None (reframed skills stay display-only).
_has_skill_id: bool | None = None


async def _skill_id_available(session: AsyncSession) -> bool:
    global _has_skill_id
    if _has_skill_id is None:
        row = (
            await session.execute(
                text(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_name = 'caregiving_map' AND column_name = 'skill_id' LIMIT 1"
                )
            )
        ).first()
        _has_skill_id = row is not None
    return _has_skill_id


async def reframe(session: AsyncSession, activities: list[str]) -> list[ReframedRow]:
    """Reframe break activities. `activities` are the activity ids the UI sends."""
    if not activities:
        return []
    # Column list is fixed (never user input); include skill_id only when the column exists.
    cols = "activity_id, reframed_label" + (", skill_id" if await _skill_id_available(session) else "")
    rows = (
        await session.execute(
            text(f"SELECT DISTINCT {cols} FROM caregiving_map WHERE activity_id = ANY(:acts)"),
            {"acts": activities},
        )
    ).all()
    return [
        ReframedRow(r.activity_id, r.reframed_label, getattr(r, "skill_id", None))
        for r in rows
    ]
