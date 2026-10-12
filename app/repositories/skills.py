"""Skill taxonomy queries: alias lookup and pgvector semantic match."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class SkillMatch:
    skill_id: str
    canonical_name: str
    similarity: float
    definition: str = ""  # ESCO description, used to re-rank candidates in context


@dataclass
class SkillRow:
    skill_id: str
    canonical_name: str
    skill_type: str
    definition: str | None = None


async def get_skill_by_id(session: AsyncSession, skill_id: str) -> SkillRow | None:
    row = (
        await session.execute(
            text(
                "SELECT skill_id, canonical_name, skill_type, definition "
                "FROM skill_taxonomy WHERE skill_id = :sid"
            ),
            {"sid": skill_id},
        )
    ).first()
    return (
        SkillRow(row.skill_id, row.canonical_name, row.skill_type, row.definition)
        if row
        else None
    )


async def find_exact_skills(
    session: AsyncSession, query: str, limit: int = 3
) -> list[SkillRow]:
    rows = (
        await session.execute(
            text(
                "SELECT DISTINCT st.skill_id, st.canonical_name, st.skill_type, st.definition "
                "FROM skill_taxonomy st "
                "LEFT JOIN skill_aliases sa ON sa.skill_id = st.skill_id "
                "WHERE lower(st.canonical_name) = lower(:q) "
                "OR lower(sa.alias) = lower(:q) "
                "ORDER BY st.canonical_name LIMIT :lim"
            ),
            {"q": query.strip(), "lim": limit},
        )
    ).all()
    return [
        SkillRow(r.skill_id, r.canonical_name, r.skill_type, r.definition)
        for r in rows
    ]


async def list_skills(session: AsyncSession) -> list[SkillRow]:
    """Canonical skill rows for building a skill_id -> canonical lookup."""
    rows = (
        await session.execute(
            text(
                "SELECT skill_id, canonical_name, skill_type, definition "
                "FROM skill_taxonomy"
            )
        )
    ).all()
    return [
        SkillRow(r.skill_id, r.canonical_name, r.skill_type, r.definition)
        for r in rows
    ]


async def load_alias_dictionary(session: AsyncSession) -> list[tuple[str, str]]:
    """(skill_id, term) pairs from canonical names and the skill_aliases table."""
    rows = (
        await session.execute(
            text(
                "SELECT skill_id, canonical_name AS term FROM skill_taxonomy "
                "UNION ALL "
                "SELECT skill_id, alias AS term FROM skill_aliases"
            )
        )
    ).all()
    return [(r.skill_id, r.term) for r in rows if r.term]


async def match_by_embedding(
    session: AsyncSession, query_vec: np.ndarray, k: int, threshold: float
) -> list[SkillMatch]:
    """Cosine kNN over skill_taxonomy.embedding, keeping matches above `threshold`."""
    vec_literal = "[" + ",".join(f"{x:.6f}" for x in query_vec.tolist()) + "]"
    rows = (
        await session.execute(
            text(
                "SELECT skill_id, canonical_name, COALESCE(definition, '') AS definition, "
                "1 - (embedding <=> CAST(:v AS vector)) AS similarity "
                "FROM skill_taxonomy "
                "ORDER BY embedding <=> CAST(:v AS vector) LIMIT :k"
            ),
            {"v": vec_literal, "k": k},
        )
    ).all()
    return [
        SkillMatch(r.skill_id, r.canonical_name, float(r.similarity), r.definition)
        for r in rows
        if float(r.similarity) >= threshold
    ]
