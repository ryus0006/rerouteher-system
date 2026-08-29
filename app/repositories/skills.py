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


@dataclass
class SkillRow:
    skill_id: str
    canonical_name: str
    aliases: str | None
    skill_type: str


async def list_skills(session: AsyncSession) -> list[SkillRow]:
    """Full skill rows for building a term -> canonical lookup."""
    rows = (
        await session.execute(
            text("SELECT skill_id, canonical_name, aliases, skill_type FROM skill_taxonomy")
        )
    ).all()
    return [SkillRow(r.skill_id, r.canonical_name, r.aliases, r.skill_type) for r in rows]


async def load_alias_dictionary(session: AsyncSession) -> list[tuple[str, str]]:
    """Return (skill_id, term) pairs from canonical_name + aliases for PhraseMatcher / rapidfuzz."""
    rows = (
        await session.execute(text("SELECT skill_id, canonical_name, aliases FROM skill_taxonomy"))
    ).all()
    pairs: list[tuple[str, str]] = []
    for r in rows:
        pairs.append((r.skill_id, r.canonical_name))
        if r.aliases:
            for alias in str(r.aliases).split("|"):
                alias = alias.strip()
                if alias:
                    pairs.append((r.skill_id, alias))
    return pairs


async def match_by_embedding(
    session: AsyncSession, query_vec: np.ndarray, k: int, threshold: float
) -> list[SkillMatch]:
    """Cosine kNN over skill_taxonomy.embedding, keeping matches above `threshold`."""
    vec_literal = "[" + ",".join(f"{x:.6f}" for x in query_vec.tolist()) + "]"
    rows = (
        await session.execute(
            text(
                "SELECT skill_id, canonical_name, "
                "1 - (embedding <=> CAST(:v AS vector)) AS similarity "
                "FROM skill_taxonomy "
                "ORDER BY embedding <=> CAST(:v AS vector) LIMIT :k"
            ),
            {"v": vec_literal, "k": k},
        )
    ).all()
    return [
        SkillMatch(r.skill_id, r.canonical_name, float(r.similarity))
        for r in rows
        if float(r.similarity) >= threshold
    ]
