"""Role queries: pgvector nearest-neighbour and role-with-skills lookup."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class RoleSkillRow:
    skill_id: str
    skill_name: str
    skill_type: str
    importance: float


@dataclass
class RoleWithSkills:
    role_id: str
    role_title: str
    ai_exposure: str
    skills: list[RoleSkillRow]


@dataclass
class NearestRole:
    role_id: str
    role_title: str
    similarity: float


@dataclass
class Role:
    role_id: str
    role_title: str
    masco_code: str


async def get_by_esco_code(
    session: AsyncSession, esco_code: str, esco_title: str | None = None
) -> Role | None:
    # One ESCO code maps to several MASCO roles (e.g. 2512.4 -> Software Developer,
    # Technical Specialist (.Net), Computer Programmer, C/C++ Programmer), so pick the role
    # whose title matches the predicted ESCO title, then fall back to a deterministic order.
    row = (
        await session.execute(
            text(
                "SELECT role_id, role_title, masco_code FROM roles WHERE esco_code = :c "
                "ORDER BY (lower(role_title) = lower(:t)) DESC, "
                "(role_title ILIKE :like) DESC, masco_code "
                "LIMIT 1"
            ),
            {
                "c": esco_code,
                "t": esco_title or "",
                "like": f"%{esco_title}%" if esco_title else "",
            },
        )
    ).first()
    return Role(row.role_id, row.role_title, row.masco_code) if row else None


async def get_by_masco_code(session: AsyncSession, masco_code: str) -> Role | None:
    row = (
        await session.execute(
            text("SELECT role_id, role_title, masco_code FROM roles WHERE masco_code = :c LIMIT 1"),
            {"c": masco_code},
        )
    ).first()
    if row is None and masco_code:
        # The classifier emits a 4-digit unit-group code (e.g. 2512) while roles are stored
        # at the granular level (e.g. 251201). Resolve by group prefix, lowest code first.
        row = (
            await session.execute(
                text(
                    "SELECT role_id, role_title, masco_code FROM roles "
                    "WHERE masco_code LIKE :p ORDER BY masco_code LIMIT 1"
                ),
                {"p": f"{masco_code}%"},
            )
        ).first()
    return Role(row.role_id, row.role_title, row.masco_code) if row else None


async def get_role_with_skills_by_id(session: AsyncSession, role_id: str) -> RoleWithSkills | None:
    """Load a role and its skills by primary key, so the target role is unambiguous
    (role titles are not unique, e.g. duplicate 'Software Developer' rows)."""
    role = (
        await session.execute(
            text("SELECT role_id, role_title, ai_exposure FROM roles WHERE role_id = :rid"),
            {"rid": role_id},
        )
    ).first()
    if role is None:
        return None
    rows = (
        await session.execute(
            text(
                "SELECT skill_id, skill_name, skill_type, importance "
                "FROM role_skills WHERE role_id = :rid"
            ),
            {"rid": role.role_id},
        )
    ).all()
    skills = [
        RoleSkillRow(str(r.skill_id), r.skill_name, r.skill_type, float(r.importance))
        for r in rows
    ]
    return RoleWithSkills(role.role_id, role.role_title, role.ai_exposure, skills)


async def best_similarity_for_role(
    session: AsyncSession, role_id: str, user_skill_ids: list[str]
) -> dict[str, float]:
    """Per role skill, the max cosine similarity to any of the user's skills.

    Bounded to this role's skills vs the given user skills only (never the whole taxonomy).
    Returns skill_id -> best similarity; skills without an embedding are absent.
    """
    if not user_skill_ids:
        return {}
    rows = (
        await session.execute(
            text(
                "SELECT rs.skill_id AS rid, "
                "MAX(1 - (rt.embedding <=> ut.embedding)) AS best_sim "
                "FROM role_skills rs "
                "JOIN skill_taxonomy rt ON rt.skill_id = rs.skill_id "
                "JOIN skill_taxonomy ut ON ut.skill_id = ANY(:uids) "
                "WHERE rs.role_id = :rid "
                "GROUP BY rs.skill_id"
            ),
            {"rid": role_id, "uids": user_skill_ids},
        )
    ).all()
    return {str(r.rid): float(r.best_sim) for r in rows}


async def get_rerank_texts(session: AsyncSession, role_ids: list[str]) -> dict[str, str]:
    """role_id -> short text for reranking: title + first slice of the summary.

    Bounded to the given ids only. Absent summary -> title alone.
    """
    if not role_ids:
        return {}
    rows = (
        await session.execute(
            text(
                "SELECT role_id, role_title, "
                "trim(role_title || '. ' || "
                "left(COALESCE(NULLIF(task_summary, ''), occupation_description, ''), 300)) "
                "AS rerank_text "
                "FROM roles WHERE role_id = ANY(:ids)"
            ),
            {"ids": role_ids},
        )
    ).all()
    out: dict[str, str] = {}
    for r in rows:
        text_val = (r.rerank_text or "").strip().rstrip(".").strip()
        out[str(r.role_id)] = text_val or r.role_title
    return out


async def nearest_by_embedding(
    session: AsyncSession, query_vec: np.ndarray, k: int
) -> list[NearestRole]:
    """Cosine kNN over roles.role_embedding (pgvector `<=>`, similarity = 1 - distance)."""
    vec_literal = "[" + ",".join(f"{x:.6f}" for x in query_vec.tolist()) + "]"
    rows = (
        await session.execute(
            text(
                "SELECT role_id, role_title, "
                "1 - (role_embedding <=> CAST(:v AS vector)) AS similarity "
                "FROM roles WHERE flexible_role = true "
                "ORDER BY role_embedding <=> CAST(:v AS vector) LIMIT :k"
            ),
            {"v": vec_literal, "k": k},
        )
    ).all()
    return [NearestRole(r.role_id, r.role_title, float(r.similarity)) for r in rows]
