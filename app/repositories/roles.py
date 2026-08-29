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


async def get_by_masco_code(session: AsyncSession, masco_code: str) -> Role | None:
    row = (
        await session.execute(
            text("SELECT role_id, role_title, masco_code FROM roles WHERE masco_code = :c LIMIT 1"),
            {"c": masco_code},
        )
    ).first()
    return Role(row.role_id, row.role_title, row.masco_code) if row else None


async def get_role_with_skills(session: AsyncSession, role_title: str) -> RoleWithSkills | None:
    role = (
        await session.execute(
            text(
                "SELECT role_id, role_title, ai_exposure FROM roles "
                "WHERE role_title = :t LIMIT 1"
            ),
            {"t": role_title},
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
        RoleSkillRow(r.skill_id, r.skill_name, r.skill_type, float(r.importance)) for r in rows
    ]
    return RoleWithSkills(role.role_id, role.role_title, role.ai_exposure, skills)


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
