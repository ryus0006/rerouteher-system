"""Schemas for POST /api/gap/compute."""
from typing import Literal

from pydantic import BaseModel


class GapRequest(BaseModel):
    # Deterministic identifiers only: the role is resolved by primary key and skills are
    # matched by ESCO skill_id, so ambiguous display names never drive the computation.
    skill_ids: list[str] = []
    target_role_id: str
    # Optional human label, kept for logs only; never used to resolve the role.
    target_role: str | None = None


class Gap(BaseModel):
    skill: str
    band: Literal["role", "ai_digital"]
    importance: float
    uplift: float


class GapResponse(BaseModel):
    readiness: float
    skills_have: list[str] = []
    gaps: list[Gap] = []
