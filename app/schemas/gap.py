"""Schemas for POST /api/gap/compute."""
from typing import Literal

from pydantic import BaseModel


class UserSkill(BaseModel):
    skill: str
    source: str | None = None


class GapRequest(BaseModel):
    skills: list[UserSkill] = []
    target_role: str


class Gap(BaseModel):
    skill: str
    band: Literal["role", "ai_digital"]
    importance: float
    uplift: float


class GapResponse(BaseModel):
    readiness: float
    skills_have: list[str] = []
    gaps: list[Gap] = []
