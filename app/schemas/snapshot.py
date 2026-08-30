"""Schemas for POST /api/snapshot/generate."""
from typing import Literal

from pydantic import BaseModel

from app.schemas.cv import CV


class Break(BaseModel):
    duration_years: float
    activities: list[str] = []


class SnapshotRequest(BaseModel):
    cv: CV
    break_: Break

    model_config = {"populate_by_name": True}

    # accept JSON key "break" (a Python keyword) via alias
    def __init__(self, **data):  # noqa: D401
        if "break" in data and "break_" not in data:
            data["break_"] = data.pop("break")
        super().__init__(**data)


class ProfessionalSkill(BaseModel):
    skill: str
    skill_id: str | None = None
    source: Literal["experience"] = "experience"
    evidence: str | None = None


class ReframedSkill(BaseModel):
    skill: str
    skill_id: str | None = None
    source: Literal["break"] = "break"
    from_activity: str | None = None


class PreviousOccupation(BaseModel):
    role: str
    role_id: str
    confidence: float
    method: Literal["classifier", "embedding"]


class RecommendedRole(BaseModel):
    role: str
    role_id: str
    similarity: float


class SnapshotResponse(BaseModel):
    professional_skills: list[ProfessionalSkill] = []
    reframed_skills: list[ReframedSkill] = []
    previous_occupation: PreviousOccupation | None = None
    recommended_roles: list[RecommendedRole] = []
