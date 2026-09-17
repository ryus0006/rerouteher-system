"""Schemas for POST /api/learning/recommend."""
from pydantic import BaseModel


class LearningRequest(BaseModel):
    # Deterministic identifiers only: resources are matched by ESCO skill_id, so
    # ambiguous display names never drive the lookup. Names are resolved server-side.
    skill_ids: list[str] = []
    target_role_id: str
    # Optional human label, kept for logs only; never used to resolve anything.
    target_role: str | None = None


class LearningResourceOut(BaseModel):
    id: str
    skill_id: str
    title: str
    provider: str
    logo: str | None = None
    format: str
    minutes: int | None = None
    cost: str
    free: bool
    url: str
    why: str


class LearningGroupOut(BaseModel):
    skill_id: str
    skill: str
    icon: str | None = None
    blurb: str | None = None


class LearningResponse(BaseModel):
    groups: list[LearningGroupOut] = []
    resources: list[LearningResourceOut] = []
