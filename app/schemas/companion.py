"""Schemas for POST /api/companion/ask (E8)."""
from pydantic import BaseModel, ConfigDict, Field

from app.schemas.cv import CV
from app.schemas.snapshot import Break


class JourneyIn(BaseModel):
    # Her client-side journey, carried in so the backend can ground answers.
    # All optional; US8.1 reads cv (and may set break). "break" is a Python
    # keyword, so it is aliased.
    model_config = ConfigDict(populate_by_name=True)
    cv: CV | None = None
    break_: Break | None = Field(default=None, alias="break")
    snapshot: dict | None = None
    gapResult: dict | None = None
    selectedRole: dict | None = None
    employerPriorities: list[str] = []


class AskRequest(BaseModel):
    question: str
    session_id: str
    journey: JourneyIn = JourneyIn()
    current_page: str | None = None


class JourneyUpdate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    cv: CV | None = None
    break_: Break | None = Field(default=None, alias="break")


class AskResponse(BaseModel):
    answer: str
    sources: list[str] = []
    journey_update: JourneyUpdate | None = None
