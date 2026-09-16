"""Schemas for POST /api/companion/ask (E8)."""
from pydantic import BaseModel, ConfigDict, Field

from app.schemas.cv import CV
from app.schemas.snapshot import Break


class SkillChoice(BaseModel):
    skill_id: str
    skill_name: str


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
    # Skills she already ticked from a role checklist, so the companion does not re-offer them.
    confirmedSkills: list[SkillChoice] = []
    # The role a checklist was last offered for, echoed back so the companion does not
    # re-offer the same role's skills but still re-offers when she changes her role.
    roleSkillsOfferedForRoleId: str | None = None
    # Her computed employer matches (EmployerMatchOut shape), carried so the companion can
    # explain them and cite the report behind each. Read defensively via .get.
    employerMatches: list[dict] = []


class AskRequest(BaseModel):
    question: str
    session_id: str
    journey: JourneyIn = JourneyIn()
    current_page: str | None = None


class JourneyUpdate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    cv: CV | None = None
    break_: Break | None = Field(default=None, alias="break")
    # Her top workplace priorities (fixed pick-list ids), captured in the profile
    # builder so E9 employer matching has them even when the Priorities page is skipped.
    employerPriorities: list[str] = []


class CtaOut(BaseModel):
    # Optional call-to-action the companion offers; the frontend renders it as a
    # button that navigates client-side while keeping the chat open. Never auto-fired.
    label: str
    to: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str] = []
    journey_update: JourneyUpdate | None = None
    cta: CtaOut | None = None
    # A checklist of role skills the companion offers her to tick (US8.1.17).
    skill_choices: list[SkillChoice] | None = None
    # The role_id the skill_choices belong to; the frontend records it as
    # roleSkillsOfferedForRoleId so the same role is not offered again.
    skill_choices_role_id: str | None = None
