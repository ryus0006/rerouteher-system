"""Schemas for CV parsing, refreshed-CV generation, and wording refinement."""
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Experience(BaseModel):
    title: str | None = None
    organisation: str | None = None
    start: str | None = None
    end: str | None = None
    description: str | None = None


class CV(BaseModel):
    raw_text: str
    experiences: list[Experience] = Field(default_factory=list)
    skill_mentions: list[str] = Field(default_factory=list)


class CVParseResponse(BaseModel):
    cv: CV


class ErrorResponse(BaseModel):
    error: str


class CvGenerateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    role_id: str | None = Field(default=None, alias="roleId", min_length=1)
    regenerate: bool = False


class CvImproveRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    role_id: str = Field(alias="roleId", min_length=1)
    section: Literal["summary", "experience", "careerBreak"]
    experience_index: int | None = Field(default=None, alias="experienceIndex", ge=0)
    # Bounded so one request cannot make the model (and its up-to-3 retries) read an
    # oversized prompt. 4000 covers the largest valid section (6 bullets of 400 chars).
    current_text: str = Field(default="", alias="currentText", max_length=4000)
    # Each item is length-checked, not sliced: cutting text before PII redaction could
    # split an email or phone number so the redaction no longer recognises it.
    previous_suggestions: list[Annotated[str, Field(max_length=4000)]] = Field(
        default_factory=list,
        alias="previousSuggestions",
    )

    @field_validator("previous_suggestions")
    @classmethod
    def _recent_suggestions(cls, value: list[str]) -> list[str]:
        # The UI appends every earlier suggestion, so keep the five most recent rather
        # than reject a long list.
        return value[-5:]


class CvPersonal(BaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""


class CvDraftExperience(BaseModel):
    title: str = ""
    organisation: str = ""
    start: str = ""
    end: str = ""
    description: str = ""


class CvDraft(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    version: int = 3
    role_id: str = Field(alias="roleId", min_length=1)
    personal: CvPersonal = Field(default_factory=CvPersonal)
    summary: str = ""
    skills: list[str] = Field(default_factory=list)
    experiences: list[CvDraftExperience] = Field(default_factory=list)
    career_break: Any | None = Field(default=None, alias="careerBreak")


class CvGenerateResponse(BaseModel):
    role_id: str
    generation_status: Literal["generated", "existing"]
    generated_at: datetime
    draft: CvDraft


class CvImproveResponse(BaseModel):
    section: Literal["summary", "experience", "careerBreak"]
    experience_index: int | None = None
    suggestion: str
    evidence: str
