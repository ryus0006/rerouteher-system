"""Schemas for CV parsing, refreshed-CV generation, and wording refinement."""
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


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
    section: Literal["summary", "experience"]
    experience_index: int | None = Field(default=None, alias="experienceIndex", ge=0)
    current_text: str = Field(default="", alias="currentText")
    previous_suggestions: list[str] = Field(
        default_factory=list,
        alias="previousSuggestions",
    )


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
    section: Literal["summary", "experience"]
    experience_index: int | None = None
    suggestion: str
    evidence: str
