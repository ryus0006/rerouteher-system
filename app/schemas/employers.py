"""Schemas for employer fit and role-specific job opening endpoints."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel
from pydantic import Field, field_validator

PRIORITY_IDS = frozenset({
    "flexible_work",
    "childcare_support",
    "parental_support",
    "inclusive_workplace",
    "returning_to_work",
})
JobSearchStatus = Literal["ready", "empty", "temporarily_unavailable"]


class EmployerMatchRequest(BaseModel):
    target_role_id: str
    priorities: list[str] = Field(min_length=1, max_length=5)

    @field_validator("target_role_id")
    @classmethod
    def target_role_must_be_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("target_role_id is required")
        return value

    @field_validator("priorities")
    @classmethod
    def priorities_must_be_known_and_unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("priorities must not contain duplicates")
        unknown = set(value) - PRIORITY_IDS
        if unknown:
            raise ValueError("priorities contains an unknown priority")
        return value


class LogoOut(BaseModel):
    text: str
    bg: str
    fg: str


class ReportOut(BaseModel):
    label: str
    url: str


class JobSearchOut(BaseModel):
    status: JobSearchStatus
    searched_at: datetime | None = None


class JobOpeningOut(BaseModel):
    title: str
    url: str
    found_at: datetime


class EmployerMatchOut(BaseModel):
    id: str
    name: str
    industry: str | None = None
    location: str | None = None
    logo: LogoOut | None = None
    website: str | None = None
    summary: str | None = None
    discloses: list[str] = []
    report: ReportOut | None = None
    met: list[str] = []
    unmet: list[str] = []
    job: JobOpeningOut | None = None


class EmployerMatchResponse(BaseModel):
    job_search: JobSearchOut
    employers: list[EmployerMatchOut] = []


class JobRefreshResponse(BaseModel):
    role_id: str
    status: JobSearchStatus
    opening_count: int
    searched_at: datetime | None = None
