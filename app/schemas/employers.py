"""Schemas for POST /api/employers/match."""
from pydantic import BaseModel


class EmployerMatchRequest(BaseModel):
    # Priorities travel as ids so a match can name exactly which one an employer
    # discloses, and which it is silent on. Up to three (enforced client-side).
    priorities: list[str] = []
    # The page sends selectedRole?.role_id, which may be absent; kept for logs only.
    target_role_id: str | None = None


class LogoOut(BaseModel):
    text: str
    bg: str
    fg: str


class ReportOut(BaseModel):
    label: str
    url: str


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


class EmployerMatchResponse(BaseModel):
    employers: list[EmployerMatchOut] = []
