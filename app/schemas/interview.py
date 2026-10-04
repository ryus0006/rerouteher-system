"""Public request/response models for the E7 AI Interview Coach endpoints.

Never carries username, audio bytes, Gemini prompts, or raw saved-journey JSON --
only what a separate UI team needs to render setup, sessions, attempts and areas.
"""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

PracticeFocus = Literal["general", "role_specific", "mixed"]
SessionStatus = Literal["active", "completed"]
FeedbackStatus = Literal["pending", "processing", "ready", "error"]


class RoleOut(BaseModel):
    role_id: str
    role_title: str


class InterviewSetupOut(BaseModel):
    selected_role: RoleOut
    available_roles: list[RoleOut] = []
    focus_choices: list[PracticeFocus] = ["general", "role_specific", "mixed"]
    question_count: int = 5


class CreateSessionRequest(BaseModel):
    role_id: str
    practice_focus: PracticeFocus


class SessionSummaryOut(BaseModel):
    session_id: str
    role: RoleOut
    practice_focus: PracticeFocus
    status: SessionStatus
    progress: int
    created_at: datetime
    updated_at: datetime


class FeedbackItemOut(BaseModel):
    criterion_id: str
    title: str
    detail: str


class AttemptOut(BaseModel):
    response_id: int
    attempt_no: int
    transcript: str | None
    feedback_status: FeedbackStatus
    feedback_summary: str | None
    strengths: list[FeedbackItemOut] = []
    improvements: list[FeedbackItemOut] = []
    detected_language: str | None
    duration_s: float | None
    error_code: str | None
    created_at: datetime
    updated_at: datetime
    content_expired: bool


class QuestionSlotOut(BaseModel):
    sequence_no: int
    question_id: str
    question_text: str
    category: str
    difficulty: str
    role_id: str | None
    kind: Literal["general", "role_specific"]
    attempts: list[AttemptOut] = []


class SessionDetailOut(BaseModel):
    session_id: str
    role: RoleOut
    practice_focus: PracticeFocus
    status: SessionStatus
    created_at: datetime
    updated_at: datetime
    questions: list[QuestionSlotOut] = []


class AreaOut(BaseModel):
    criterion_id: str
    title: str
    response_count: int


class AreasOut(BaseModel):
    improvements: list[AreaOut] = []
    strengths: list[AreaOut] = []


class ErrorOut(BaseModel):
    error: str
