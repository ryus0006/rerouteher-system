"""E7 AI Interview Coach routes. Every route requires a signed-in username, read only
from the session cookie -- the client can never supply or override it. Service errors
map to the approved error contract; mutating routes commit at the route edge (a
preserved transcript is committed even when feedback fails), and any other raised
exception rolls back via the normal request-session lifecycle.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_session
from app.schemas.interview import (
    AreasOut,
    CreateSessionRequest,
    InterviewSetupOut,
    SessionDetailOut,
    SessionSummaryOut,
)
from app.services.interview import InterviewError
from app.services.transcription import AudioInput

router = APIRouter(prefix="/api/interview", tags=["interview"])

_STATUS_BY_CODE = {
    "journey_prerequisite_incomplete": 409,
    "invalid_setup": 422,
    "interview_resource_not_found": 404,
    "unsupported_audio_type": 415,
    "recording_too_large": 413,
    "invalid_audio": 422,
    "recording_too_long": 422,
    "no_speech_detected": 422,
    "transcription_unavailable": 503,
}


def _error(exc: InterviewError) -> JSONResponse:
    return JSONResponse(status_code=_STATUS_BY_CODE.get(exc.code, 400), content={"error": exc.code})


def _unauthenticated() -> JSONResponse:
    return JSONResponse(status_code=401, content={"error": "authentication_required"})


def _feedback_outcome_response(outcome, *, ok_status: int) -> JSONResponse:
    attempt = outcome.attempt.model_dump(mode="json")
    if outcome.status == "feedback_error":
        return JSONResponse(
            status_code=503, content={"error": "feedback_service_unavailable", "attempt": attempt}
        )
    return JSONResponse(status_code=ok_status, content=attempt)


@router.get("/setup", response_model=InterviewSetupOut)
async def get_setup(request: Request, session: AsyncSession = Depends(get_session)):
    username = request.session.get("username")
    if not username:
        return _unauthenticated()
    try:
        return await request.app.state.interview_service.get_setup(session, username)
    except InterviewError as exc:
        return _error(exc)


@router.get("/sessions", response_model=list[SessionSummaryOut])
async def list_sessions(request: Request, session: AsyncSession = Depends(get_session)):
    username = request.session.get("username")
    if not username:
        return _unauthenticated()
    return await request.app.state.interview_service.list_sessions(session, username)


@router.post("/sessions")
async def create_session(
    req: CreateSessionRequest, request: Request, session: AsyncSession = Depends(get_session),
):
    username = request.session.get("username")
    if not username:
        return _unauthenticated()
    try:
        detail, created = await request.app.state.interview_service.create_session(
            session, username, req.role_id, req.practice_focus
        )
    except InterviewError as exc:
        return _error(exc)
    await session.commit()
    return JSONResponse(status_code=201 if created else 200, content=detail.model_dump(mode="json"))


@router.get("/sessions/{session_id}", response_model=SessionDetailOut)
async def get_session_detail(
    session_id: str, request: Request, session: AsyncSession = Depends(get_session),
):
    username = request.session.get("username")
    if not username:
        return _unauthenticated()
    try:
        return await request.app.state.interview_service.get_session(session, username, session_id)
    except InterviewError as exc:
        return _error(exc)


@router.post("/sessions/{session_id}/refresh")
async def refresh_session(
    session_id: str, request: Request, session: AsyncSession = Depends(get_session),
):
    username = request.session.get("username")
    if not username:
        return _unauthenticated()
    try:
        detail = await request.app.state.interview_service.refresh_session(session, username, session_id)
    except InterviewError as exc:
        return _error(exc)
    await session.commit()
    return JSONResponse(status_code=201, content=detail.model_dump(mode="json"))


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str, request: Request, session: AsyncSession = Depends(get_session),
):
    username = request.session.get("username")
    if not username:
        return _unauthenticated()
    try:
        await request.app.state.interview_service.delete_session(session, username, session_id)
    except InterviewError as exc:
        return _error(exc)
    await session.commit()
    return Response(status_code=204)


@router.post("/sessions/{session_id}/questions/{sequence_no}/attempts")
async def record_attempt(
    session_id: str,
    sequence_no: int,
    request: Request,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
):
    username = request.session.get("username")
    if not username:
        return _unauthenticated()

    max_bytes = get_settings().interview_max_audio_bytes
    content = await file.read(max_bytes + 1)
    if len(content) > max_bytes:
        return _error(InterviewError("recording_too_large"))

    audio = AudioInput(
        content=content, filename=file.filename or "", content_type=file.content_type or ""
    )
    try:
        outcome = await request.app.state.interview_service.process_recording(
            session, username, session_id, sequence_no, audio
        )
    except InterviewError as exc:
        return _error(exc)

    await session.commit()  # commits a preserved transcript even on a feedback_error outcome
    return _feedback_outcome_response(outcome, ok_status=200)


@router.post("/attempts/{response_id}/feedback")
async def retry_feedback(
    response_id: int, request: Request, session: AsyncSession = Depends(get_session),
):
    username = request.session.get("username")
    if not username:
        return _unauthenticated()
    try:
        outcome = await request.app.state.interview_service.retry_feedback(session, username, response_id)
    except InterviewError as exc:
        return _error(exc)
    await session.commit()
    return _feedback_outcome_response(outcome, ok_status=200)


@router.get("/areas", response_model=AreasOut)
async def get_areas(request: Request, session: AsyncSession = Depends(get_session)):
    username = request.session.get("username")
    if not username:
        return _unauthenticated()
    return await request.app.state.interview_service.get_areas(session, username)
