"""CV parsing and authenticated refreshed-CV endpoints."""
import logging

import anyio
from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import get_session
from app.schemas.cv import (
    CVParseResponse,
    CvGenerateRequest,
    CvGenerateResponse,
    CvImproveRequest,
    CvImproveResponse,
)
from app.services.cv_extractor import UnreadableCVError
from app.services.cv_generation import CvGenerationError

logger = logging.getLogger("rerouteher")

router = APIRouter(prefix="/api/cv", tags=["cv"])

_STATUS_BY_CODE = {
    "journey_prerequisite_incomplete": 409,
    "invalid_cv_setup": 422,
    "invalid_cv_content": 422,
    "cv_generation_unavailable": 503,
}


def _error(exc: CvGenerationError) -> JSONResponse:
    return JSONResponse(
        status_code=_STATUS_BY_CODE.get(exc.code, 400),
        content={"error": exc.code},
    )


def _unauthenticated() -> JSONResponse:
    return JSONResponse(status_code=401, content={"error": "authentication_required"})


@router.post("/parse", response_model=CVParseResponse, responses={400: {"model": dict}})
async def parse_cv(request: Request, file: UploadFile = File(...)):
    settings = get_settings()

    if file.content_type != "application/pdf":
        return JSONResponse(status_code=400, content={"error": "PDF only"})

    data = await file.read()
    if len(data) > settings.max_cv_bytes:
        return JSONResponse(status_code=400, content={"error": "file too large (max 10MB)"})

    extractor = request.app.state.cv_extractor
    try:
        # PDF parsing + NLP is CPU-bound; run it off the event loop.
        cv = await anyio.to_thread.run_sync(extractor.parse, data)
    except UnreadableCVError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    # Diagnostic: how the parser segmented the CV (title mislabels show up here). Logs the
    # experience structure only - titles/orgs/dates + lengths - not the full CV narrative.
    logger.info("cv parsed: experiences=%d skill_mentions=%d", len(cv.experiences), len(cv.skill_mentions))
    for i, exp in enumerate(cv.experiences):
        logger.info(
            "cv experience[%d]: title=%r org=%r start=%r end=%r desc_len=%d",
            i, exp.title, exp.organisation, exp.start, exp.end, len(exp.description or ""),
        )

    return CVParseResponse(cv=cv)


@router.post(
    "/generate",
    response_model=CvGenerateResponse,
    response_model_by_alias=True,
)
async def generate_cv(
    req: CvGenerateRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    username = request.session.get("username")
    if not username:
        return _unauthenticated()

    try:
        result = await request.app.state.cv_generation_service.generate(
            session,
            username,
            role_id=req.role_id,
            regenerate=req.regenerate,
        )
    except CvGenerationError as exc:
        return _error(exc)

    await session.commit()
    return CvGenerateResponse(
        role_id=result.role_id,
        generation_status=result.status,
        generated_at=result.generated_at,
        draft=result.draft,
    )


@router.post(
    "/improve",
    response_model=CvImproveResponse,
    response_model_by_alias=True,
)
async def improve_cv(
    req: CvImproveRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    username = request.session.get("username")
    if not username:
        return _unauthenticated()

    try:
        result = await request.app.state.cv_generation_service.improve(
            session,
            username,
            role_id=req.role_id,
            section=req.section,
            experience_index=req.experience_index,
            current_text=req.current_text,
            previous_suggestions=req.previous_suggestions,
        )
    except CvGenerationError as exc:
        return _error(exc)

    return CvImproveResponse(
        section=result.section,
        experience_index=result.experience_index,
        suggestion=result.suggestion,
        evidence=result.evidence,
    )
