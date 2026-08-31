"""POST /api/cv/parse."""
import logging

import anyio
from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.schemas.cv import CVParseResponse
from app.services.cv_extractor import UnreadableCVError

logger = logging.getLogger("rerouteher")

router = APIRouter(prefix="/api/cv", tags=["cv"])


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
