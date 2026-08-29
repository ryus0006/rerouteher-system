"""POST /api/cv/parse."""
from fastapi import APIRouter, File, Request, UploadFile
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.schemas.cv import CVParseResponse
from app.services.cv_extractor import UnreadableCVError

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
        cv = extractor.parse(data)
    except UnreadableCVError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    return CVParseResponse(cv=cv)
