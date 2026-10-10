"""Employer-fit matching and public role-opening refresh endpoints."""
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.schemas.employers import (
    EmployerMatchRequest,
    EmployerMatchResponse,
    JobRefreshResponse,
)

router = APIRouter(prefix="/api/employers", tags=["employers"])


@router.post("/match", response_model=EmployerMatchResponse)
async def match_employers(
    req: EmployerMatchRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    service = request.app.state.employer_service
    try:
        result = await service.match(req, session)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="target role not found") from exc
    await session.commit()
    return result


@router.post("/jobs/refresh/{role_id}", response_model=JobRefreshResponse)
async def refresh_jobs(
    role_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    service = request.app.state.employer_service
    try:
        result = await service.refresh(role_id, session)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="target role not found") from exc
    await session.commit()
    if result.status == "temporarily_unavailable":
        return JSONResponse(
            status_code=503,
            content=jsonable_encoder(result),
        )
    return result
