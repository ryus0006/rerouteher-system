"""POST /api/gap/compute."""
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.schemas.gap import GapRequest, GapResponse

router = APIRouter(prefix="/api/gap", tags=["gap"])


@router.post("/compute", response_model=GapResponse)
async def compute_gap(
    req: GapRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    service = request.app.state.gap_service
    return await service.compute(req, session)
