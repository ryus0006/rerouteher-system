"""POST /api/employers/match."""
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.schemas.employers import EmployerMatchRequest, EmployerMatchResponse

router = APIRouter(prefix="/api/employers", tags=["employers"])


@router.post("/match", response_model=EmployerMatchResponse)
async def match_employers(
    req: EmployerMatchRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    service = request.app.state.employer_service
    return await service.match(req, session)
