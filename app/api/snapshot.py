"""POST /api/snapshot/generate."""
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.schemas.snapshot import SnapshotRequest, SnapshotResponse

router = APIRouter(prefix="/api/snapshot", tags=["snapshot"])


@router.post("/generate", response_model=SnapshotResponse)
async def generate_snapshot(
    req: SnapshotRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    service = request.app.state.snapshot_service
    return await service.generate(req, session)
