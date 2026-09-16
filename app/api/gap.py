"""POST /api/gap/compute."""
from fastapi import APIRouter, BackgroundTasks, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.schemas.gap import GapRequest, GapResponse
from app.services.learning_fill import focus_area_skill_ids

router = APIRouter(prefix="/api/gap", tags=["gap"])


@router.post("/compute", response_model=GapResponse)
async def compute_gap(
    req: GapRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    service = request.app.state.gap_service
    resp = await service.compute(req, session)
    # Fill only the gap skills the learning page actually surfaces (a gap can return
    # 100+ skills; she sees a handful), in the background, so her learning page shows
    # a real course by the time she opens it. Best effort: a missing/disabled service
    # or a failed fill never affects this response.
    fill = getattr(request.app.state, "learning_fill_service", None)
    if fill is not None and fill.enabled and resp.gaps:
        skill_ids = focus_area_skill_ids(resp.gaps)
        if skill_ids:
            background_tasks.add_task(fill.fill_for_skills, skill_ids)
    return resp
