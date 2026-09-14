"""POST /api/learning/recommend."""
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.schemas.learning import LearningRequest, LearningResponse

router = APIRouter(prefix="/api/learning", tags=["learning"])


@router.post("/recommend", response_model=LearningResponse)
async def recommend_learning(
    req: LearningRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    service = request.app.state.learning_service
    return await service.recommend(req, session)
