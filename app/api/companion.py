"""POST /api/companion/ask (E8). Guest-allowed; username read from the E5 session
cookie when present. Commits history writes at the request edge."""
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.schemas.companion import AskRequest, AskResponse

router = APIRouter(prefix="/api/companion", tags=["companion"])


@router.post("/ask", response_model=AskResponse, response_model_by_alias=True)
async def ask_companion(
    req: AskRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    # Guest-allowed: username only when the E5 session cookie is present. Read via
    # scope so it works whether or not SessionMiddleware is installed (tests).
    username = (request.scope.get("session") or {}).get("username")
    service = request.app.state.companion_service
    result = await service.ask(req, session, username)
    await session.commit()
    return result
