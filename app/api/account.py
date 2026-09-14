"""Account endpoints (E5). Session identity lives in a signed cookie, not the body."""
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.schemas.account import (
    AccountResponse,
    CreateAccountRequest,
    SavePlanRequest,
    SignInRequest,
    SignInResponse,
    StatusResponse,
)
from app.services.account import AccountError

router = APIRouter(prefix="/api/account", tags=["account"])

_STATUS = {"validation": 400, "conflict": 409, "auth": 401}


def _error(exc: AccountError) -> JSONResponse:
    return JSONResponse(status_code=_STATUS.get(exc.kind, 400), content={"error": exc.message})


@router.post("/create", response_model=AccountResponse)
async def create_account(
    req: CreateAccountRequest, request: Request,
    session: AsyncSession = Depends(get_session),
):
    try:
        result = await request.app.state.account_service.create(
            session, username=req.username, password=req.password,
            display_name=req.display_name, plan=req.plan,
        )
    except AccountError as exc:
        return _error(exc)
    await session.commit()
    request.session["username"] = result["username"]
    return AccountResponse(**result)


@router.post("/sign-in", response_model=SignInResponse)
async def sign_in(
    req: SignInRequest, request: Request,
    session: AsyncSession = Depends(get_session),
):
    try:
        result = await request.app.state.account_service.sign_in(
            session, username=req.username, password=req.password
        )
    except AccountError as exc:
        return _error(exc)
    await session.commit()
    request.session["username"] = result["username"]
    return SignInResponse(**result)


@router.post("/plan", response_model=StatusResponse)
async def save_plan(
    req: SavePlanRequest, request: Request,
    session: AsyncSession = Depends(get_session),
):
    username = request.session.get("username")
    if not username:
        return JSONResponse(status_code=401, content={"error": "Not signed in."})
    await request.app.state.account_service.save_plan(session, username=username, plan=req.plan)
    await session.commit()
    return StatusResponse(status="saved")


@router.post("/sign-out", response_model=StatusResponse)
async def sign_out(request: Request):
    request.session.clear()
    return StatusResponse(status="signed_out")
