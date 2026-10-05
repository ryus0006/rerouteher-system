"""Account endpoints (E5). Session identity lives in a signed cookie, not the body."""
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.schemas.account import (
    AccountResponse,
    CreateAccountRequest,
    ProfileSkillMutationResponse,
    SavePlanRequest,
    SignInRequest,
    SignInResponse,
    StatusResponse,
)
from app.services.account import AccountError
from app.services.profile_skills import ProfileSkillError

router = APIRouter(prefix="/api/account", tags=["account"])

_STATUS = {"validation": 400, "conflict": 409, "auth": 401}


def _error(exc: AccountError) -> JSONResponse:
    return JSONResponse(status_code=_STATUS.get(exc.kind, 400), content={"error": exc.message})


_PROFILE_STATUS = {"journey": 409, "not_found": 404, "internal": 500}


def _profile_error(exc: ProfileSkillError) -> JSONResponse:
    return JSONResponse(
        status_code=_PROFILE_STATUS.get(exc.kind, 500),
        content={"error": exc.message},
    )


def _profile_response(result) -> ProfileSkillMutationResponse:
    return ProfileSkillMutationResponse(
        status=result.status,
        skill_id=result.skill_id,
        skill=result.skill,
        definition=result.definition,
        snapshot=result.snapshot,
        gap_result=result.gap_result,
        learned_skills=result.learned_skills,
    )


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


@router.put(
    "/professional-skills/{skill_id}",
    response_model=ProfileSkillMutationResponse,
)
async def add_professional_skill(
    skill_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    username = request.session.get("username")
    if not username:
        return JSONResponse(status_code=401, content={"error": "Not signed in."})
    try:
        result = await request.app.state.profile_skill_service.add_skill(
            session,
            username,
            skill_id,
        )
    except ProfileSkillError as exc:
        return _profile_error(exc)
    await session.commit()
    return _profile_response(result)


@router.delete(
    "/professional-skills/{skill_id}",
    response_model=ProfileSkillMutationResponse,
)
async def remove_professional_skill(
    skill_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    username = request.session.get("username")
    if not username:
        return JSONResponse(status_code=401, content={"error": "Not signed in."})
    try:
        result = await request.app.state.profile_skill_service.remove_skill(
            session,
            username,
            skill_id,
        )
    except ProfileSkillError as exc:
        return _profile_error(exc)
    await session.commit()
    return _profile_response(result)


@router.post("/sign-out", response_model=StatusResponse)
async def sign_out(request: Request):
    request.session.clear()
    return StatusResponse(status="signed_out")
