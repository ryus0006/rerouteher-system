"""Account logic (E5). Pure: no FastAPI. The router maps AccountError to HTTP."""
import re

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password, verify_password
from app.repositories import accounts as accounts_repo

MIN_PASSWORD_LENGTH = 8
MAX_DISPLAY_NAME_LENGTH = 40
_USERNAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9_]{1,28}[a-z0-9])?$")


class AccountError(Exception):
    """User-safe message plus a kind the router maps to a status code."""

    def __init__(self, message: str, kind: str) -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind  # "validation" -> 400, "conflict" -> 409, "auth" -> 401


def _normalise_username(username: str) -> str:
    return username.strip().lower()


def _validate_username(username: str) -> None:
    if len(username) < 3:
        raise AccountError("Use at least 3 characters.", "validation")
    if len(username) > 30:
        raise AccountError("Use 30 characters or fewer.", "validation")
    if not _USERNAME_RE.match(username):
        raise AccountError(
            "Use letters, numbers and underscores, "
            "starting and ending with a letter or number.",
            "validation",
        )


def _validate_password(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise AccountError(f"Use at least {MIN_PASSWORD_LENGTH} characters.", "validation")


def _resolve_display_name(display_name: str | None, username: str) -> str:
    name = (display_name or "").strip()
    if len(name) > MAX_DISPLAY_NAME_LENGTH:
        raise AccountError(f"Use {MAX_DISPLAY_NAME_LENGTH} characters or fewer.", "validation")
    return name or username


class AccountService:
    async def create(
        self, session: AsyncSession, *, username: str, password: str,
        display_name: str | None, plan: dict,
    ) -> dict:
        uname = _normalise_username(username)
        _validate_username(uname)
        _validate_password(password)
        display = _resolve_display_name(display_name, uname)

        if await accounts_repo.get_user(session, uname) is not None:
            raise AccountError("That username is taken.", "conflict")

        await accounts_repo.insert_user(session, uname, hash_password(password), display)
        await accounts_repo.upsert_plan(session, uname, plan or {})
        return {"username": uname, "display_name": display}

    async def sign_in(self, session: AsyncSession, *, username: str, password: str) -> dict:
        uname = _normalise_username(username)
        user = await accounts_repo.get_user(session, uname)
        if user is None or not verify_password(password, user.password_hash):
            raise AccountError("Wrong username or password.", "auth")
        await accounts_repo.touch_last_seen(session, uname)
        plan = await accounts_repo.get_plan(session, uname)
        return {"username": user.username, "display_name": user.display_name, "plan": plan}

    async def save_plan(self, session: AsyncSession, *, username: str, plan: dict) -> None:
        await accounts_repo.upsert_plan(session, username, plan or {})
