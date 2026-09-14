"""SQL for rerouteher. Repositories run SQL; the router owns the transaction."""
import json
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class UserRow:
    username: str
    password_hash: str
    display_name: str


async def get_user(session: AsyncSession, username: str) -> UserRow | None:
    row = (
        await session.execute(
            text(
                "SELECT username, password_hash, display_name "
                "FROM rerouteher.app_user WHERE username = :u"
            ),
            {"u": username},
        )
    ).first()
    return UserRow(row.username, row.password_hash, row.display_name) if row else None


async def insert_user(
    session: AsyncSession, username: str, password_hash: str, display_name: str
) -> None:
    await session.execute(
        text(
            "INSERT INTO rerouteher.app_user (username, password_hash, display_name) "
            "VALUES (:u, :p, :d)"
        ),
        {"u": username, "p": password_hash, "d": display_name},
    )


async def touch_last_seen(session: AsyncSession, username: str) -> None:
    await session.execute(
        text("UPDATE rerouteher.app_user SET last_seen_at = now() WHERE username = :u"),
        {"u": username},
    )


async def get_plan(session: AsyncSession, username: str) -> dict | None:
    row = (
        await session.execute(
            text("SELECT plan_json FROM rerouteher.saved_journey WHERE username = :u"),
            {"u": username},
        )
    ).first()
    if row is None:
        return None
    value = row.plan_json
    # asyncpg may hand back jsonb as a str; normalise to a dict.
    return json.loads(value) if isinstance(value, str) else value


async def upsert_plan(session: AsyncSession, username: str, plan: dict) -> None:
    await session.execute(
        text(
            "INSERT INTO rerouteher.saved_journey (username, plan_json, updated_at) "
            "VALUES (:u, cast(:p as jsonb), now()) "
            "ON CONFLICT (username) DO UPDATE "
            "SET plan_json = excluded.plan_json, updated_at = now()"
        ),
        {"u": username, "p": json.dumps(plan)},
    )
