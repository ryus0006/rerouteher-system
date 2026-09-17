"""Read/write for E8 companion conversation history. Writes flush; the router
commits at the request edge. Guest-usable (username nullable)."""
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class Turn:
    role: str
    content: str


async def load_recent(
    session: AsyncSession, session_id: str, username: str | None = None, limit: int = 10
) -> list[Turn]:
    """The newest `limit` turns, oldest-first for the model. For a signed-in mother load by
    username so her history follows her across sessions (US8.4.4); for a guest load by the
    browser session id."""
    if username:
        where, params = "username = :uname", {"uname": username, "lim": limit}
    else:
        where, params = "session_id = :sid", {"sid": session_id, "lim": limit}
    rows = (
        await session.execute(
            text(
                "SELECT role, content FROM ("
                "  SELECT role, content, created_at, id FROM companion_message "
                f"  WHERE {where} ORDER BY created_at DESC, id DESC LIMIT :lim"
                ") t ORDER BY t.created_at ASC, t.id ASC"
            ),
            params,
        )
    ).all()
    return [Turn(r.role, r.content) for r in rows]


async def save_turn(
    session: AsyncSession,
    session_id: str,
    username: str | None,
    role: str,
    content: str,
    tokens_in: int = 0,
    tokens_out: int = 0,
) -> None:
    await session.execute(
        text(
            "INSERT INTO companion_message "
            "(session_id, username, role, content, tokens_in, tokens_out) "
            "VALUES (:session_id, :username, :role, :content, :tokens_in, :tokens_out)"
        ),
        {
            "session_id": session_id,
            "username": username,
            "role": role,
            "content": content,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        },
    )
