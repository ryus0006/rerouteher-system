"""Async database engine and session.

Reference tables live in the `rerouteher` schema; embedding columns are pgvector
`vector(384)`. Requests write nothing.
"""
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings

_settings = get_settings()

engine = create_async_engine(
    _settings.database_url,
    pool_pre_ping=True,
    # ensure every connection resolves unqualified names against the project schema
    connect_args={"server_settings": {"search_path": "rerouteher,public"}},
)

SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with SessionLocal() as session:
        yield session
