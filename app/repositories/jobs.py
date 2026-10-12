"""Persistence helpers for role-level job-search state and openings."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class RoleSearch:
    role_id: str
    role_title: str
    search_title: str


@dataclass(frozen=True)
class JobSearchRecord:
    role_id: str
    search_title: str
    status: str
    searched_at: datetime
    providers_attempted: list[str]


@dataclass(frozen=True)
class StoredJobOpening:
    role_id: str
    employer_id: str
    title: str
    canonical_url: str
    location: str | None
    provider: str
    last_seen_at: datetime
    status: str


def _search_record(row) -> JobSearchRecord:
    return JobSearchRecord(
        role_id=str(row.role_id),
        search_title=row.search_title,
        status=row.status,
        searched_at=row.searched_at,
        providers_attempted=list(row.providers_attempted or []),
    )


def _opening(row) -> StoredJobOpening:
    return StoredJobOpening(
        role_id=str(row.role_id),
        employer_id=str(row.employer_id),
        title=row.title,
        canonical_url=row.canonical_url,
        location=row.location,
        provider=row.provider,
        last_seen_at=row.last_seen_at,
        status=row.status,
    )


async def get_role_search(session: AsyncSession, role_id: str) -> RoleSearch | None:
    row = (
        await session.execute(
            text(
                "SELECT role_id, role_title, search_title "
                "FROM roles WHERE role_id = :role_id"
            ),
            {"role_id": role_id},
        )
    ).first()
    if row is None:
        return None
    return RoleSearch(
        role_id=str(row.role_id),
        role_title=row.role_title,
        search_title=row.search_title or row.role_title,
    )


async def get_search_record(
    session: AsyncSession, role_id: str
) -> JobSearchRecord | None:
    row = (
        await session.execute(
            text(
                "SELECT role_id, search_title, status, searched_at, providers_attempted "
                "FROM job_search WHERE role_id = :role_id"
            ),
            {"role_id": role_id},
        )
    ).first()
    return _search_record(row) if row is not None else None


async def get_active_openings_by_employer(
    session: AsyncSession, role_id: str
) -> dict[str, StoredJobOpening]:
    rows = (
        await session.execute(
            text(
                "SELECT role_id, employer_id, title, canonical_url, location, provider, "
                "last_seen_at, status "
                "FROM job_opening "
                "WHERE role_id = :role_id AND status = 'active' "
                "ORDER BY employer_id, last_seen_at DESC, job_opening_id DESC"
            ),
            {"role_id": role_id},
        )
    ).all()
    openings: dict[str, StoredJobOpening] = {}
    for row in rows:
        opening = _opening(row)
        previous = openings.get(opening.employer_id)
        if previous is None or opening.last_seen_at > previous.last_seen_at:
            openings[opening.employer_id] = opening
    return openings


async def replace_role_results(
    session: AsyncSession,
    role: RoleSearch,
    openings: list[StoredJobOpening],
    providers_attempted: list[str],
    searched_at: datetime,
) -> JobSearchRecord:
    canonical_urls = [opening.canonical_url for opening in openings]
    if canonical_urls:
        await session.execute(
            text(
                "UPDATE job_opening SET status = 'inactive' "
                "WHERE role_id = :role_id AND status = 'active' "
                "AND NOT (canonical_url = ANY(:canonical_urls))"
            ),
            {"role_id": role.role_id, "canonical_urls": canonical_urls},
        )
    else:
        await session.execute(
            text(
                "UPDATE job_opening SET status = 'inactive' "
                "WHERE role_id = :role_id AND status = 'active'"
            ),
            {"role_id": role.role_id},
        )

    for opening in openings:
        await session.execute(
            text(
                "INSERT INTO job_opening "
                "(role_id, employer_id, canonical_url, title, location, provider, "
                "query_text, retrieved_at, first_seen_at, last_seen_at, status) "
                "VALUES (:role_id, :employer_id, :canonical_url, :title, :location, "
                ":provider, :query_text, :searched_at, :searched_at, :searched_at, 'active') "
                "ON CONFLICT (role_id, employer_id, canonical_url) DO UPDATE SET "
                "title = EXCLUDED.title, location = EXCLUDED.location, "
                "provider = EXCLUDED.provider, query_text = EXCLUDED.query_text, "
                "retrieved_at = EXCLUDED.retrieved_at, last_seen_at = EXCLUDED.last_seen_at, "
                "status = 'active'"
            ),
            {
                "role_id": role.role_id,
                "employer_id": opening.employer_id,
                "canonical_url": opening.canonical_url,
                "title": opening.title,
                "location": opening.location,
                "provider": opening.provider,
                "query_text": f"{role.search_title} Malaysia",
                "searched_at": searched_at,
            },
        )

    status = "ready" if openings else "empty"
    await session.execute(
        text(
            "INSERT INTO job_search "
            "(role_id, search_title, status, searched_at, providers_attempted) "
            "VALUES (:role_id, :search_title, :status, :searched_at, :providers_attempted) "
            "ON CONFLICT (role_id) DO UPDATE SET "
            "search_title = EXCLUDED.search_title, status = EXCLUDED.status, "
            "searched_at = EXCLUDED.searched_at, "
            "providers_attempted = EXCLUDED.providers_attempted"
        ),
        {
            "role_id": role.role_id,
            "search_title": role.search_title,
            "status": status,
            "searched_at": searched_at,
            "providers_attempted": providers_attempted,
        },
    )
    return JobSearchRecord(
        role_id=role.role_id,
        search_title=role.search_title,
        status=status,
        searched_at=searched_at,
        providers_attempted=providers_attempted,
    )


async def record_unavailable_if_absent(
    session: AsyncSession,
    role_id: str,
    search_title: str,
    providers_attempted: list[str],
    searched_at: datetime,
) -> JobSearchRecord:
    existing = await get_search_record(session, role_id)
    if existing is not None:
        return existing

    await session.execute(
        text(
            "INSERT INTO job_search "
            "(role_id, search_title, status, searched_at, providers_attempted) "
            "VALUES (:role_id, :search_title, 'temporarily_unavailable', "
            ":searched_at, :providers_attempted)"
        ),
        {
            "role_id": role_id,
            "search_title": search_title,
            "searched_at": searched_at,
            "providers_attempted": providers_attempted,
        },
    )
    return JobSearchRecord(
        role_id=role_id,
        search_title=search_title,
        status="temporarily_unavailable",
        searched_at=searched_at,
        providers_attempted=providers_attempted,
    )
