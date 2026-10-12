from datetime import datetime, timezone

import pytest

from app.repositories import jobs as jobs_repo

pytestmark = pytest.mark.asyncio


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return self._rows


class FakeRow:
    def __init__(self, **values):
        self.__dict__.update(values)


class FakeSession:
    def __init__(self, results=None):
        self._results = iter(results or [])
        self.executed = []

    async def execute(self, statement, params=None):
        self.executed.append((str(statement), params))
        return next(self._results, FakeResult([]))


def _searched_at():
    return datetime(2026, 10, 4, 10, 0, tzinfo=timezone.utc)


async def test_get_role_search_maps_role_and_clean_search_title():
    session = FakeSession(
        [
            FakeResult(
                [
                    FakeRow(
                        role_id="R03",
                        role_title="Human Resources Officer Grade N41",
                        search_title="Human Resources Officer",
                    )
                ]
            )
        ]
    )

    result = await jobs_repo.get_role_search(session, "R03")

    assert result.role_id == "R03"
    assert result.role_title == "Human Resources Officer Grade N41"
    assert result.search_title == "Human Resources Officer"


async def test_get_search_record_maps_status_and_provider_trace():
    searched_at = _searched_at()
    session = FakeSession(
        [
            FakeResult(
                [
                    FakeRow(
                        role_id="R03",
                        search_title="Human Resources Officer",
                        status="ready",
                        searched_at=searched_at,
                        providers_attempted=["jooble"],
                    )
                ]
            )
        ]
    )

    result = await jobs_repo.get_search_record(session, "R03")

    assert result.status == "ready"
    assert result.searched_at == searched_at
    assert result.providers_attempted == ["jooble"]


async def test_get_active_openings_returns_latest_opening_per_employer():
    first_seen = _searched_at()
    session = FakeSession(
        [
            FakeResult(
                [
                    FakeRow(
                        role_id="R03",
                        employer_id="1023",
                        title="Human Resources Officer",
                        canonical_url="https://jobs.example.test/old",
                        location="Malaysia",
                        provider="jooble",
                        last_seen_at=first_seen,
                        status="active",
                    ),
                    FakeRow(
                        role_id="R03",
                        employer_id="1023",
                        title="Senior Human Resources Officer",
                        canonical_url="https://jobs.example.test/latest",
                        location="Malaysia",
                        provider="foundit",
                        last_seen_at=_searched_at().replace(hour=11),
                        status="active",
                    ),
                    FakeRow(
                        role_id="R03",
                        employer_id="4707",
                        title="People Officer",
                        canonical_url="https://jobs.example.test/nestle",
                        location="Malaysia",
                        provider="tavily",
                        last_seen_at=first_seen,
                        status="active",
                    ),
                ]
            )
        ]
    )

    result = await jobs_repo.get_active_openings_by_employer(session, "R03")

    assert result["1023"].title == "Senior Human Resources Officer"
    assert result["4707"].title == "People Officer"


async def test_replace_role_results_writes_ready_state_and_opening():
    from app.repositories.jobs import RoleSearch, StoredJobOpening

    searched_at = _searched_at()
    role = RoleSearch(
        role_id="R03",
        role_title="Human Resources Officer",
        search_title="Human Resources Officer",
    )
    opening = StoredJobOpening(
        role_id="R03",
        employer_id="1023",
        title="Human Resources Officer",
        canonical_url="https://jobs.example.test/cimb",
        location="Malaysia",
        provider="jooble",
        last_seen_at=searched_at,
        status="active",
    )
    session = FakeSession()

    result = await jobs_repo.replace_role_results(
        session,
        role,
        [opening],
        ["jooble"],
        searched_at,
    )

    assert result.status == "ready"
    assert result.searched_at == searched_at
    assert result.providers_attempted == ["jooble"]
    assert len(session.executed) >= 2


async def test_record_unavailable_if_absent_does_not_overwrite_existing_cache():
    searched_at = _searched_at()
    session = FakeSession(
        [
            FakeResult(
                [
                    FakeRow(
                        role_id="R03",
                        search_title="Human Resources Officer",
                        status="ready",
                        searched_at=searched_at,
                        providers_attempted=["jooble"],
                    )
                ]
            )
        ]
    )

    result = await jobs_repo.record_unavailable_if_absent(
        session,
        "R03",
        "Human Resources Officer",
        ["jooble", "foundit", "tavily"],
        searched_at,
    )

    assert result.status == "ready"
    assert result.searched_at == searched_at
    assert len(session.executed) == 1
