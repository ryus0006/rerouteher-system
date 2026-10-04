"""Cache-first orchestration for role-specific tracked-employer openings."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from app.repositories import employers as employers_repo
from app.repositories import jobs as jobs_repo
from app.repositories.jobs import JobSearchRecord, RoleSearch, StoredJobOpening
from app.services.job_sources import JobCandidate, JobSourceUnavailable

logger = logging.getLogger("rerouteher")


@dataclass(frozen=True)
class JobSearchOutcome:
    status: str
    searched_at: datetime
    openings: dict[str, StoredJobOpening]


def _company_key(value: str) -> str:
    return value.strip().casefold()


def _title_key(value: str) -> str:
    return " ".join(value.strip().casefold().split())


# Legal/structure words dropped before comparing company names. Job boards give
# short trade names ("CIMB"); our data holds full Bursa names ("CIMB Group
# Holdings Berhad"). Stripping these leaves the distinctive core for matching.
_LEGAL_WORDS = frozenset({
    "berhad", "bhd", "sdn", "limited", "ltd", "group", "holdings",
    "corporation", "corp", "co", "company", "the", "malaysia", "m", "inc", "plc",
})


def _core_tokens(value: str) -> tuple[str, ...]:
    cleaned = re.sub(r"[^a-z0-9]+", " ", (value or "").casefold())
    return tuple(t for t in cleaned.split() if t and t not in _LEGAL_WORDS)


def _is_contiguous_sublist(short: tuple[str, ...], long: tuple[str, ...]) -> bool:
    n = len(short)
    if n == 0 or n > len(long):
        return False
    return any(long[i : i + n] == short for i in range(len(long) - n + 1))


def _company_matches(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    # Whole-word containment on the core tokens: one name's tokens appear as a
    # contiguous run of whole words in the other. This matches "CIMB" to "CIMB
    # Group Holdings Berhad" while rejecting mid-word hits like "Axi"/"Axis".
    if not a or not b:
        return False
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    return _is_contiguous_sublist(short, long)


class JobSearchService:
    def __init__(
        self,
        jobs_repo=jobs_repo,
        employer_repo=employers_repo,
        sources=None,
        location: str = "Malaysia",
        clock=None,
    ) -> None:
        self._jobs_repo = jobs_repo
        self._employer_repo = employer_repo
        self._sources = list(sources or [])
        self._location = location
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    async def get_or_search(self, session, role_id: str) -> JobSearchOutcome:
        role = await self._load_role(session, role_id)
        record = await self._jobs_repo.get_search_record(session, role_id)
        if record is not None:
            return await self._cached_outcome(session, record)
        return await self._run_search(session, role)

    async def refresh(self, session, role_id: str) -> JobSearchOutcome:
        role = await self._load_role(session, role_id)
        return await self._run_search(session, role)

    async def _load_role(self, session, role_id: str) -> RoleSearch:
        role = await self._jobs_repo.get_role_search(session, role_id)
        if role is None:
            raise LookupError(f"role not found: {role_id}")
        return role

    async def _cached_outcome(
        self, session, record: JobSearchRecord
    ) -> JobSearchOutcome:
        active = {}
        if record.status == "ready":
            active = await self._jobs_repo.get_active_openings_by_employer(
                session, record.role_id
            )
        return JobSearchOutcome(
            status=record.status,
            searched_at=record.searched_at,
            openings=active,
        )

    async def _tracked_employers(
        self, session
    ) -> list[tuple[tuple[str, ...], str]]:
        employers = await self._employer_repo.get_employers_with_evidence(session)
        return [(_core_tokens(employer.name), str(employer.id)) for employer in employers]

    @staticmethod
    def _match_employer(
        company_name: str, tracked: list[tuple[tuple[str, ...], str]]
    ) -> str | None:
        tokens = _core_tokens(company_name)
        if not tokens:
            return None
        for emp_tokens, employer_id in tracked:
            if _company_matches(tokens, emp_tokens):
                return employer_id
        return None

    async def _run_search(
        self, session, role: RoleSearch
    ) -> JobSearchOutcome:
        tracked = await self._tracked_employers(session)
        provider_names: list[str] = []
        successful_provider = False
        candidates_by_key: dict[tuple[str, str], JobCandidate] = {}
        employer_by_key: dict[tuple[str, str], str] = {}

        for provider_name, source in self._sources:
            if source is None:
                continue
            provider_names.append(provider_name)
            try:
                candidates = await source.search(
                    role.search_title,
                    self._location,
                    20 if provider_name == "tavily" else 50,
                )
                successful_provider = True
            except JobSourceUnavailable as exc:
                logger.warning(
                    "job provider unavailable | provider=%s | error=%s",
                    provider_name,
                    type(exc).__name__,
                )
                continue
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "job provider failed | provider=%s | error=%s",
                    provider_name,
                    type(exc).__name__,
                )
                continue

            for candidate in candidates:
                if not candidate.title.strip() or not candidate.url.strip():
                    continue
                employer_id = self._match_employer(candidate.company_name, tracked)
                if employer_id is None:
                    continue
                key = (_title_key(candidate.title), _company_key(candidate.company_name))
                previous = candidates_by_key.get(key)
                if previous is None or (
                    candidate.provider == "foundit"
                    and previous.provider != "foundit"
                ):
                    candidates_by_key[key] = candidate
                    employer_by_key[key] = employer_id

            if candidates_by_key:
                break

        searched_at = self._clock()
        if not successful_provider:
            await self._jobs_repo.record_unavailable_if_absent(
                session,
                role.role_id,
                role.search_title,
                provider_names,
                searched_at,
            )
            return JobSearchOutcome(
                status="temporarily_unavailable",
                searched_at=searched_at,
                openings={},
            )

        openings = [
            StoredJobOpening(
                role_id=role.role_id,
                employer_id=employer_by_key[key],
                title=candidate.title.strip(),
                canonical_url=candidate.url.strip(),
                location=candidate.location,
                provider=candidate.provider,
                last_seen_at=searched_at,
                status="active",
            )
            for key, candidate in candidates_by_key.items()
        ]
        record = await self._jobs_repo.replace_role_results(
            session,
            role,
            openings,
            provider_names,
            searched_at,
        )
        active = await self._jobs_repo.get_active_openings_by_employer(
            session, role.role_id
        )
        if not active:
            active = {opening.employer_id: opening for opening in openings}
        return JobSearchOutcome(
            status=record.status,
            searched_at=record.searched_at,
            openings=active,
        )
