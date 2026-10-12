from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.repositories.jobs import JobSearchRecord, RoleSearch, StoredJobOpening
from app.services.job_search import JobSearchService, _company_matches, _core_tokens
from app.services.job_sources import JobCandidate, JobSourceUnavailable

pytestmark = pytest.mark.asyncio


SEARCHED_AT = datetime(2026, 10, 4, 10, 0, tzinfo=timezone.utc)


class FakeJobsRepo:
    def __init__(self, record=None):
        self.role = RoleSearch("R03", "Human Resources Officer", "Human Resources Officer")
        self.record = record
        self.active = {}
        self.replacements = []
        self.unavailable_calls = []

    async def get_role_search(self, session, role_id):
        return self.role if role_id == self.role.role_id else None

    async def get_search_record(self, session, role_id):
        return self.record

    async def get_active_openings_by_employer(self, session, role_id):
        return self.active

    async def replace_role_results(
        self, session, role, openings, providers_attempted, searched_at
    ):
        self.replacements.append((role, openings, providers_attempted, searched_at))
        self.record = JobSearchRecord(
            role_id=role.role_id,
            search_title=role.search_title,
            status="ready" if openings else "empty",
            searched_at=searched_at,
            providers_attempted=providers_attempted,
        )
        return self.record

    async def record_unavailable_if_absent(
        self, session, role_id, search_title, providers_attempted, searched_at
    ):
        self.unavailable_calls.append(
            (role_id, search_title, providers_attempted, searched_at)
        )
        if self.record is not None:
            return self.record
        self.record = JobSearchRecord(
            role_id=role_id,
            search_title=search_title,
            status="temporarily_unavailable",
            searched_at=searched_at,
            providers_attempted=providers_attempted,
        )
        return self.record


class FakeEmployerRepo:
    async def get_employers_with_evidence(self, session):
        return [
            SimpleNamespace(id="1023", name="CIMB Group Holdings Berhad"),
            SimpleNamespace(id="4707", name="Nestle (Malaysia) Berhad"),
        ]


class FakeSource:
    def __init__(self, result=None, error=None):
        self.result = result or []
        self.error = error
        self.calls = []

    async def search(self, title, location, limit):
        self.calls.append((title, location, limit))
        if self.error:
            raise self.error
        return self.result


def _service(jobs, sources):
    return JobSearchService(
        jobs_repo=jobs,
        employer_repo=FakeEmployerRepo(),
        sources=sources,
        location="Malaysia",
        clock=lambda: SEARCHED_AT,
    )


async def test_get_or_search_uses_ready_cache_without_calling_sources():
    opening = StoredJobOpening(
        "R03",
        "1023",
        "HR Officer",
        "https://jobs.example.test/hr",
        "Malaysia",
        "jooble",
        SEARCHED_AT,
        "active",
    )
    jobs = FakeJobsRepo(
        JobSearchRecord("R03", "Human Resources Officer", "ready", SEARCHED_AT, ["jooble"])
    )
    jobs.active = {"1023": opening}
    source = FakeSource(error=AssertionError("cached search must not call providers"))

    result = await _service(jobs, [("jooble", source)]).get_or_search(object(), "R03")

    assert result.status == "ready"
    assert result.openings["1023"] == opening
    assert source.calls == []


async def test_missing_cache_falls_back_from_jooble_to_foundit_for_tracked_employer():
    jooble = FakeSource(
        [
            JobCandidate(
                "HR Officer",
                "Untracked Employer",
                "https://jobs.example.test/untracked",
                "Malaysia",
                "jooble",
            )
        ]
    )
    foundit = FakeSource(
        [
            JobCandidate(
                "Human Resources Officer",
                " CIMB Group Holdings Berhad ",
                "https://careers.example.test/cimb",
                "Kuala Lumpur",
                "foundit",
            )
        ]
    )
    jobs = FakeJobsRepo()

    result = await _service(
        jobs, [("jooble", jooble), ("foundit", foundit)]
    ).get_or_search(object(), "R03")

    assert result.status == "ready"
    assert result.openings["1023"].canonical_url == "https://careers.example.test/cimb"
    assert jooble.calls == [("Human Resources Officer", "Malaysia", 50)]
    assert foundit.calls == [("Human Resources Officer", "Malaysia", 50)]
    assert jobs.replacements[0][2] == ["jooble", "foundit"]


async def test_successful_search_without_tracked_opening_is_empty():
    jooble = FakeSource(
        [
            JobCandidate(
                "HR Officer",
                "Untracked Employer",
                "https://jobs.example.test/untracked",
                "Malaysia",
                "jooble",
            )
        ]
    )
    foundit = FakeSource(result=[])
    jobs = FakeJobsRepo()

    result = await _service(
        jobs, [("jooble", jooble), ("foundit", foundit)]
    ).get_or_search(object(), "R03")

    assert result.status == "empty"
    assert result.openings == {}
    assert jobs.replacements[0][1] == []


async def test_all_provider_failures_persist_unavailable_state():
    jobs = FakeJobsRepo()
    sources = [
        ("jooble", FakeSource(error=JobSourceUnavailable("jooble failed"))),
        ("foundit", FakeSource(error=JobSourceUnavailable("foundit failed"))),
    ]

    result = await _service(jobs, sources).get_or_search(object(), "R03")

    assert result.status == "temporarily_unavailable"
    assert result.openings == {}
    assert jobs.replacements == []
    assert jobs.unavailable_calls[0][2] == ["jooble", "foundit"]


async def test_cached_unavailable_state_does_not_retry_sources():
    jobs = FakeJobsRepo(
        JobSearchRecord(
            "R03",
            "Human Resources Officer",
            "temporarily_unavailable",
            SEARCHED_AT,
            ["jooble", "foundit"],
        )
    )
    source = FakeSource(error=AssertionError("cached unavailable must not retry"))

    result = await _service(jobs, [("jooble", source)]).get_or_search(object(), "R03")

    assert result.status == "temporarily_unavailable"
    assert source.calls == []


async def test_cached_empty_state_does_not_retry_sources():
    jobs = FakeJobsRepo(
        JobSearchRecord(
            "R03",
            "Human Resources Officer",
            "empty",
            SEARCHED_AT,
            ["jooble", "foundit"],
        )
    )
    source = FakeSource(error=AssertionError("cached empty must not retry"))

    result = await _service(jobs, [("jooble", source)]).get_or_search(object(), "R03")

    assert result.status == "empty"
    assert result.openings == {}
    assert source.calls == []


async def test_duplicate_title_and_company_prefers_foundit_candidate():
    source = FakeSource(
        [
            JobCandidate(
                "HR Officer",
                "CIMB Group Holdings Berhad",
                "https://my.jooble.org/jdp/123",
                "Malaysia",
                "jooble",
            ),
            JobCandidate(
                " HR   Officer ",
                " CIMB Group Holdings Berhad ",
                "https://careers.example.test/cimb/hr",
                "Kuala Lumpur",
                "foundit",
            ),
        ]
    )
    jobs = FakeJobsRepo()

    result = await _service(jobs, [("jooble", source)]).get_or_search(object(), "R03")

    assert result.openings["1023"].canonical_url == "https://careers.example.test/cimb/hr"
    assert jobs.replacements[0][1][0].provider == "foundit"


async def test_company_matcher_matches_short_brand_to_full_legal_name():
    assert _company_matches(_core_tokens("CIMB"), _core_tokens("CIMB Group Holdings Berhad"))
    assert _company_matches(_core_tokens("RHB Bank"), _core_tokens("RHB Bank Berhad"))
    assert _company_matches(_core_tokens("Astro Holdings"), _core_tokens("Astro Malaysia Holdings Berhad"))
    assert _company_matches(_core_tokens(" CIMB  Group Holdings Berhad "), _core_tokens("CIMB Group Holdings Berhad"))


async def test_company_matcher_rejects_midword_and_generic_token_hits():
    # mid-word substrings that plain `in` would wrongly accept
    assert not _company_matches(_core_tokens("Axi"), _core_tokens("Axis Real Estate Investment Trust"))
    assert not _company_matches(_core_tokens("Regask"), _core_tokens("Gas Malaysia Berhad"))
    assert not _company_matches(_core_tokens("UCT"), _core_tokens("Sunway Construction Group Berhad"))
    # shared-but-generic single token is not a contiguous-run match
    assert not _company_matches(_core_tokens("MVC Resources"), _core_tokens("QL Resources Berhad"))
    # empty core (name made only of legal words) never matches
    assert not _company_matches(_core_tokens("Berhad"), _core_tokens("CIMB Group Holdings Berhad"))


async def test_short_provider_brand_attaches_to_full_legal_employer():
    jooble = FakeSource(
        [
            JobCandidate(
                "Credit Analyst",
                "CIMB",
                "https://my.jooble.org/jdp/9",
                "Kuala Lumpur",
                "jooble",
            )
        ]
    )
    jobs = FakeJobsRepo()

    result = await _service(jobs, [("jooble", jooble)]).get_or_search(object(), "R03")

    assert result.status == "ready"
    assert result.openings["1023"].canonical_url == "https://my.jooble.org/jdp/9"


async def test_refresh_failure_returns_temporary_state_without_replacing_previous_cache():
    jobs = FakeJobsRepo(
        JobSearchRecord("R03", "Human Resources Officer", "ready", SEARCHED_AT, ["jooble"])
    )
    jobs.active = {
        "1023": StoredJobOpening(
            "R03",
            "1023",
            "HR Officer",
            "https://jobs.example.test/old",
            "Malaysia",
            "jooble",
            SEARCHED_AT,
            "active",
        )
    }
    sources = [
        ("jooble", FakeSource(error=JobSourceUnavailable("jooble failed"))),
        ("foundit", FakeSource(error=JobSourceUnavailable("foundit failed"))),
    ]

    result = await _service(jobs, sources).refresh(object(), "R03")

    assert result.status == "temporarily_unavailable"
    assert result.openings == {}
    assert jobs.replacements == []
    assert jobs.record.status == "ready"
