from datetime import datetime, timezone

import pytest

from app.repositories.employers import EmployerEvidence
from app.repositories.jobs import StoredJobOpening
from app.schemas.employers import EmployerMatchRequest
from app.services.employers import EmployerService

pytestmark = pytest.mark.asyncio


def _emp(id, name, discloses, report_url="https://r", logo_text="L"):
    return EmployerEvidence(
        id=id,
        name=name,
        industry="X",
        location="KL",
        website="https://w",
        summary="...",
        discloses=discloses,
        report_label="Sustainability Report 2024",
        report_url=report_url,
    )


class FakeRepo:
    def __init__(self, employers):
        self._employers = employers

    async def get_employers_with_evidence(self, session):
        return self._employers


class FakeJobSearch:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    async def get_or_search(self, session, role_id):
        self.calls.append(("get", role_id))
        return self.outcome

    async def refresh(self, session, role_id):
        self.calls.append(("refresh", role_id))
        return self.outcome


def _empty_job_search():
    return FakeJobSearch(type("Outcome", (), {
        "status": "empty",
        "searched_at": datetime(2026, 10, 4, tzinfo=timezone.utc),
        "openings": {},
    })())


async def test_ranks_by_met_count_and_drops_zero_matches():
    repo = FakeRepo([
        _emp("a", "Alpha", ["flexible_work"]),
        _emp("b", "Bravo", ["flexible_work", "parental_support", "inclusive_workplace"]),
        _emp("c", "Charlie", ["childcare_support"]),  # matches none -> dropped
    ])
    svc = EmployerService(repo=repo, job_search=_empty_job_search())
    resp = await svc.match(
        EmployerMatchRequest(
            target_role_id="role_dev",
            priorities=["flexible_work", "parental_support", "inclusive_workplace"],
        ),
        session=object(),
    )
    ids = [e.id for e in resp.employers]
    assert ids == ["b", "a"]  # b (3 met) before a (1 met); c dropped
    assert resp.employers[0].met == ["flexible_work", "parental_support", "inclusive_workplace"]
    assert resp.employers[1].unmet == ["parental_support", "inclusive_workplace"]


async def test_ties_broken_by_name():
    repo = FakeRepo([
        _emp("z", "Zeta", ["flexible_work"]),
        _emp("a", "Alpha", ["flexible_work"]),
    ])
    svc = EmployerService(repo=repo, job_search=_empty_job_search())
    resp = await svc.match(
        EmployerMatchRequest(target_role_id="role_dev", priorities=["flexible_work"]),
        session=object(),
    )
    assert [e.name for e in resp.employers] == ["Alpha", "Zeta"]


async def test_missing_report_and_logo_pass_through_as_none():
    repo = FakeRepo([_emp("a", "Alpha", ["flexible_work"], report_url=None, logo_text=None)])
    svc = EmployerService(repo=repo, job_search=_empty_job_search())
    resp = await svc.match(
        EmployerMatchRequest(target_role_id="role_dev", priorities=["flexible_work"]),
        session=object(),
    )
    assert resp.employers[0].report is None
    assert resp.employers[0].logo is None


async def test_empty_priorities_returns_no_employers():
    repo = FakeRepo([_emp("a", "Alpha", ["flexible_work"])])
    svc = EmployerService(repo=repo, job_search=_empty_job_search())
    req = EmployerMatchRequest.model_construct(target_role_id="role_dev", priorities=[])
    resp = await svc.match(req, session=object())
    assert resp.employers == []


async def test_job_openings_are_attached_and_sorted_before_fit_only_matches():
    searched_at = __import__("datetime").datetime(2026, 10, 4, tzinfo=__import__("datetime").timezone.utc)
    outcome = type("Outcome", (), {
        "status": "ready", "searched_at": searched_at,
        "openings": {
            "a": StoredJobOpening(
                role_id="role_dev", employer_id="a", title="Analyst",
                canonical_url="https://jobs.example.test/a", location="KL",
                provider="foundit", last_seen_at=searched_at, status="active",
            )
        },
    })()
    job_search = FakeJobSearch(outcome)
    svc = EmployerService(repo=FakeRepo([
        _emp("b", "Bravo", ["flexible_work", "parental_support"]),
        _emp("a", "Alpha", ["flexible_work"]),
    ]), job_search=job_search)
    resp = await svc.match(
        EmployerMatchRequest(target_role_id="role_dev", priorities=["flexible_work", "parental_support"]),
        session=object(),
    )
    assert [emp.id for emp in resp.employers] == ["a", "b"]
    assert resp.employers[0].job.title == "Analyst"
    assert resp.employers[1].job is None
    assert resp.job_search.status == "ready"
