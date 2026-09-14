import pytest

from app.repositories.employers import CuratedEmployer
from app.schemas.employers import EmployerMatchRequest
from app.services.employers import EmployerService

pytestmark = pytest.mark.asyncio


def _emp(id, name, discloses, report_url="https://r", logo_text="L"):
    return CuratedEmployer(
        id=id,
        name=name,
        industry="X",
        location="KL",
        website="https://w",
        summary="...",
        logo_text=logo_text,
        logo_bg="#000",
        logo_fg="#fff",
        discloses=discloses,
        report_label="Sustainability Report 2024",
        report_url=report_url,
    )


class FakeRepo:
    def __init__(self, employers):
        self._employers = employers

    async def get_curated_employers(self, session):
        return self._employers


async def test_ranks_by_met_count_and_drops_zero_matches():
    repo = FakeRepo([
        _emp("a", "Alpha", ["flexible_work"]),
        _emp("b", "Bravo", ["flexible_work", "parental_support", "inclusive_workplace"]),
        _emp("c", "Charlie", ["childcare_support"]),  # matches none -> dropped
    ])
    svc = EmployerService(repo=repo)
    resp = await svc.match(
        EmployerMatchRequest(priorities=["flexible_work", "parental_support", "inclusive_workplace"]),
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
    svc = EmployerService(repo=repo)
    resp = await svc.match(EmployerMatchRequest(priorities=["flexible_work"]), session=object())
    assert [e.name for e in resp.employers] == ["Alpha", "Zeta"]


async def test_missing_report_and_logo_pass_through_as_none():
    repo = FakeRepo([_emp("a", "Alpha", ["flexible_work"], report_url=None, logo_text=None)])
    svc = EmployerService(repo=repo)
    resp = await svc.match(EmployerMatchRequest(priorities=["flexible_work"]), session=object())
    assert resp.employers[0].report is None
    assert resp.employers[0].logo is None


async def test_empty_priorities_returns_no_employers():
    repo = FakeRepo([_emp("a", "Alpha", ["flexible_work"])])
    svc = EmployerService(repo=repo)
    resp = await svc.match(EmployerMatchRequest(priorities=[]), session=object())
    assert resp.employers == []
