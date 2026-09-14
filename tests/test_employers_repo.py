import pytest

from app.repositories import employers as employers_repo

pytestmark = pytest.mark.asyncio


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class FakeRow:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeSession:
    def __init__(self, rows):
        self._rows = rows
        self.executed = []

    async def execute(self, stmt, params=None):
        self.executed.append((str(stmt), params))
        return FakeResult(self._rows)


def _row(**over):
    base = dict(
        employer_id="1023",
        name="CIMB",
        industry="Financial Services",
        location="Kuala Lumpur",
        website="https://www.cimb.com/",
        summary="...",
        logo_text="CIMB",
        logo_bg="#c4161c",
        logo_fg="#ffffff",
        discloses=["flexible_work", "inclusive_workplace"],
        report_label="Sustainability Report 2024",
        overlay_report_url="https://www.cimb.com/en/sustainability.html",
        overlay_report_year=2024,
        db_report_url="https://www.cimb.com/content/dam/report.pdf",
        db_report_year=2024,
    )
    base.update(over)
    return FakeRow(**base)


async def test_report_prefers_db_url_over_overlay():
    out = await employers_repo.get_curated_employers(FakeSession([_row()]))
    assert out[0].report_url == "https://www.cimb.com/content/dam/report.pdf"
    assert out[0].report_label == "Sustainability Report 2024"


async def test_report_falls_back_to_overlay_when_db_missing():
    out = await employers_repo.get_curated_employers(
        FakeSession([_row(db_report_url=None, db_report_year=None)])
    )
    assert out[0].report_url == "https://www.cimb.com/en/sustainability.html"


async def test_report_is_none_when_neither_present():
    out = await employers_repo.get_curated_employers(
        FakeSession([_row(db_report_url=None, db_report_year=None, overlay_report_url=None)])
    )
    assert out[0].report_url is None
    assert out[0].report_label is None


async def test_discloses_passthrough():
    out = await employers_repo.get_curated_employers(FakeSession([_row()]))
    assert out[0].discloses == ["flexible_work", "inclusive_workplace"]
