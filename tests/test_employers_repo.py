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
        name="CIMB Group Holdings Berhad",
        sector="Financial Services",
        location="Kuala Lumpur",
        website="https://www.cimb.com/",
        evidence_note="Family-friendly policies and flexible work.",
        evidence_report_url="https://www.cimb.com/content/dam/report.pdf",
        evidence_report_year=2024,
        employer_report_url="https://www.cimb.com/group-report.pdf",
        employer_report_year=2023,
        women_workforce_pct=57,
        women_management_pct=42,
        women_board_pct=None,
        flexible_remote_disclosed=True,
        childcare_nursing_phased_return=True,
        extraction_status="pending_review",
    )
    base.update(over)
    return FakeRow(**base)


async def test_evidence_report_and_display_fields_are_mapped():
    out = await employers_repo.get_employers_with_evidence(FakeSession([_row()]))
    assert out[0].name == "CIMB Group Holdings Berhad"
    assert out[0].industry == "Financial Services"
    assert out[0].report_url == "https://www.cimb.com/content/dam/report.pdf"
    assert out[0].report_label == "Sustainability Report 2024"
    assert out[0].summary == "Family-friendly policies and flexible work."


async def test_report_falls_back_to_employer_report_when_evidence_url_missing():
    out = await employers_repo.get_employers_with_evidence(
        FakeSession([_row(evidence_report_url=None, evidence_report_year=None)])
    )
    assert out[0].report_url == "https://www.cimb.com/group-report.pdf"
    assert out[0].report_label == "Sustainability Report 2023"


async def test_report_is_none_when_neither_authoritative_report_exists():
    out = await employers_repo.get_employers_with_evidence(
        FakeSession(
            [
                _row(
                    evidence_report_url=None,
                    evidence_report_year=None,
                    employer_report_url=None,
                    employer_report_year=None,
                )
            ]
        )
    )
    assert out[0].report_url is None
    assert out[0].report_label is None


async def test_priority_disclosures_are_derived_from_evidence_flags_and_percentages():
    out = await employers_repo.get_employers_with_evidence(FakeSession([_row()]))
    assert out[0].discloses == [
        "flexible_work",
        "childcare_support",
        "parental_support",
        "inclusive_workplace",
        "returning_to_work",
    ]


async def test_all_extraction_statuses_are_included():
    out = await employers_repo.get_employers_with_evidence(
        FakeSession([_row(extraction_status="needs_review")])
    )
    assert len(out) == 1


async def test_inclusive_rules_require_a_non_null_women_percentage():
    out = await employers_repo.get_employers_with_evidence(
        FakeSession(
            [
                _row(
                    women_workforce_pct=None,
                    women_management_pct=None,
                    women_board_pct=None,
                    flexible_remote_disclosed=False,
                    childcare_nursing_phased_return=False,
                )
            ]
        )
    )
    assert out[0].discloses == []
