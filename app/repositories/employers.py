"""Read-only queries for E9 employer matching from authoritative employer evidence."""
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class EmployerEvidence:
    id: str
    name: str
    industry: str | None
    location: str | None
    website: str | None
    logo_url: str | None
    summary: str | None
    discloses: list[str]
    report_label: str | None
    report_url: str | None


def _resolve_report(row) -> tuple[str | None, str | None]:
    """(label, url): evidence report first, then employers report fallback."""
    evidence_url = (row.evidence_report_url or "").strip()
    if evidence_url:
        year = row.evidence_report_year or row.employer_report_year
        label = f"Sustainability Report {year}" if year else "Sustainability Report"
        return label, evidence_url
    employer_url = (row.employer_report_url or "").strip()
    if employer_url:
        label = (
            f"Sustainability Report {row.employer_report_year}"
            if row.employer_report_year
            else "Sustainability Report"
        )
        return label, employer_url
    return None, None


def _disclosures(row) -> list[str]:
    discloses = []
    if row.flexible_remote_disclosed is True:
        discloses.append("flexible_work")
    if row.childcare_nursing_phased_return is True:
        discloses.extend(["childcare_support", "parental_support"])
    if any(
        percentage is not None
        for percentage in (
            row.women_workforce_pct,
            row.women_management_pct,
            row.women_board_pct,
        )
    ):
        discloses.extend(["inclusive_workplace", "returning_to_work"])
    return discloses


async def get_employers_with_evidence(session: AsyncSession) -> list[EmployerEvidence]:
    rows = (
        await session.execute(
            text(
                "SELECT em.employer_id, em.name, em.sector, em.location, em.website, "
                "em.logo_url, "
                "em.report_url AS employer_report_url, "
                "em.report_year AS employer_report_year, "
                "ev.evidence_note, ev.report_url AS evidence_report_url, "
                "ev.report_year AS evidence_report_year, "
                "ev.women_workforce_pct, ev.women_management_pct, "
                "ev.women_board_pct, ev.flexible_remote_disclosed, "
                "ev.childcare_nursing_phased_return "
                "FROM employers em "
                "JOIN employer_report_evidence ev ON ev.employer_id = em.employer_id "
                "ORDER BY em.name"
            )
        )
    ).all()
    result: list[EmployerEvidence] = []
    for r in rows:
        label, url = _resolve_report(r)
        result.append(
            EmployerEvidence(
                id=str(r.employer_id),
                name=r.name,
                industry=r.sector,
                location=r.location,
                website=r.website,
                logo_url=r.logo_url,
                summary=r.evidence_note,
                discloses=_disclosures(r),
                report_label=label,
                report_url=url,
            )
        )
    return result
