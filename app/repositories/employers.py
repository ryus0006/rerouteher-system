"""Read-only queries for E9 employer matching.

Hybrid source: our curated employer_overlay (display + 5-priority discloses)
LEFT JOINed to the db team's employer_report_evidence for the real ESG report
link, falling back to the overlay's own report. Requests write nothing.
"""
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class CuratedEmployer:
    id: str
    name: str
    industry: str | None
    location: str | None
    website: str | None
    summary: str | None
    logo_text: str | None
    logo_bg: str | None
    logo_fg: str | None
    discloses: list[str]
    report_label: str | None
    report_url: str | None


def _resolve_report(row) -> tuple[str | None, str | None]:
    """(label, url): real DB report first, then overlay fallback, else None."""
    db_url = (row.db_report_url or "").strip()
    if db_url:
        year = row.db_report_year or row.overlay_report_year
        label = f"Sustainability Report {year}" if year else "Sustainability Report"
        return label, db_url
    overlay_url = (row.overlay_report_url or "").strip()
    if overlay_url:
        label = row.report_label or (
            f"Sustainability Report {row.overlay_report_year}"
            if row.overlay_report_year
            else "Sustainability Report"
        )
        return label, overlay_url
    return None, None


async def get_curated_employers(session: AsyncSession) -> list[CuratedEmployer]:
    rows = (
        await session.execute(
            text(
                "SELECT o.employer_id, o.name, o.industry, o.location, o.website, "
                "o.summary, o.logo_text, o.logo_bg, o.logo_fg, o.discloses, "
                "o.report_label, o.report_url AS overlay_report_url, "
                "o.report_year AS overlay_report_year, "
                "e.report_url AS db_report_url, e.report_year AS db_report_year "
                "FROM employer_overlay o "
                "LEFT JOIN employer_report_evidence e ON e.employer_id = o.employer_id "
                "ORDER BY o.name"
            )
        )
    ).all()
    result: list[CuratedEmployer] = []
    for r in rows:
        label, url = _resolve_report(r)
        result.append(
            CuratedEmployer(
                id=str(r.employer_id),
                name=r.name,
                industry=r.industry,
                location=r.location,
                website=r.website,
                summary=r.summary,
                logo_text=r.logo_text,
                logo_bg=r.logo_bg,
                logo_fg=r.logo_fg,
                discloses=list(r.discloses or []),
                report_label=label,
                report_url=url,
            )
        )
    return result
