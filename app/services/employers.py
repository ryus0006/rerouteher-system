"""E9 employer fit service.

Ranks a curated employer set by how many of the mother's chosen priorities each
employer's published report discloses. Employers that disclose none of her
priorities are dropped (silence is not a match). No PII: only priority ids and
an optional role id. Read-only.
"""
import logging

from app.repositories import employers as employers_repo
from app.schemas.employers import (
    EmployerMatchOut,
    EmployerMatchRequest,
    EmployerMatchResponse,
    LogoOut,
    ReportOut,
)

logger = logging.getLogger("rerouteher")


class EmployerService:
    def __init__(self, repo=employers_repo) -> None:
        self._repo = repo

    async def match(self, req: EmployerMatchRequest, session) -> EmployerMatchResponse:
        if not req.priorities:
            return EmployerMatchResponse(employers=[])

        curated = await self._repo.get_curated_employers(session)
        matched: list[EmployerMatchOut] = []
        for emp in curated:
            met = [p for p in req.priorities if p in emp.discloses]
            if not met:
                continue  # silent on everything she asked for is not a match
            unmet = [p for p in req.priorities if p not in emp.discloses]
            logo = (
                LogoOut(text=emp.logo_text, bg=emp.logo_bg or "#1f2a44", fg=emp.logo_fg or "#ffffff")
                if emp.logo_text
                else None
            )
            report = (
                ReportOut(label=emp.report_label or "Report", url=emp.report_url)
                if emp.report_url
                else None
            )
            matched.append(
                EmployerMatchOut(
                    id=emp.id,
                    name=emp.name,
                    industry=emp.industry,
                    location=emp.location,
                    logo=logo,
                    website=emp.website,
                    summary=emp.summary,
                    discloses=emp.discloses,
                    report=report,
                    met=met,
                    unmet=unmet,
                )
            )

        # Most priorities met first; name breaks ties so the order is stable.
        matched.sort(key=lambda e: (-len(e.met), e.name))
        logger.info(
            "employers: priorities=%d curated=%d matched=%d",
            len(req.priorities), len(curated), len(matched),
        )
        return EmployerMatchResponse(employers=matched)
