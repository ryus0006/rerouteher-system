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
    JobOpeningOut,
    JobRefreshResponse,
    JobSearchOut,
    ReportOut,
)

logger = logging.getLogger("rerouteher")


class EmployerService:
    def __init__(self, repo=employers_repo, job_search=None) -> None:
        self._repo = repo
        self._job_search = job_search

    async def match(self, req: EmployerMatchRequest, session) -> EmployerMatchResponse:
        if self._job_search is None:
            raise RuntimeError("job search service is not configured")

        job_search = await self._job_search.get_or_search(session, req.target_role_id)

        curated = await self._repo.get_employers_with_evidence(session)
        matched: list[EmployerMatchOut] = []
        for emp in curated:
            met = [p for p in req.priorities if p in emp.discloses]
            if not met:
                continue  # silent on everything she asked for is not a match
            unmet = [p for p in req.priorities if p not in emp.discloses]
            report = (
                ReportOut(label=emp.report_label or "Report", url=emp.report_url)
                if emp.report_url
                else None
            )
            opening = job_search.openings.get(emp.id)
            matched.append(
                EmployerMatchOut(
                    id=emp.id,
                    name=emp.name,
                    industry=emp.industry,
                    location=emp.location,
                    logo=None,
                    website=emp.website,
                    summary=emp.summary,
                    discloses=emp.discloses,
                    report=report,
                    met=met,
                    unmet=unmet,
                    job=(
                        JobOpeningOut(
                            title=opening.title,
                            url=opening.canonical_url,
                            found_at=opening.last_seen_at,
                        )
                        if opening
                        else None
                    ),
                )
            )

        # Hiring employers are the actionable first group; fit ordering is unchanged
        # inside each group.
        matched.sort(key=lambda e: (e.job is None, -len(e.met), e.name))
        logger.info(
            "employers: priorities=%d curated=%d matched=%d",
            len(req.priorities), len(curated), len(matched),
        )
        return EmployerMatchResponse(
            job_search=JobSearchOut(
                status=job_search.status, searched_at=job_search.searched_at
            ),
            employers=matched,
        )

    async def refresh(self, role_id: str, session) -> JobRefreshResponse:
        if self._job_search is None:
            raise RuntimeError("job search service is not configured")
        outcome = await self._job_search.refresh(session, role_id)
        return JobRefreshResponse(
            role_id=role_id,
            status=outcome.status,
            opening_count=len(outcome.openings),
            searched_at=outcome.searched_at,
        )
