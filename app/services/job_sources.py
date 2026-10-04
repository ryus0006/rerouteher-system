"""External job-source adapters with one normalised candidate contract."""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import httpx

logger = logging.getLogger("rerouteher")


class JobSourceUnavailable(RuntimeError):
    """Raised when a provider cannot complete a search."""


@dataclass(frozen=True)
class JobCandidate:
    title: str
    company_name: str
    url: str
    location: str | None
    provider: str


def _clean(value) -> str:
    return str(value or "").strip()


def _location(value) -> str | None:
    if isinstance(value, list):
        value = ", ".join(_clean(item) for item in value if _clean(item))
    elif isinstance(value, dict):
        value = value.get("name") or value.get("label") or ""
    value = _clean(value)
    return value or None


class JoobleJobSource:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout_s: float,
        api_key_2: str = "",
        transport=None,
    ) -> None:
        self._api_keys = [key for key in (api_key, api_key_2) if key]
        if not self._api_keys:
            raise ValueError("JoobleJobSource needs at least one API key")
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._transport = transport

    @classmethod
    def from_settings(cls, settings) -> "JoobleJobSource | None":
        if not settings.jooble_api_key:
            logger.warning("JOOBLE_API_KEY not set; Jooble job search disabled")
            return None
        return cls(
            api_key=settings.jooble_api_key,
            api_key_2=settings.jooble_api_key_2,
            base_url=settings.jooble_base_url,
            timeout_s=settings.jooble_timeout_s,
        )

    async def search(
        self, search_title: str, location: str, limit: int
    ) -> list[JobCandidate]:
        body = {
            "keywords": search_title,
            "location": location,
            "page": "1",
            "ResultOnPage": min(limit, 50),
        }
        for key_index, key in enumerate(self._api_keys):
            try:
                async with httpx.AsyncClient(
                    timeout=self._timeout_s, transport=self._transport
                ) as client:
                    response = await client.post(
                        f"{self._base_url}/{key}",
                        json=body,
                    )
            except httpx.HTTPError as exc:
                raise JobSourceUnavailable("jooble request failed") from exc

            if response.status_code == 429 and key_index < len(self._api_keys) - 1:
                logger.warning("jooble key #%d rate-limited; rotating key", key_index + 1)
                continue
            if response.status_code >= 400:
                raise JobSourceUnavailable(
                    f"jooble provider returned HTTP {response.status_code}"
                )
            try:
                payload = response.json()
            except ValueError as exc:
                raise JobSourceUnavailable("jooble returned invalid JSON") from exc
            return [
                JobCandidate(
                    title=_clean(job.get("title")),
                    company_name=_clean(job.get("company")),
                    url=_clean(job.get("link")),
                    location=_location(job.get("location")),
                    provider="jooble",
                )
                for job in payload.get("jobs", [])
                if _clean(job.get("title"))
                and _clean(job.get("company"))
                and _clean(job.get("link"))
            ]

        raise JobSourceUnavailable("jooble keys are unavailable")


class FounditJobSource:
    def __init__(
        self,
        base_url: str,
        timeout_s: float,
        transport=None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._transport = transport

    async def search(
        self, search_title: str, location: str, limit: int
    ) -> list[JobCandidate]:
        # foundit.my is already Malaysia-scoped; a country name in `locations`
        # returns 400, so it is left empty rather than passing `location`.
        params = {
            "start": 0,
            "limit": min(limit, 50),
            "query": search_title,
            "locations": "",
            "sort": 1,
        }
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_s, transport=self._transport
            ) as client:
                response = await client.get(
                    f"{self._base_url}/middleware/jobsearch",
                    params=params,
                )
        except httpx.HTTPError as exc:
            raise JobSourceUnavailable("foundit request failed") from exc
        if response.status_code >= 400:
            raise JobSourceUnavailable(
                f"foundit provider returned HTTP {response.status_code}"
            )
        try:
            rows = (response.json().get("jobSearchResponse") or {}).get("data") or []
        except ValueError as exc:
            raise JobSourceUnavailable("foundit returned invalid JSON") from exc

        candidates = []
        for row in rows:
            title = _clean(row.get("title"))
            company = _clean(row.get("companyName"))
            url = next(
                (
                    _clean(row.get(field))
                    for field in ("applyUrl", "redirectUrl", "jdUrl", "seoJdUrl")
                    if _clean(row.get(field))
                ),
                "",
            )
            if title and company and url:
                candidates.append(
                    JobCandidate(
                        title=title,
                        company_name=company,
                        url=url,
                        location=_location(row.get("locations")),
                        provider="foundit",
                    )
                )
        return candidates


class TavilyGeminiJobSource:
    def __init__(self, tavily, llm) -> None:
        self._tavily = tavily
        self._llm = llm

    async def search(
        self, search_title: str, location: str, limit: int
    ) -> list[JobCandidate]:
        if self._tavily is None or self._llm is None:
            raise JobSourceUnavailable("tavily fallback is not configured")
        try:
            results = await self._tavily.search(
                f'"{search_title}" {location} jobs',
                min(limit, 20),
            )
            if not results:
                return []
            prompt = json.dumps(
                [
                    {"title": result.title, "url": result.url, "snippet": result.snippet}
                    for result in results
                ]
            )
            generated = await self._llm.generate(
                system_instruction=(
                    "Extract only real job openings from the supplied search results. "
                    "Return a JSON array with title, company, and url. Do not invent "
                    "openings or URLs."
                ),
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                tools=None,
            )
            text = "".join(
                part.get("text", "")
                for part in generated.content.get("parts", [])
                if isinstance(part, dict)
            )
            match = re.search(r"\[[\s\S]*\]", text)
            extracted = json.loads(match.group(0)) if match else []
        except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
            raise JobSourceUnavailable("tavily fallback returned invalid data") from exc
        except Exception as exc:  # noqa: BLE001
            raise JobSourceUnavailable("tavily fallback failed") from exc

        allowed_urls = {result.url for result in results}
        candidates = []
        for item in extracted if isinstance(extracted, list) else []:
            title = _clean(item.get("title"))
            company = _clean(item.get("company"))
            url = _clean(item.get("url"))
            if title and company and url in allowed_urls:
                candidates.append(
                    JobCandidate(
                        title=title,
                        company_name=company,
                        url=url,
                        location=location,
                        provider="tavily",
                    )
                )
        return candidates
