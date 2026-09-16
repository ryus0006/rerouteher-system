"""On-demand learning-material fill (E6).

When a gap is computed, any gap skill with no curated resource is filled in the
background: Tavily (official search API) returns candidate resources, a plain
Gemini call picks ONE free course from those candidates, the URL is verified, and
the resource is persisted so the next person with that skill sees it instantly.
The chosen URL must be one of the search results (no fabricated links). No PII is
used: only the skill name and definition are sent out; the mother is never
identified, and we store only the link plus factual metadata, never page content.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import httpx

from app.db import SessionLocal
from app.repositories import learning as learning_repo
from app.services.llm import GeminiClient, LlmError
from app.services.tavily import SearchResult, TavilyError, TavilySearcher

logger = logging.getLogger("rerouteher")

_PICK_SYSTEM = (
    "You are given a job skill (with its definition) and a list of web search "
    "results (title, url, snippet). Rank the FREE online courses or tutorials that "
    "build this skill as defined, best first. You MUST use urls from the provided "
    "results, do not invent any. Return ONLY a JSON array (no prose, no markdown "
    "fences) of up to 3 objects, best first, each with keys: title (string), url "
    "(string, one of the results), provider_name (string), delivery_mode (one of "
    "Course, Video, Article), duration_minutes (integer or null), level (Beginner, "
    "Intermediate, Advanced, or null), licence_note (string or null), relevance "
    "(number 0 to 1), evidence_note (one sentence on why it builds this skill). If "
    "none of the results are a free skill-building resource, return an empty array []."
)

# A browser-like UA and the statuses that mean "reachable but bot-guarded": a real
# page that refuses a bare programmatic request (seen with MIT Sloan in testing).
_BROWSER_UA = "Mozilla/5.0 (compatible; ReRouteHer/1.0; +learning-resource-check)"
_REACHABLE_GUARDED = {401, 403, 405, 429}

# How many gap skills to fill: exactly the ones the learning page surfaces. This
# mirrors the UI's MAX_FOCUS_AREAS (components/gap/FocusAreaList.jsx). A gap can
# return 100+ skills, but she only ever sees these, so we never fill more.
MAX_FILL_FOCUS_AREAS = 3


def focus_area_skill_ids(gaps, limit: int = MAX_FILL_FOCUS_AREAS) -> list[str]:
    """The skill_ids the learning page shows, mirroring the UI's pickFocusAreas.

    Reserve one slot for the top AI-usage gap so AI-literacy gaps do not crowd out
    role skills, fill the rest with the top role gaps, backfill any leftover slots
    from the remaining gaps, then order by uplift. `gaps` is the backend's uplift-
    ranked list; each item has `.skill_id`, `.band`, and `.uplift`.
    """
    role_gaps = [g for g in gaps if g.band != "ai_usage"]
    top_ai = next((g for g in gaps if g.band == "ai_usage"), None)
    role_slots = limit - 1 if top_ai else limit
    chosen = list(role_gaps[:role_slots])
    if top_ai is not None:
        chosen.append(top_ai)
    if len(chosen) < limit:
        for gap in gaps:
            if len(chosen) >= limit:
                break
            if gap not in chosen:
                chosen.append(gap)
    chosen.sort(key=lambda g: g.uplift, reverse=True)
    return [g.skill_id for g in chosen]


@dataclass
class ChosenResource:
    skill_id: str
    title: str
    url: str
    provider_name: str
    delivery_mode: str | None
    duration_minutes: int | None
    level: str | None
    licence_note: str | None
    relevance: float | None
    evidence_note: str | None


def _extract_text(content: dict) -> str:
    parts = (content or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts if isinstance(p, dict))


def _loads_json(text: str):
    """Parse the model's text into JSON, tolerating markdown fences and surrounding
    prose. Returns a dict, a list, or None."""
    raw = (text or "").strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        m = re.search(r"\[.*\]|\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except (ValueError, TypeError):
            return None


def _choice_from_dict(skill_id: str, data) -> "ChosenResource | None":
    if not isinstance(data, dict):
        return None
    url = (data.get("url") or "").strip() if isinstance(data.get("url"), str) else ""
    if not url or not url.lower().startswith(("http://", "https://")):
        return None

    def _int(v):
        try:
            return int(v) if v is not None else None
        except (ValueError, TypeError):
            return None

    def _float(v):
        try:
            return float(v) if v is not None else None
        except (ValueError, TypeError):
            return None

    return ChosenResource(
        skill_id=skill_id,
        title=(data.get("title") or "").strip() or "Untitled resource",
        url=url,
        provider_name=(data.get("provider_name") or "Web").strip() or "Web",
        delivery_mode=(data.get("delivery_mode") or None),
        duration_minutes=_int(data.get("duration_minutes")),
        level=(data.get("level") or None),
        licence_note=(data.get("licence_note") or None),
        relevance=_float(data.get("relevance")),
        evidence_note=(data.get("evidence_note") or None),
    )


def _parse_choice(skill_id: str, text: str) -> "ChosenResource | None":
    """Parse a single pick (one JSON object)."""
    return _choice_from_dict(skill_id, _loads_json(text))


def _parse_choices(skill_id: str, text: str) -> list[ChosenResource]:
    """Parse a ranked shortlist (a JSON array; a lone object is treated as one)."""
    data = _loads_json(text)
    items = data if isinstance(data, list) else [data]
    out: list[ChosenResource] = []
    for item in items:
        choice = _choice_from_dict(skill_id, item)
        if choice is not None:
            out.append(choice)
    return out


class LearningFillService:
    def __init__(self, llm: "GeminiClient | None", searcher: "TavilySearcher | None",
                 *, enabled: bool = True, url_timeout_s: float = 6.0, candidates: int = 6) -> None:
        self._llm = llm
        self._searcher = searcher
        self._enabled = bool(enabled and llm is not None and searcher is not None)
        self._url_timeout_s = url_timeout_s
        self._candidates = candidates
        self._verify_transport = None  # tests inject an httpx.MockTransport here

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def _verify_url(self, url: str) -> bool:
        headers = {"User-Agent": _BROWSER_UA}
        try:
            async with httpx.AsyncClient(
                timeout=self._url_timeout_s, follow_redirects=True,
                headers=headers, transport=self._verify_transport,
            ) as client:
                resp = await client.head(url)
                # A HEAD 401/403/405 can mean "method blocked / guarded", not "gone":
                # some servers refuse HEAD but serve GET. Only then is a GET worth it.
                # For 404 / 5xx a GET would not change the answer, so we trust HEAD.
                if resp.status_code in _REACHABLE_GUARDED:
                    resp = await client.get(url)
                if resp.status_code < 400:
                    return True
                if resp.status_code in _REACHABLE_GUARDED:
                    logger.info("learning-fill URL %s guarded (%s); accepting as reachable",
                                url, resp.status_code)
                    return True
                return False
        except Exception as exc:  # noqa: BLE001
            logger.info("learning-fill URL check failed for %s (%s)", url, exc)
            return False

    async def _rank(self, skill_name: str, definition: "str | None",
                    results: list[SearchResult]) -> list[ChosenResource]:
        """One Gemini call: a ranked shortlist (best free first), grounded to the
        candidate URLs and de-duplicated. Empty if the call fails or nothing fits."""
        candidates = [{"title": r.title, "url": r.url, "snippet": r.snippet} for r in results]
        candidate_urls = {r.url for r in results}
        user = (f"Skill: {skill_name}\nDefinition: {definition or ''}\n"
                f"Results:\n{json.dumps(candidates, ensure_ascii=False)}")
        try:
            result = await self._llm.generate(
                system_instruction=_PICK_SYSTEM,
                contents=[{"role": "user", "parts": [{"text": user}]}],
                tools=None,
            )
        except LlmError as exc:
            logger.info("learning-fill rank failed for '%s' (%s)", skill_name, exc)
            return []
        ranked: list[ChosenResource] = []
        seen: set[str] = set()
        for choice in _parse_choices("", _extract_text(result.content)):
            if choice.url not in candidate_urls:  # grounding guard: no fabricated links
                logger.info("learning-fill pick '%s' not among candidates; skipping", choice.url)
                continue
            if choice.url in seen:
                continue
            seen.add(choice.url)
            ranked.append(choice)
        return ranked

    async def find_resource(self, skill_id: str, skill_name: str,
                            definition: "str | None" = None) -> "ChosenResource | None":
        """Search, rank free resources with one Gemini call, then walk the shortlist
        and return the first URL that verifies. No DB writes."""
        if not self._enabled:
            return None
        query = f"free online course tutorial: {skill_name}. {definition or ''}".strip()
        try:
            results = await self._searcher.search(query, self._candidates)
        except TavilyError as exc:
            logger.info("learning-fill search failed for '%s' (%s)", skill_name, exc)
            return None
        if not results:
            return None
        for choice in await self._rank(skill_name, definition, results):
            if await self._verify_url(choice.url):
                choice.skill_id = skill_id
                return choice
        logger.info("learning-fill found no reachable resource for '%s'", skill_name)
        return None

    async def fill_for_skills(self, skill_ids: list[str]) -> None:
        """Background entry point: fill any of these skills with no curated resource.

        Opens its own DB session (the request session is gone), and commits per
        skill so a later failure never rolls back an earlier success. Each skill is
        isolated: a failure is logged and skipped, never raised, so a background
        task cannot crash the worker. Idempotent, so overlapping gaps are safe.
        """
        if not self._enabled or not skill_ids:
            return
        async with SessionLocal() as session:
            missing = await learning_repo.skills_missing_resources(session, skill_ids)
            if not missing:
                return
            labels = await learning_repo.get_skill_labels(session, missing)
        for skill_id in missing:
            label = labels.get(skill_id)
            if label is None:
                continue
            try:
                chosen = await self.find_resource(skill_id, label.name, label.definition)
                if chosen is None:
                    continue
                async with SessionLocal() as session:
                    await learning_repo.upsert_filled_resource(session, chosen)
                    await session.commit()
                logger.info("learning-fill persisted '%s' for skill %s", chosen.url, skill_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("learning-fill failed for skill %s (%s)", skill_id, exc)
