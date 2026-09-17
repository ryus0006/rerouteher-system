"""Tavily web search over direct HTTP (no SDK). Official search API used to find
candidate learning resources; the LLM then picks one. Loaded once, injected.

Supports key rotation: when the active key is rate-limited (HTTP 429), it falls
back to the next configured key and remembers it, mirroring the Gemini client."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger("rerouteher")


class TavilyError(RuntimeError):
    pass


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


class TavilySearcher:
    def __init__(self, api_keys: list[str], base_url: str, timeout_s: float) -> None:
        if not api_keys:
            raise ValueError("TavilySearcher needs at least one api key")
        self._api_keys = api_keys
        self._index = 0  # active key; advances when a key is rate-limited
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._transport = None  # tests inject an httpx.MockTransport here

    @classmethod
    def from_settings(cls, settings) -> "TavilySearcher | None":
        keys = settings.tavily_api_keys
        if not keys:
            logger.warning("TAVILY_API_KEY not set; learning fill search disabled")
            return None
        return cls(keys, settings.tavily_base_url, settings.tavily_timeout_s)

    async def search(self, query: str, k: int) -> list[SearchResult]:
        body = {"query": query, "max_results": k, "search_depth": "basic"}
        # Try the active key, then each remaining key on a 429. Non-429 failures are
        # not a quota problem, so they surface immediately.
        for _ in range(len(self._api_keys)):
            key = self._api_keys[self._index]
            try:
                async with httpx.AsyncClient(timeout=self._timeout_s, transport=self._transport) as client:
                    resp = await client.post(
                        f"{self._base_url}/search",
                        json={**body, "api_key": key},
                    )
            except Exception as exc:  # noqa: BLE001
                raise TavilyError(str(exc)) from exc

            if resp.status_code == 429:
                logger.warning(
                    "tavily key #%d rate-limited (429); rotating to the next key",
                    self._index + 1,
                )
                self._index = (self._index + 1) % len(self._api_keys)
                continue

            try:
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:  # noqa: BLE001
                raise TavilyError(str(exc)) from exc

            results: list[SearchResult] = []
            for r in data.get("results", []):
                url = (r.get("url") or "").strip()
                if not url:
                    continue
                results.append(
                    SearchResult(
                        title=(r.get("title") or "").strip(),
                        url=url,
                        snippet=(r.get("content") or "")[:300],
                    )
                )
            return results

        raise TavilyError("all Tavily keys are rate-limited")
