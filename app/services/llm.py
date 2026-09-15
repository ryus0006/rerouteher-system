"""Gemini over direct HTTP (no SDK). Loaded once; injected into the companion
service. Returns the first candidate's content dict verbatim so the caller can
read text parts and functionCall parts (tool calling)."""
from __future__ import annotations

import logging

import httpx

logger = logging.getLogger("rerouteher")


class LlmError(RuntimeError):
    pass


class GeminiClient:
    def __init__(self, api_key: str, model: str, base_url: str, timeout_s: float) -> None:
        self._api_key = api_key
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    @classmethod
    def from_settings(cls, settings) -> "GeminiClient | None":
        if not settings.gemini_api_key:
            logger.warning("GEMINI_API_KEY not set; companion LLM disabled")
            return None
        return cls(
            settings.gemini_api_key,
            settings.gemini_model,
            settings.gemini_base_url,
            settings.gemini_timeout_s,
        )

    async def generate(
        self, *, system_instruction: str, contents: list[dict], tools: list[dict] | None
    ) -> dict:
        url = f"{self._base_url}/models/{self.model}:generateContent"
        body: dict = {
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "contents": contents,
        }
        if tools:
            body["tools"] = tools
        try:
            async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                resp = await client.post(url, json=body, headers={"x-goog-api-key": self._api_key})
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise LlmError(str(exc)) from exc
        candidates = data.get("candidates") or []
        if not candidates:
            raise LlmError("no candidates in Gemini response")
        return candidates[0].get("content") or {"role": "model", "parts": []}
