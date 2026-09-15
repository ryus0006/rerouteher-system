"""Gemini over direct HTTP (no SDK). Loaded once; injected into the companion
service. `generate` returns the first candidate's content plus the call's token
usage, so callers can render tool calls and record cost.

Supports key rotation: when the active key is rate-limited (HTTP 429), it falls
back to the next configured key and remembers it for later requests."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

logger = logging.getLogger("rerouteher")


class LlmError(RuntimeError):
    pass


@dataclass
class GenerateResult:
    content: dict
    tokens_in: int = 0
    tokens_out: int = 0


class GeminiClient:
    def __init__(self, api_keys: list[str], model: str, base_url: str, timeout_s: float) -> None:
        if not api_keys:
            raise ValueError("GeminiClient needs at least one API key")
        self._api_keys = api_keys
        self._index = 0  # active key; advances when a key is rate-limited
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    @classmethod
    def from_settings(cls, settings) -> "GeminiClient | None":
        keys = settings.gemini_api_keys
        if not keys:
            logger.warning("GEMINI_API_KEY not set; companion LLM disabled")
            return None
        return cls(
            keys,
            settings.gemini_model,
            settings.gemini_base_url,
            settings.gemini_timeout_s,
        )

    async def generate(
        self, *, system_instruction: str, contents: list[dict], tools: list[dict] | None
    ) -> GenerateResult:
        url = f"{self._base_url}/models/{self.model}:generateContent"
        body: dict = {
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "contents": contents,
        }
        if tools:
            body["tools"] = tools

        # Try the active key, then each remaining key on a 429. Non-429 failures
        # are not a quota problem, so they surface immediately.
        for _ in range(len(self._api_keys)):
            key = self._api_keys[self._index]
            try:
                async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                    resp = await client.post(url, json=body, headers={"x-goog-api-key": key})
            except Exception as exc:  # noqa: BLE001
                raise LlmError(str(exc)) from exc

            if resp.status_code == 429:
                logger.warning(
                    "gemini key #%d rate-limited (429); rotating to the next key",
                    self._index + 1,
                )
                self._index = (self._index + 1) % len(self._api_keys)
                continue

            try:
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:  # noqa: BLE001
                raise LlmError(str(exc)) from exc

            candidates = data.get("candidates") or []
            if not candidates:
                raise LlmError("no candidates in Gemini response")
            usage = data.get("usageMetadata") or {}
            prompt = int(usage.get("promptTokenCount") or 0)
            total = int(usage.get("totalTokenCount") or 0)
            # Output = everything billed beyond the prompt: the answer plus any
            # thinking tokens (thoughtsTokenCount), which Gemini reports in the
            # total but not in candidatesTokenCount and bills at the output rate.
            # Fall back to candidatesTokenCount when total is absent.
            tokens_out = (total - prompt) if total else int(usage.get("candidatesTokenCount") or 0)
            return GenerateResult(
                content=candidates[0].get("content") or {"role": "model", "parts": []},
                tokens_in=prompt,
                tokens_out=max(tokens_out, 0),
            )

        raise LlmError("all Gemini keys are rate-limited")
