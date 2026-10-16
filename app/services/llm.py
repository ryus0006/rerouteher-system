"""Gemini over direct HTTP (no SDK). Loaded once; injected into the companion
service. `generate` returns the first candidate's content plus the call's token
usage, so callers can render tool calls and record cost.

Supports key rotation: when a call on the active key fails (network error, rate
limit, or any other HTTP error), it falls back to the next configured key and
remembers the key that worked for later requests. When every key fails on the
main model (e.g. a 503 "model overloaded"), it tries each fallback model in turn
with the same keys."""
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
    model: str = ""  # the model that produced this answer (main or a fallback)


class GeminiClient:
    def __init__(
        self,
        api_keys: list[str],
        model: str,
        base_url: str,
        timeout_s: float,
        fallback_models: list[str] | None = None,
    ) -> None:
        if not api_keys:
            raise ValueError("GeminiClient needs at least one API key")
        self._api_keys = api_keys
        self._index = 0  # active key; advances when a key is rate-limited
        self.model = model
        self.fallback_models = [m for m in (fallback_models or []) if m != model]
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
            settings.gemini_fallback_model_list,
        )

    async def generate(
        self, *, system_instruction: str, contents: list[dict], tools: list[dict] | None
    ) -> GenerateResult:
        body: dict = {
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "contents": contents,
        }
        if tools:
            body["tools"] = tools

        last_error = "gemini call failed"
        models = [self.model, *self.fallback_models]
        for n, model in enumerate(models):
            if n:
                logger.warning(
                    "gemini model %s failed on every key (%s); falling back to %s",
                    models[n - 1],
                    last_error,
                    model,
                )
            result, last_error = await self._generate_with_keys(model, body)
            if result is not None:
                return result
        raise LlmError(f"all Gemini keys failed: {last_error}")

    async def _generate_with_keys(self, model: str, body: dict) -> tuple[GenerateResult | None, str]:
        """One model across every key. Returns (result, "") on success, or
        (None, last_error) once every key has failed."""
        url = f"{self._base_url}/models/{model}:generateContent"
        # Try the active key, then each remaining key on any failure: a network
        # error, a rate limit (429), or any other HTTP error (a revoked key gives
        # 403, a transient Gemini fault gives 5xx). The last error is surfaced
        # only once every key has been tried.
        last_error = "gemini call failed"
        for _ in range(len(self._api_keys)):
            key = self._api_keys[self._index]
            try:
                async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                    resp = await client.post(url, json=body, headers={"x-goog-api-key": key})
            except Exception as exc:  # noqa: BLE001
                # Many httpx errors (e.g. ConnectTimeout/ReadTimeout) stringify to
                # an empty message, so include the exception type and the target
                # URL/timeout to make the real cause visible in the logs.
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "gemini key #%d call failed (%s) url=%s timeout=%ss; trying the next key",
                    self._index + 1,
                    last_error,
                    url,
                    self._timeout_s,
                    exc_info=True,
                )
                self._index = (self._index + 1) % len(self._api_keys)
                continue

            if resp.status_code >= 400:
                last_error = f"http {resp.status_code}"
                logger.warning(
                    "gemini key #%d returned %d; trying the next key",
                    self._index + 1,
                    resp.status_code,
                )
                self._index = (self._index + 1) % len(self._api_keys)
                continue

            try:
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
                model=model,
            ), ""

        return None, last_error
