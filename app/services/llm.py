"""Gemini over direct HTTP (no SDK). Loaded once; injected into the companion
service. `generate` returns the first candidate's content plus the call's token
usage, so callers can render tool calls and record cost.

Supports key rotation: when a call on the active key fails (network error, rate
limit, or any other HTTP error), it falls back to the next configured key and
remembers the key that worked for later requests. When every key fails on the
main model (e.g. a 503 "model overloaded"), it tries each fallback model in turn
with the same keys. The model that worked is remembered too: later calls start on
it for a cooldown, then the main model is tried again.

EdgemodeClient is the alternative provider (OpenAI-compatible chat completions).
It takes and returns the same Gemini-shaped contents and tools, converting at the
edge, so the services work unchanged whichever provider LLM_PROVIDER selects."""
from __future__ import annotations

import json
import logging
import re
import time
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


def _log_call(provider: str, started: float, result: GenerateResult, tools: list[dict] | None) -> None:
    """One line per LLM pass, so a slow request can be split into its passes:
    duration, tokens, and what the output tokens went on (text vs each tool call)."""
    parts = result.content.get("parts") or []
    text_chars = sum(len(p["text"]) for p in parts if isinstance(p.get("text"), str))
    calls = [
        f"{p['functionCall'].get('name')}({len(json.dumps(p['functionCall'].get('args') or {}))} chars)"
        for p in parts
        if "functionCall" in p
    ]
    offered = sum(len(t.get("function_declarations") or []) for t in tools or [])
    logger.info(
        "llm pass: provider=%s model=%s ms=%.0f tokens_in=%d tokens_out=%d tools_offered=%d text_chars=%d calls=%s",
        provider,
        result.model,
        (time.monotonic() - started) * 1000,
        result.tokens_in,
        result.tokens_out,
        offered,
        text_chars,
        ",".join(calls) or "-",
    )


class GeminiClient:
    provider = "gemini"

    def __init__(
        self,
        api_keys: list[str],
        model: str,
        base_url: str,
        timeout_s: float,
        fallback_models: list[str] | None = None,
        fallback_cooldown_s: float = 600.0,
    ) -> None:
        if not api_keys:
            raise ValueError("GeminiClient needs at least one API key")
        self._api_keys = api_keys
        self._index = 0  # active key; advances when a key is rate-limited
        self.model = model
        self.fallback_models = [m for m in (fallback_models or []) if m != model]
        self._fallback_cooldown_s = fallback_cooldown_s
        self._active_model = 0  # index into [model, *fallback_models]; sticky like the key
        self._retry_main_at = 0.0  # monotonic time after which the main model is tried again
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
            settings.gemini_fallback_cooldown_s,
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

        started = time.monotonic()
        models = [self.model, *self.fallback_models]
        # Start on the model that last worked while its cooldown runs, so a down
        # main model is not re-checked (and waited on) by every call.
        start = self._active_model
        if start and time.monotonic() >= self._retry_main_at:
            logger.info("gemini fallback cooldown over; trying main model %s again", self.model)
            start = self._active_model = 0
        order = list(range(start, len(models))) + list(range(start))

        last_error = "gemini call failed"
        for n, i in enumerate(order):
            if n:
                logger.warning(
                    "gemini model %s failed on every key (%s); falling back to %s",
                    models[order[n - 1]],
                    last_error,
                    models[i],
                )
            result, last_error = await self._generate_with_keys(models[i], body)
            if result is None:
                continue
            if i != start:  # had to switch this call: remember it and start the cooldown
                self._active_model = i
                self._retry_main_at = time.monotonic() + self._fallback_cooldown_s
                if i:
                    logger.warning(
                        "gemini now using %s; main model %s retried in %ss",
                        models[i], self.model, int(self._fallback_cooldown_s),
                    )
            _log_call("gemini", started, result, tools)
            return result
        raise LlmError(f"all Gemini keys failed: {last_error}")

    async def _generate_with_keys(self, model: str, body: dict) -> tuple[GenerateResult | None, str]:
        """One model across every key. Returns (result, "") on success, or
        (None, last_error) once every key has failed."""
        url = f"{self._base_url}/models/{model}:generateContent"
        # Try the active key, then each remaining key on any failure: a network
        # error, a rate limit (429), or any other HTTP error (a revoked key gives
        # 403, a transient Gemini fault gives 5xx). The last error is surfaced
        # only once every key has been tried. A timeout is the exception: it
        # ends this model's turn at once (see below).
        last_error = "gemini call failed"
        for _ in range(len(self._api_keys)):
            key = self._api_keys[self._index]
            try:
                async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                    resp = await client.post(url, json=body, headers={"x-goog-api-key": key})
            except httpx.TimeoutException as exc:
                # A timeout means the model itself is slow, not this key, so the
                # other keys would wait just as long. Move on to the next model.
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "gemini key #%d timed out (%s) url=%s timeout=%ss; skipping the other keys for this model",
                    self._index + 1,
                    last_error,
                    url,
                    self._timeout_s,
                )
                return None, last_error
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


# Qwen3 can emit its reasoning inline; it is never part of the answer. A block cut
# off before its closing tag (e.g. at the token limit) is dropped to the end too.
_THINK_BLOCK = re.compile(r"<\s*think\s*>.*?(?:<\s*/\s*think\s*>\s*|$)", re.DOTALL | re.IGNORECASE)


def _to_openai_tools(tools: list[dict] | None) -> list[dict]:
    """Gemini [{"function_declarations": [...]}] -> OpenAI [{"type": "function", ...}].
    The declarations already use JSON-schema parameters, so they pass through."""
    return [
        {"type": "function", "function": decl}
        for tool in tools or []
        for decl in tool.get("function_declarations") or []
    ]


def _to_openai_messages(system_instruction: str, contents: list[dict]) -> list[dict]:
    """Gemini contents -> OpenAI messages. Gemini tool calls carry no id, so each
    call gets one here and the following functionResponse is paired with the
    earliest unanswered call of the same name."""
    messages: list[dict] = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    pending: list[tuple[str, str]] = []  # (tool name, call id) still awaiting a response
    n = 0
    for turn in contents:
        parts = turn.get("parts") or []
        text = "".join(p["text"] for p in parts if isinstance(p.get("text"), str))
        if turn.get("role") == "model":
            msg: dict = {"role": "assistant", "content": text}
            calls = [p["functionCall"] for p in parts if "functionCall" in p]
            if calls:
                msg["content"] = text or None
                msg["tool_calls"] = []
                for call in calls:
                    n += 1
                    call_id = f"call_{n}"
                    name = call.get("name", "")
                    pending.append((name, call_id))
                    msg["tool_calls"].append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(call.get("args") or {})},
                        }
                    )
            messages.append(msg)
            continue
        for p in parts:
            resp = p.get("functionResponse")
            if resp is None:
                continue
            name = resp.get("name", "")
            match = next((i for i, (pname, _) in enumerate(pending) if pname == name), 0)
            if pending:
                call_id = pending.pop(match)[1]
            else:
                n += 1
                call_id = f"call_{n}"
            messages.append(
                {"role": "tool", "tool_call_id": call_id, "content": json.dumps(resp.get("response"))}
            )
        if text:
            messages.append({"role": "user", "content": text})
    return messages


def _decode_nested(args: dict) -> dict:
    """Qwen sometimes sends a nested object argument as a JSON string, e.g.
    {"cv": "{\\"raw_text\\": ...}"}. Decode a top-level string that holds a JSON object
    or array so the tool gets the structure its schema declares."""
    out = {}
    for key, value in args.items():
        if isinstance(value, str) and value.lstrip()[:1] in ("{", "["):
            try:
                decoded = json.loads(value)
            except (ValueError, RecursionError):
                decoded = value
            out[key] = decoded if isinstance(decoded, (dict, list)) else value
        else:
            out[key] = value
    return out


def _to_gemini_content(message: dict) -> dict:
    """OpenAI assistant message -> Gemini content (text part + functionCall parts)."""
    parts: list[dict] = []
    text = _THINK_BLOCK.sub("", message.get("content") or "").strip()
    if text:
        parts.append({"text": text})
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        raw = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except (ValueError, RecursionError) as exc:
            raise LlmError(f"tool call {fn.get('name')!r} arguments are not valid JSON") from exc
        if args is None:
            args = {}
        # Callers read args with .get(); anything but an object would crash them.
        if not isinstance(args, dict):
            raise LlmError(f"tool call {fn.get('name')!r} arguments are not a JSON object")
        parts.append({"functionCall": {"name": fn.get("name", ""), "args": _decode_nested(args)}})
    return {"role": "model", "parts": parts}


class EdgemodeClient:
    """Edgemode over its OpenAI-compatible API, with GeminiClient's interface."""

    provider = "edgemode"

    def __init__(self, api_key: str, model: str, base_url: str, timeout_s: float) -> None:
        if not api_key:
            raise ValueError("EdgemodeClient needs an API key")
        self._api_key = api_key
        self.model = model
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s

    @classmethod
    def from_settings(cls, settings) -> "EdgemodeClient | None":
        if not settings.edgemode_api_key:
            logger.warning("EDGEMODE_API_KEY not set; companion LLM disabled")
            return None
        return cls(
            settings.edgemode_api_key,
            settings.edgemode_model,
            settings.edgemode_base_url,
            settings.edgemode_timeout_s,
        )

    async def generate(
        self, *, system_instruction: str, contents: list[dict], tools: list[dict] | None
    ) -> GenerateResult:
        body: dict = {"model": self.model, "messages": _to_openai_messages(system_instruction, contents)}
        openai_tools = _to_openai_tools(tools)
        if openai_tools:
            body["tools"] = openai_tools
        url = f"{self._base_url}/chat/completions"
        started = time.monotonic()
        try:
            # A short connect timeout so a host that is down fails in seconds, letting the
            # provider fallback take over instead of waiting the full read timeout.
            timeout = httpx.Timeout(self._timeout_s, connect=min(5.0, self._timeout_s))
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    url, json=body, headers={"Authorization": f"Bearer {self._api_key}"}
                )
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            logger.warning("edgemode call failed (%s) url=%s timeout=%ss", error, url, self._timeout_s)
            raise LlmError(f"edgemode call failed: {error}") from exc

        if resp.status_code >= 400:
            logger.warning("edgemode returned %d", resp.status_code)
            raise LlmError(f"edgemode call failed: http {resp.status_code}")

        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise LlmError(str(exc)) from exc

        choices = data.get("choices") or []
        if not choices:
            raise LlmError("no choices in Edgemode response")
        usage = data.get("usage") or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        # completion_tokens already includes any reasoning tokens.
        completion = int(usage.get("completion_tokens") or 0)
        if not completion and usage.get("total_tokens"):
            completion = int(usage["total_tokens"]) - prompt
        result = GenerateResult(
            content=_to_gemini_content(choices[0].get("message") or {}),
            tokens_in=prompt,
            tokens_out=max(completion, 0),
            model=self.model,
        )
        _log_call("edgemode", started, result, tools)
        return result


class FallbackLlmClient:
    """Main provider first, the other provider when it fails. Like the Gemini model
    fallback, the provider that answered is remembered for a cooldown so a provider
    that is down is not retried (and waited on) by every call."""

    def __init__(self, primary, secondary, cooldown_s: float = 600.0) -> None:
        self._clients = [primary, secondary]
        self._cooldown_s = cooldown_s
        self._active = 0  # index into _clients; sticky during the cooldown
        self._retry_primary_at = 0.0

    @property
    def provider(self) -> str:
        """The provider currently answering, so callers can adapt (verify mode)."""
        return self._clients[self._active].provider

    @property
    def model(self) -> str:
        return self._clients[self._active].model

    async def generate(
        self, *, system_instruction: str, contents: list[dict], tools: list[dict] | None
    ) -> GenerateResult:
        if self._active and time.monotonic() >= self._retry_primary_at:
            logger.info("llm fallback cooldown over; trying %s again", self._clients[0].provider)
            self._active = 0
        start = self._active
        errors = []
        for i in (start, 1 - start):
            client = self._clients[i]
            try:
                result = await client.generate(
                    system_instruction=system_instruction, contents=contents, tools=tools
                )
            except LlmError as exc:
                errors.append(f"{client.provider}: {exc}")
                if i == start:
                    logger.warning(
                        "llm provider %s failed (%s); falling back to %s",
                        client.provider,
                        exc,
                        self._clients[1 - start].provider,
                    )
                continue
            if i != start:
                self._active = i
                if i:  # moved off the main provider: start the cooldown
                    self._retry_primary_at = time.monotonic() + self._cooldown_s
                    logger.warning(
                        "llm now using %s; %s retried in %ss",
                        client.provider,
                        self._clients[0].provider,
                        int(self._cooldown_s),
                    )
            return result
        raise LlmError("all LLM providers failed: " + "; ".join(errors))


_CLIENTS = {"gemini": GeminiClient, "edgemode": EdgemodeClient}


def create_llm_client(settings) -> "GeminiClient | EdgemodeClient | FallbackLlmClient | None":
    """The client for settings.llm_provider, wrapped with the other provider as a fallback
    when LLM_FALLBACK_PROVIDER allows it and that provider has a key. None when the main
    provider's key is unset."""
    provider = settings.llm_provider.strip().lower()
    if provider not in _CLIENTS:
        raise ValueError(f"unknown LLM_PROVIDER {settings.llm_provider!r}; use 'gemini' or 'edgemode'")
    client = _CLIENTS[provider].from_settings(settings)
    if client is None:
        return None
    logger.info("LLM provider: %s (%s)", provider, client.model)

    fallback = settings.llm_fallback_provider.strip().lower()
    if fallback == "auto":
        fallback = "edgemode" if provider == "gemini" else "gemini"
    if fallback in ("", "none") or fallback == provider:
        return client
    if fallback not in _CLIENTS:
        raise ValueError(
            f"unknown LLM_FALLBACK_PROVIDER {settings.llm_fallback_provider!r}; "
            "use 'auto', 'none', 'gemini' or 'edgemode'"
        )
    backup = _CLIENTS[fallback].from_settings(settings)
    if backup is None:
        logger.info("LLM fallback %s has no key; running without a provider fallback", fallback)
        return client
    logger.info("LLM fallback provider: %s (%s)", fallback, backup.model)
    return FallbackLlmClient(client, backup, settings.llm_fallback_cooldown_s)
