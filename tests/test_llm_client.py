import pytest

from app.config import Settings
from app.services.llm import GeminiClient, LlmError


def test_from_settings_returns_none_without_key():
    assert GeminiClient.from_settings(Settings(gemini_api_key="")) is None


def test_from_settings_builds_client_with_key():
    client = GeminiClient.from_settings(Settings(gemini_api_key="k", gemini_model="m"))
    assert client is not None and client.model == "m"


def test_settings_llm_configured_flag():
    assert Settings(gemini_api_key="").llm_configured is False
    assert Settings(gemini_api_key="k").llm_configured is True


def test_settings_gemini_api_keys_orders_and_skips_blanks():
    s = Settings(gemini_api_key="a", gemini_api_key_2="", gemini_api_key_3="c")
    assert s.gemini_api_keys == ["a", "c"]


class _Resp:
    def __init__(self, status, key, usage=None):
        self.status_code = status
        self._key = key
        self._usage = usage or {}

    def json(self):
        return {
            "candidates": [{"content": {"role": "model", "parts": [{"text": f"ok:{self._key}"}]}}],
            "usageMetadata": self._usage,
        }

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


def _fake_client_factory(status_for_key, calls, usage=None):
    class FakeAsyncClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json, headers):
            key = headers["x-goog-api-key"]
            calls.append(key)
            return _Resp(status_for_key(key), key, usage)

    return FakeAsyncClient


@pytest.mark.asyncio
async def test_generate_returns_content_and_token_usage(monkeypatch):
    calls = []
    import app.services.llm as llm_mod

    usage = {"promptTokenCount": 10, "candidatesTokenCount": 20, "totalTokenCount": 30}
    monkeypatch.setattr(
        llm_mod.httpx, "AsyncClient", _fake_client_factory(lambda k: 200, calls, usage)
    )
    client = GeminiClient(api_keys=["k"], model="m", base_url="https://x/v1beta", timeout_s=5)
    result = await client.generate(
        system_instruction="sys",
        contents=[{"role": "user", "parts": [{"text": "hello"}]}],
        tools=None,
    )
    assert result.content["parts"][0]["text"] == "ok:k"
    assert result.tokens_in == 10
    # output = total - prompt, so thinking tokens (if any) are included
    assert result.tokens_out == 20
    assert calls == ["k"]


@pytest.mark.asyncio
async def test_tokens_out_counts_thinking_via_total_minus_prompt(monkeypatch):
    import app.services.llm as llm_mod

    # candidatesTokenCount is the visible answer only; total includes thinking.
    usage = {"promptTokenCount": 10, "candidatesTokenCount": 20, "totalTokenCount": 55}
    monkeypatch.setattr(llm_mod.httpx, "AsyncClient", _fake_client_factory(lambda k: 200, [], usage))
    client = GeminiClient(api_keys=["k"], model="m", base_url="https://x/v1beta", timeout_s=5)
    result = await client.generate(system_instruction="s", contents=[], tools=None)
    assert result.tokens_out == 45  # 55 total - 10 prompt (answer + thinking)


@pytest.mark.asyncio
async def test_generate_rotates_to_next_key_on_429(monkeypatch):
    calls = []
    import app.services.llm as llm_mod

    status = lambda k: 429 if k == "k1" else 200  # noqa: E731
    monkeypatch.setattr(llm_mod.httpx, "AsyncClient", _fake_client_factory(status, calls))
    client = GeminiClient(api_keys=["k1", "k2"], model="m", base_url="https://x/v1beta", timeout_s=5)

    result = await client.generate(system_instruction="s", contents=[], tools=None)
    assert result.content["parts"][0]["text"] == "ok:k2"
    assert calls == ["k1", "k2"]

    calls.clear()
    await client.generate(system_instruction="s", contents=[], tools=None)
    assert calls == ["k2"]  # starts from the key that worked


@pytest.mark.asyncio
async def test_generate_rotates_to_next_key_on_other_http_error(monkeypatch):
    # A non-429 failure (e.g. 500, or a revoked key returning 403) should also
    # fall through to the next key rather than surfacing immediately.
    calls = []
    import app.services.llm as llm_mod

    status = lambda k: 500 if k == "k1" else 200  # noqa: E731
    monkeypatch.setattr(llm_mod.httpx, "AsyncClient", _fake_client_factory(status, calls))
    client = GeminiClient(api_keys=["k1", "k2"], model="m", base_url="https://x/v1beta", timeout_s=5)

    result = await client.generate(system_instruction="s", contents=[], tools=None)
    assert result.content["parts"][0]["text"] == "ok:k2"
    assert calls == ["k1", "k2"]


@pytest.mark.asyncio
async def test_generate_rotates_to_next_key_on_network_error(monkeypatch):
    # A transport-level failure on one key should try the next key, not abort.
    calls = []
    import app.services.llm as llm_mod

    class FlakyThenOk:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json, headers):
            key = headers["x-goog-api-key"]
            calls.append(key)
            if key == "k1":
                raise RuntimeError("connection reset")
            return _Resp(200, key)

    monkeypatch.setattr(llm_mod.httpx, "AsyncClient", FlakyThenOk)
    client = GeminiClient(api_keys=["k1", "k2"], model="m", base_url="https://x/v1beta", timeout_s=5)

    result = await client.generate(system_instruction="s", contents=[], tools=None)
    assert result.content["parts"][0]["text"] == "ok:k2"
    assert calls == ["k1", "k2"]


@pytest.mark.asyncio
async def test_generate_raises_when_all_keys_fail(monkeypatch):
    calls = []
    import app.services.llm as llm_mod

    monkeypatch.setattr(llm_mod.httpx, "AsyncClient", _fake_client_factory(lambda k: 429, calls))
    client = GeminiClient(api_keys=["k1", "k2"], model="m", base_url="https://x/v1beta", timeout_s=5)

    with pytest.raises(LlmError):
        await client.generate(system_instruction="s", contents=[], tools=None)
    assert calls == ["k1", "k2"]


def test_settings_fallback_model_list_default_and_parsing():
    assert Settings().gemini_fallback_model_list == ["gemini-3.5-flash-lite"]
    assert Settings(gemini_fallback_models=" a , ,b ").gemini_fallback_model_list == ["a", "b"]
    assert Settings(gemini_fallback_models="").gemini_fallback_model_list == []


def test_from_settings_passes_fallback_models_and_drops_the_main_model():
    client = GeminiClient.from_settings(
        Settings(gemini_api_key="k", gemini_model="m", gemini_fallback_models="m,f")
    )
    assert client.fallback_models == ["f"]


def _fake_client_by_model(status_for, calls):
    class FakeAsyncClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json, headers):
            model = url.split("/models/")[1].split(":")[0]
            key = headers["x-goog-api-key"]
            calls.append((model, key))
            return _Resp(status_for(model, key), f"{model}/{key}")

    return FakeAsyncClient


@pytest.mark.asyncio
async def test_generate_falls_back_to_next_model_when_every_key_fails(monkeypatch):
    calls = []
    import app.services.llm as llm_mod

    status = lambda model, key: 503 if model == "main" else 200  # noqa: E731
    monkeypatch.setattr(llm_mod.httpx, "AsyncClient", _fake_client_by_model(status, calls))
    client = GeminiClient(
        api_keys=["k1", "k2"], model="main", base_url="https://x/v1beta", timeout_s=5,
        fallback_models=["backup"],
    )

    result = await client.generate(system_instruction="s", contents=[], tools=None)
    assert result.model == "backup"
    assert calls[:2] == [("main", "k1"), ("main", "k2")]  # every key on the main model first
    assert calls[2][0] == "backup"


@pytest.mark.asyncio
async def test_generate_reports_main_model_when_it_answers(monkeypatch):
    calls = []
    import app.services.llm as llm_mod

    monkeypatch.setattr(llm_mod.httpx, "AsyncClient", _fake_client_by_model(lambda m, k: 200, calls))
    client = GeminiClient(
        api_keys=["k"], model="main", base_url="https://x/v1beta", timeout_s=5,
        fallback_models=["backup"],
    )

    result = await client.generate(system_instruction="s", contents=[], tools=None)
    assert result.model == "main"
    assert calls == [("main", "k")]


@pytest.mark.asyncio
async def test_generate_raises_when_every_model_fails(monkeypatch):
    calls = []
    import app.services.llm as llm_mod

    monkeypatch.setattr(llm_mod.httpx, "AsyncClient", _fake_client_by_model(lambda m, k: 503, calls))
    client = GeminiClient(
        api_keys=["k"], model="main", base_url="https://x/v1beta", timeout_s=5,
        fallback_models=["backup"],
    )

    with pytest.raises(LlmError, match="http 503"):
        await client.generate(system_instruction="s", contents=[], tools=None)
    assert calls == [("main", "k"), ("backup", "k")]


@pytest.mark.asyncio
async def test_timeout_skips_remaining_keys_and_goes_to_fallback_model(monkeypatch):
    # A timeout is the model being slow, not the key: the other keys would wait
    # just as long, so the client moves straight to the fallback model.
    calls = []
    import app.services.llm as llm_mod

    class SlowMainModel:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json, headers):
            model = url.split("/models/")[1].split(":")[0]
            key = headers["x-goog-api-key"]
            calls.append((model, key))
            if model == "main":
                raise llm_mod.httpx.ReadTimeout("")
            return _Resp(200, f"{model}/{key}")

    monkeypatch.setattr(llm_mod.httpx, "AsyncClient", SlowMainModel)
    client = GeminiClient(
        api_keys=["k1", "k2", "k3"], model="main", base_url="https://x/v1beta", timeout_s=5,
        fallback_models=["backup"],
    )

    result = await client.generate(system_instruction="s", contents=[], tools=None)
    assert result.model == "backup"
    assert calls == [("main", "k1"), ("backup", "k1")]
