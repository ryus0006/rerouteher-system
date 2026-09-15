import pytest

from app.config import Settings
from app.services.llm import GeminiClient


def test_from_settings_returns_none_without_key():
    assert GeminiClient.from_settings(Settings(gemini_api_key="")) is None


def test_from_settings_builds_client_with_key():
    client = GeminiClient.from_settings(Settings(gemini_api_key="k", gemini_model="m"))
    assert client is not None and client.model == "m"


def test_settings_llm_configured_flag():
    assert Settings(gemini_api_key="").llm_configured is False
    assert Settings(gemini_api_key="k").llm_configured is True


@pytest.mark.asyncio
async def test_generate_posts_and_returns_candidate_content(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"candidates": [{"content": {"role": "model", "parts": [{"text": "hi"}]}}]}

        def raise_for_status(self):
            return None

    class FakeAsyncClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json, headers):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return FakeResponse()

    import app.services.llm as llm_mod

    monkeypatch.setattr(llm_mod.httpx, "AsyncClient", FakeAsyncClient)
    client = GeminiClient(api_key="k", model="m", base_url="https://x/v1beta", timeout_s=5)
    content = await client.generate(
        system_instruction="sys",
        contents=[{"role": "user", "parts": [{"text": "hello"}]}],
        tools=None,
    )
    assert content["parts"][0]["text"] == "hi"
    assert "m:generateContent" in captured["url"]
    assert captured["headers"]["x-goog-api-key"] == "k"
    assert captured["json"]["system_instruction"]["parts"][0]["text"] == "sys"
