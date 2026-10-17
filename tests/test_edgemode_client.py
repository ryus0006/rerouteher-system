import json

import httpx
import pytest

from app.config import Settings
from app.services.llm import (
    EdgemodeClient,
    GeminiClient,
    LlmError,
    _to_gemini_content,
    _to_openai_messages,
    _to_openai_tools,
    create_llm_client,
)

TOOLS = [
    {
        "function_declarations": [
            {"name": "get_gap", "description": "d", "parameters": {"type": "object", "properties": {}}},
            {"name": "save", "description": "d2", "parameters": {"type": "object", "properties": {}}},
        ]
    }
]


def test_create_llm_client_defaults_to_gemini():
    client = create_llm_client(Settings(gemini_api_key="g", edgemode_api_key="e"))
    assert isinstance(client, GeminiClient)


def test_create_llm_client_builds_edgemode():
    client = create_llm_client(Settings(llm_provider="Edgemode", edgemode_api_key="e", edgemode_model="m"))
    assert isinstance(client, EdgemodeClient) and client.model == "m"


def test_create_llm_client_none_without_edgemode_key():
    assert create_llm_client(Settings(llm_provider="edgemode", edgemode_api_key="", gemini_api_key="g")) is None


def test_create_llm_client_rejects_unknown_provider():
    with pytest.raises(ValueError):
        create_llm_client(Settings(llm_provider="groq"))


def test_llm_configured_follows_provider():
    assert Settings(llm_provider="edgemode", edgemode_api_key="", gemini_api_key="g").llm_configured is False
    assert Settings(llm_provider="edgemode", edgemode_api_key="e").llm_configured is True


def test_tools_flatten_to_openai_functions():
    out = _to_openai_tools(TOOLS)
    assert [t["type"] for t in out] == ["function", "function"]
    assert [t["function"]["name"] for t in out] == ["get_gap", "save"]
    assert _to_openai_tools(None) == []


def test_messages_convert_system_text_and_roles():
    msgs = _to_openai_messages(
        "sys",
        [
            {"role": "user", "parts": [{"text": "hi"}]},
            {"role": "model", "parts": [{"text": "hello"}]},
            {"role": "user", "parts": [{"text": "a"}, {"text": "b"}]},
        ],
    )
    assert msgs == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "ab"},
    ]


def test_tool_calls_and_responses_are_paired_by_id():
    msgs = _to_openai_messages(
        "",
        [
            {"role": "user", "parts": [{"text": "q"}]},
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "get_gap", "args": {"role": "x"}}},
                    {"functionCall": {"name": "save", "args": {}}},
                ],
            },
            # responses arrive in a different order than the calls
            {
                "role": "user",
                "parts": [
                    {"functionResponse": {"name": "save", "response": {"status": "ok"}}},
                    {"functionResponse": {"name": "get_gap", "response": {"missing": ["sql"]}}},
                ],
            },
        ],
    )
    assistant = msgs[1]
    assert assistant["role"] == "assistant" and assistant["content"] is None
    ids = {c["function"]["name"]: c["id"] for c in assistant["tool_calls"]}
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"role": "x"}
    assert msgs[2] == {"role": "tool", "tool_call_id": ids["save"], "content": '{"status": "ok"}'}
    assert msgs[3]["tool_call_id"] == ids["get_gap"]
    assert len(msgs) == 4  # no empty user message for a tool-only turn


def test_response_converts_text_and_tool_calls():
    content = _to_gemini_content(
        {
            "content": "<think>plan</think>\nSure.",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_gap", "arguments": '{"role": "x"}'}}],
        }
    )
    assert content == {
        "role": "model",
        "parts": [{"text": "Sure."}, {"functionCall": {"name": "get_gap", "args": {"role": "x"}}}],
    }


def test_response_with_bad_tool_arguments_raises():
    with pytest.raises(LlmError):
        _to_gemini_content({"tool_calls": [{"function": {"name": "f", "arguments": "{not json"}}]})


class _Resp:
    def __init__(self, status, data=None):
        self.status_code = status
        self._data = data

    def json(self):
        return self._data


def _fake_client(response=None, exc=None, seen=None):
    class FakeAsyncClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json, headers):
            if seen is not None:
                seen.update(url=url, body=json, headers=headers)
            if exc is not None:
                raise exc
            return response

    return FakeAsyncClient


@pytest.mark.asyncio
async def test_generate_sends_openai_body_and_returns_gemini_shape(monkeypatch):
    import app.services.llm as llm_mod

    seen: dict = {}
    data = {
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 16, "total_tokens": 27},
    }
    monkeypatch.setattr(llm_mod.httpx, "AsyncClient", _fake_client(_Resp(200, data), seen=seen))
    client = EdgemodeClient("key", "qwen3-235b", "https://e/v1/", 5)
    result = await client.generate(
        system_instruction="sys", contents=[{"role": "user", "parts": [{"text": "hi"}]}], tools=TOOLS
    )
    assert seen["url"] == "https://e/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer key"
    assert seen["body"]["model"] == "qwen3-235b"
    assert [t["function"]["name"] for t in seen["body"]["tools"]] == ["get_gap", "save"]
    assert result.content == {"role": "model", "parts": [{"text": "ok"}]}
    assert (result.tokens_in, result.tokens_out, result.model) == (11, 16, "qwen3-235b")


@pytest.mark.asyncio
async def test_generate_omits_tools_when_none(monkeypatch):
    import app.services.llm as llm_mod

    seen: dict = {}
    data = {"choices": [{"message": {"content": "ok"}}]}
    monkeypatch.setattr(llm_mod.httpx, "AsyncClient", _fake_client(_Resp(200, data), seen=seen))
    await EdgemodeClient("k", "m", "https://e/v1", 5).generate(system_instruction="s", contents=[], tools=None)
    assert "tools" not in seen["body"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response,exc",
    [
        (_Resp(503, {}), None),
        (_Resp(200, {"choices": []}), None),
        (None, httpx.ConnectTimeout("timed out")),
    ],
)
async def test_generate_failures_raise_llm_error(monkeypatch, response, exc):
    import app.services.llm as llm_mod

    monkeypatch.setattr(llm_mod.httpx, "AsyncClient", _fake_client(response, exc))
    with pytest.raises(LlmError):
        await EdgemodeClient("k", "m", "https://e/v1", 5).generate(system_instruction="s", contents=[], tools=None)


@pytest.mark.parametrize("raw", ['["a", "b"]', '"text"', "42"])
def test_response_with_non_object_tool_arguments_raises(raw):
    with pytest.raises(LlmError):
        _to_gemini_content({"tool_calls": [{"function": {"name": "f", "arguments": raw}}]})


@pytest.mark.parametrize(
    "text,expected",
    [
        ("<THINK>secret</THINK>Answer", [{"text": "Answer"}]),
        ("< think >secret</ think > Answer", [{"text": "Answer"}]),
        ("<think>cut off before the closing tag", []),
        ("Answer <think>trailing reasoning", [{"text": "Answer"}]),
    ],
)
def test_reasoning_is_stripped_even_when_unclosed(text, expected):
    assert _to_gemini_content({"content": text})["parts"] == expected
