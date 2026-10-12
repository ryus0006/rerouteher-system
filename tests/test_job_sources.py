import json

import httpx
import pytest

from app.services.job_sources import (
    FounditJobSource,
    JoobleJobSource,
    TavilyGeminiJobSource,
)
from app.services.llm import GenerateResult
from app.services.tavily import SearchResult

pytestmark = pytest.mark.asyncio


def _jooble_source(handler):
    source = JoobleJobSource(
        api_key="jooble-test-key",
        base_url="https://my.jooble.org/api",
        timeout_s=5.0,
    )
    source._transport = httpx.MockTransport(handler)
    return source


def _foundit_source(handler):
    source = FounditJobSource(
        base_url="https://www.foundit.my",
        timeout_s=5.0,
    )
    source._transport = httpx.MockTransport(handler)
    return source


async def test_jooble_search_maps_structured_jobs_and_limits_results():
    def handler(request):
        assert request.url.path == "/api/jooble-test-key"
        body = json.loads(request.read())
        assert body["keywords"] == "Human Resources Officer"
        assert body["location"] == "Malaysia"
        assert body["ResultOnPage"] == 50
        return httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "title": "Human Resources Officer",
                        "company": "CIMB Group Holdings Berhad",
                        "location": "Kuala Lumpur",
                        "link": "https://my.jooble.org/jdp/123",
                    },
                    {"title": "", "company": "Missing title", "link": "https://example.test/x"},
                ]
            },
        )

    result = await _jooble_source(handler).search(
        "Human Resources Officer", "Malaysia", 50
    )

    assert len(result) == 1
    assert result[0].company_name == "CIMB Group Holdings Berhad"
    assert result[0].provider == "jooble"


async def test_jooble_rotates_to_second_configured_key_on_rate_limit():
    seen_keys = []

    def handler(request):
        key = request.url.path.rsplit("/", 1)[-1]
        seen_keys.append(key)
        if key == "jooble-test-key":
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(
            200,
            json={
                "jobs": [
                    {
                        "title": "HR Officer",
                        "company": "CIMB Group Holdings Berhad",
                        "link": "https://jobs.example.test/hr",
                    }
                ]
            },
        )

    source = JoobleJobSource(
        api_key="jooble-test-key",
        api_key_2="jooble-fallback-key",
        base_url="https://my.jooble.org/api",
        timeout_s=5.0,
    )
    source._transport = httpx.MockTransport(handler)

    result = await source.search("Human Resources Officer", "Malaysia", 50)

    assert [candidate.url for candidate in result] == ["https://jobs.example.test/hr"]
    assert seen_keys == ["jooble-test-key", "jooble-fallback-key"]


async def test_foundit_prefers_direct_apply_url_and_filters_placeholders():
    def handler(request):
        assert request.url.path == "/middleware/jobsearch"
        assert request.url.params["start"] == "0"
        assert request.url.params["limit"] == "50"
        assert request.url.params["query"] == "Human Resources Officer"
        assert request.url.params["locations"] == ""
        return httpx.Response(
            200,
            json={
                "jobSearchResponse": {
                    "data": [
                        {
                            "title": "HR Officer",
                            "companyName": "CIMB Group Holdings Berhad",
                            "locations": ["Kuala Lumpur"],
                            "applyUrl": "https://careers.example.test/apply/123",
                            "redirectUrl": "https://foundit.example.test/redirect/123",
                        },
                        {"title": "", "companyName": "", "applyUrl": ""},
                    ]
                }
            },
        )

    result = await _foundit_source(handler).search(
        "Human Resources Officer", "Malaysia", 50
    )

    assert len(result) == 1
    assert result[0].url == "https://careers.example.test/apply/123"
    assert result[0].location == "Kuala Lumpur"
    assert result[0].provider == "foundit"


async def test_tavily_gemini_source_accepts_only_search_result_urls():
    class FakeTavily:
        async def search(self, query, k):
            assert query == '"Human Resources Officer" Malaysia jobs'
            assert k == 20
            return [
                SearchResult(
                    "CIMB HR Officer",
                    "https://careers.example.test/hr-123",
                    "CIMB is hiring an HR officer.",
                )
            ]

    class FakeGemini:
        async def generate(self, *, system_instruction, contents, tools):
            return GenerateResult(
                content={
                    "parts": [
                        {
                            "text": json.dumps(
                                [
                                    {
                                        "title": "HR Officer",
                                        "company": "CIMB Group Holdings Berhad",
                                        "url": "https://careers.example.test/hr-123",
                                    },
                                    {
                                        "title": "Invented",
                                        "company": "Invented Co",
                                        "url": "https://not-in-search.example/test",
                                    },
                                ]
                            )
                        }
                    ]
                }
            )

    result = await TavilyGeminiJobSource(FakeTavily(), FakeGemini()).search(
        "Human Resources Officer", "Malaysia", 20
    )

    assert len(result) == 1
    assert result[0].title == "HR Officer"
    assert result[0].provider == "tavily"


async def test_provider_http_errors_are_normalised():
    def handler(request):
        return httpx.Response(503, json={"error": "unavailable"})

    with pytest.raises(Exception, match="jooble"):
        await _jooble_source(handler).search("q", "Malaysia", 50)
