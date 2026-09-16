import httpx
import pytest

from app.services.tavily import SearchResult, TavilyError, TavilySearcher


def _searcher(handler, keys=("k",)):
    s = TavilySearcher(list(keys), "https://api.tavily.com", 10.0)
    s._transport = httpx.MockTransport(handler)  # test hook, see impl note
    return s


async def test_search_maps_results():
    def handler(request):
        assert request.url.path == "/search"
        body = request.read()
        assert b'"query"' in body and b'"max_results"' in body and b'"api_key"' in body
        return httpx.Response(200, json={"results": [
            {"title": "Learn SQL", "url": "https://ex.com/sql", "content": "a free course"},
            {"title": "No URL", "url": "", "content": "skip me"},
        ]})
    out = await _searcher(handler).search("free sql", 6)
    assert out == [SearchResult("Learn SQL", "https://ex.com/sql", "a free course")]


async def test_search_raises_tavilyerror_on_http_error():
    def handler(request):
        return httpx.Response(401, json={"error": "bad key"})
    with pytest.raises(TavilyError):
        await _searcher(handler).search("q", 6)


async def test_search_rotates_key_on_429():
    seen_keys = []

    def handler(request):
        import json
        key = json.loads(request.read())["api_key"]
        seen_keys.append(key)
        if key == "k1":
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json={"results": [
            {"title": "T", "url": "https://ex.com/x", "content": "c"},
        ]})

    out = await _searcher(handler, keys=("k1", "k2")).search("q", 6)
    assert out == [SearchResult("T", "https://ex.com/x", "c")]
    assert seen_keys == ["k1", "k2"]  # rotated from the rate-limited key to the backup


async def test_search_raises_when_all_keys_rate_limited():
    def handler(request):
        return httpx.Response(429, json={"error": "rate limited"})
    with pytest.raises(TavilyError):
        await _searcher(handler, keys=("k1", "k2")).search("q", 6)


def test_from_settings_none_without_key():
    class S:
        tavily_api_keys = []
        tavily_base_url = "https://api.tavily.com"
        tavily_timeout_s = 20.0
    assert TavilySearcher.from_settings(S()) is None


def test_from_settings_builds_with_keys():
    class S:
        tavily_api_keys = ["tvly-x", "tvly-y"]
        tavily_base_url = "https://api.tavily.com"
        tavily_timeout_s = 20.0
    assert isinstance(TavilySearcher.from_settings(S()), TavilySearcher)
