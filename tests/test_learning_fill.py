from types import SimpleNamespace

from app.services.learning_fill import (
    ChosenResource,
    LearningFillService,
    _extract_text,
    _parse_choice,
    focus_area_skill_ids,
)
from app.services.llm import GenerateResult, LlmError
from app.services.tavily import SearchResult, TavilyError


def _gap(skill_id, band, uplift):
    return SimpleNamespace(skill_id=skill_id, band=band, uplift=uplift)


def test_focus_area_reserves_one_ai_slot_and_caps_to_limit():
    gaps = [
        _gap("r1", "role", 0.9), _gap("r2", "role", 0.8), _gap("r3", "role", 0.7),
        _gap("a1", "ai_usage", 0.5), _gap("a2", "ai_usage", 0.4),
    ]
    # 2 top role gaps + 1 top ai gap, ordered by uplift; r3 and a2 dropped.
    assert focus_area_skill_ids(gaps, 3) == ["r1", "r2", "a1"]


def test_focus_area_all_role_when_no_ai():
    gaps = [_gap("r1", "role", 0.9), _gap("r2", "role", 0.8), _gap("r3", "role", 0.7), _gap("r4", "role", 0.6)]
    assert focus_area_skill_ids(gaps, 3) == ["r1", "r2", "r3"]


def test_focus_area_backfills_when_too_few_role_gaps():
    gaps = [_gap("r1", "role", 0.9), _gap("a1", "ai_usage", 0.5), _gap("a2", "ai_usage", 0.4)]
    # role_slots=2 but only 1 role gap; top ai reserved; backfill with next ai.
    assert focus_area_skill_ids(gaps, 3) == ["r1", "a1", "a2"]


class FakeLlm:
    """Returns fixed model texts in order; records the prompts it saw."""
    def __init__(self, texts):
        self._texts = list(texts)
        self.prompts = []

    async def generate(self, *, system_instruction, contents, tools):
        self.prompts.append(contents[0]["parts"][0]["text"])
        text = self._texts.pop(0)
        if isinstance(text, Exception):
            raise text
        return GenerateResult(content={"role": "model", "parts": [{"text": text}]})


class FakeSearcher:
    def __init__(self, results, error=None):
        self._results = results
        self._error = error
        self.queries = []

    async def search(self, query, k):
        self.queries.append(query)
        if self._error:
            raise self._error
        return list(self._results)


_RESULTS = [
    SearchResult("Learn SQL", "https://good.com/sql", "a free interactive sql course"),
    SearchResult("Other", "https://other.com/x", "something else"),
]


async def _always_ok(self, url):
    return True


def _svc(llm, searcher, monkeypatch, verify=_always_ok):
    svc = LearningFillService(llm, searcher)
    monkeypatch.setattr(LearningFillService, "_verify_url", verify)
    return svc


def test_extract_text_joins_parts():
    assert _extract_text({"parts": [{"text": "a"}, {"text": "b"}]}) == "ab"
    assert _extract_text({}) == ""


def test_parse_choice_reads_json_and_rejects_bad_url():
    c = _parse_choice("s1", '{"url":"https://ex.com/x","title":"X","provider_name":"P"}')
    assert c is not None and c.skill_id == "s1"
    assert _parse_choice("s1", '{"url": null}') is None
    assert _parse_choice("s1", "not json") is None


def test_parse_choices_reads_array_and_filters_bad():
    from app.services.learning_fill import _parse_choices
    raw = ('[{"url":"https://a.com","title":"A","provider_name":"P"},'
           '{"url":null},'
           '{"url":"https://b.com","title":"B","provider_name":"P"}]')
    out = _parse_choices("s1", raw)
    assert [c.url for c in out] == ["https://a.com", "https://b.com"]


def test_parse_choices_tolerates_single_object():
    from app.services.learning_fill import _parse_choices
    out = _parse_choices("s1", '{"url":"https://a.com","title":"A","provider_name":"P"}')
    assert len(out) == 1 and out[0].url == "https://a.com"


async def test_find_resource_returns_grounded_verified_pick(monkeypatch):
    llm = FakeLlm(['{"url":"https://good.com/sql","title":"Learn SQL","provider_name":"Site","relevance":0.9}'])
    searcher = FakeSearcher(_RESULTS)
    svc = _svc(llm, searcher, monkeypatch)
    c = await svc.find_resource("s1", "SQL", "query databases")
    assert c is not None and c.url == "https://good.com/sql" and c.skill_id == "s1"
    # definition is passed into the search query and the pick prompt
    assert "query databases" in searcher.queries[0]
    assert "query databases" in llm.prompts[0]


async def test_find_resource_rejects_url_not_in_candidates(monkeypatch):
    # LLM fabricates a url not in the Tavily results -> filtered out -> nothing to walk
    llm = FakeLlm(['[{"url":"https://fabricated.com","title":"X","provider_name":"P"}]'])
    svc = _svc(llm, FakeSearcher(_RESULTS), monkeypatch)
    assert await svc.find_resource("s1", "SQL") is None


async def test_find_resource_walks_shortlist_past_dead_url(monkeypatch):
    # One ranked response with both urls; the first is dead, so walk to the second.
    # This is the failed-skill fix: no second Gemini call, just walk the shortlist.
    llm = FakeLlm([
        '[{"url":"https://good.com/sql","title":"A","provider_name":"P"},'
        ' {"url":"https://other.com/x","title":"B","provider_name":"P"}]',
    ])
    svc = LearningFillService(llm, FakeSearcher(_RESULTS))
    async def only_other(self, url):
        return url == "https://other.com/x"
    monkeypatch.setattr(LearningFillService, "_verify_url", only_other)
    c = await svc.find_resource("s1", "SQL")
    assert c is not None and c.url == "https://other.com/x"
    assert len(llm.prompts) == 1  # a single Gemini call, then walk the list


async def test_find_resource_none_when_no_results(monkeypatch):
    svc = _svc(FakeLlm([]), FakeSearcher([]), monkeypatch)
    assert await svc.find_resource("s1", "SQL") is None


async def test_find_resource_none_on_search_error(monkeypatch):
    svc = _svc(FakeLlm([]), FakeSearcher(_RESULTS, error=TavilyError("boom")), monkeypatch)
    assert await svc.find_resource("s1", "SQL") is None


async def test_find_resource_swallows_llm_error(monkeypatch):
    svc = _svc(FakeLlm([LlmError("boom"), LlmError("again")]), FakeSearcher(_RESULTS), monkeypatch)
    assert await svc.find_resource("s1", "SQL") is None


def test_disabled_without_llm_or_searcher():
    assert LearningFillService(None, FakeSearcher(_RESULTS)).enabled is False
    assert LearningFillService(FakeLlm([]), None).enabled is False
    assert LearningFillService(FakeLlm([]), FakeSearcher(_RESULTS)).enabled is True


async def test_verify_url_get_fallback_on_403_and_accepts_guarded():
    import httpx
    calls = []
    def handler(request):
        calls.append(request.method)
        return httpx.Response(403)
    svc = LearningFillService(FakeLlm([]), FakeSearcher(_RESULTS))
    svc._verify_transport = httpx.MockTransport(handler)  # test hook, see impl
    assert await svc._verify_url("https://guarded.com") is True
    assert calls == ["HEAD", "GET"]  # 403 -> GET fallback, then accepted as guarded


async def test_verify_url_rejects_404_without_get_fallback():
    import httpx
    calls = []
    def handler(request):
        calls.append(request.method)
        return httpx.Response(404)
    svc = LearningFillService(FakeLlm([]), FakeSearcher(_RESULTS))
    svc._verify_transport = httpx.MockTransport(handler)
    assert await svc._verify_url("https://missing.com") is False
    assert calls == ["HEAD"]  # tightened: no wasteful GET on a definitive 404


class _FakeDbSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def commit(self):
        pass


async def test_fill_for_skills_persists_only_missing(monkeypatch):
    import app.services.learning_fill as mod
    from app.repositories.learning import SkillLabel

    monkeypatch.setattr(mod, "SessionLocal", lambda: _FakeDbSession())

    async def fake_missing(session, ids):
        return ["s2"]
    async def fake_labels(session, ids):
        return {"s2": SkillLabel("s2", "Excel", "spreadsheets")}
    persisted = []
    async def fake_upsert(session, chosen):
        persisted.append(chosen)
    monkeypatch.setattr(mod.learning_repo, "skills_missing_resources", fake_missing)
    monkeypatch.setattr(mod.learning_repo, "get_skill_labels", fake_labels)
    monkeypatch.setattr(mod.learning_repo, "upsert_filled_resource", fake_upsert)

    svc = LearningFillService(FakeLlm([]), FakeSearcher(_RESULTS))
    async def fake_find(self, skill_id, name, definition=None):
        return ChosenResource(skill_id, "T", "https://ok.com", "P", None, None, None, None, 0.5, None)
    monkeypatch.setattr(LearningFillService, "find_resource", fake_find)

    await svc.fill_for_skills(["s1", "s2"])
    assert [c.skill_id for c in persisted] == ["s2"]


async def test_fill_for_skills_isolates_per_skill_errors(monkeypatch):
    import app.services.learning_fill as mod
    from app.repositories.learning import SkillLabel

    monkeypatch.setattr(mod, "SessionLocal", lambda: _FakeDbSession())

    async def fake_missing(session, ids):
        return ["s1", "s2"]
    async def fake_labels(session, ids):
        return {"s1": SkillLabel("s1", "A", None), "s2": SkillLabel("s2", "B", None)}
    persisted = []
    async def fake_upsert(session, chosen):
        persisted.append(chosen)
    monkeypatch.setattr(mod.learning_repo, "skills_missing_resources", fake_missing)
    monkeypatch.setattr(mod.learning_repo, "get_skill_labels", fake_labels)
    monkeypatch.setattr(mod.learning_repo, "upsert_filled_resource", fake_upsert)

    svc = LearningFillService(FakeLlm([]), FakeSearcher(_RESULTS))
    async def flaky_find(self, skill_id, name, definition=None):
        if skill_id == "s1":
            raise RuntimeError("boom")
        return ChosenResource(skill_id, "T", "https://ok.com", "P", None, None, None, None, None, None)
    monkeypatch.setattr(LearningFillService, "find_resource", flaky_find)

    await svc.fill_for_skills(["s1", "s2"])
    assert [c.skill_id for c in persisted] == ["s2"]


async def test_fill_for_skills_noop_when_disabled():
    await LearningFillService(None, None).fill_for_skills(["s1"])  # no raise, no DB
