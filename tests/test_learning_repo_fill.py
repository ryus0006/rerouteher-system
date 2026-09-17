from datetime import timedelta

from app.repositories import learning as learning_repo
from app.repositories.learning import _provider_slug
from app.services.learning_fill import ChosenResource


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows

    def first(self):
        return self._rows[0] if self._rows else None


class Row:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeSession:
    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    async def execute(self, stmt, params=None):
        self.calls.append((str(stmt), params))
        return self._results.pop(0) if self._results else FakeResult([])


def test_provider_slug():
    assert _provider_slug("Open Learn") == "web-open-learn"
    assert _provider_slug("freeCodeCamp") == "web-freecodecamp"
    assert _provider_slug("") == "web"


async def test_learning_fill_state_counts_total_ai_and_urls():
    # skill a: 1 curated + 1 ai; skill b: 1 curated only. c: absent (no rows).
    rows = [
        Row(skill_id="a", resource_id="seed-e6-x", url="https://c.com/a"),
        Row(skill_id="a", resource_id="ai-a-1", url="https://ai.com/a"),
        Row(skill_id="b", resource_id="seed-e6-y", url="https://c.com/b"),
    ]
    state = await learning_repo.learning_fill_state(FakeSession([FakeResult(rows)]), ["a", "b", "c"])
    assert state["a"].total == 2 and state["a"].ai_count == 1
    assert set(state["a"].urls) == {"https://c.com/a", "https://ai.com/a"}
    assert state["b"].total == 1 and state["b"].ai_count == 0
    assert "c" not in state


async def test_learning_fill_state_empty_returns_empty():
    assert await learning_repo.learning_fill_state(FakeSession([]), []) == {}


async def test_upsert_writes_provider_resource_link_with_existing_provider_id():
    chosen = ChosenResource(
        skill_id="s1", title="Learn SQL", url="https://ex.com/sql",
        provider_name="OpenLearn", delivery_mode="Course", duration_minutes=90,
        level="Beginner", licence_note="CC-BY", relevance=0.9, evidence_note="Builds SQL.",
    )
    session = FakeSession([FakeResult([]), FakeResult([Row(provider_id="OPENLEARN")])])
    await learning_repo.upsert_filled_resource(session, chosen)
    sql = " ".join(c[0] for c in session.calls).lower()
    assert "insert into provider" in sql
    assert "insert into learning_resource" in sql
    assert "insert into learning_resource_skill" in sql
    res_call = next(c for c in session.calls if "insert into learning_resource " in c[0].lower())
    assert res_call[1]["pid"] == "OPENLEARN"
    # id is numbered by position so a skill can hold more than one filled resource.
    assert res_call[1]["rid"] == "ai-s1-1"
    assert res_call[1]["duration"] == timedelta(minutes=90)
    link_call = next(c for c in session.calls if "insert into learning_resource_skill" in c[0].lower())
    assert link_call[1]["sid"] == "s1" and link_call[1]["rid"] == "ai-s1-1"


async def test_upsert_uses_slug_when_provider_is_new():
    chosen = ChosenResource(
        skill_id="s2", title="X", url="https://ex.com/x", provider_name="New Prov",
        delivery_mode=None, duration_minutes=None, level=None, licence_note=None,
        relevance=None, evidence_note=None,
    )
    session = FakeSession([FakeResult([]), FakeResult([Row(provider_id="web-new-prov")])])
    await learning_repo.upsert_filled_resource(session, chosen)
    res_call = next(c for c in session.calls if "insert into learning_resource " in c[0].lower())
    assert res_call[1]["pid"] == "web-new-prov"
    assert res_call[1]["rid"] == "ai-s2-1"
    assert res_call[1]["duration"] is None


async def test_upsert_numbers_ids_so_two_resources_do_not_collide():
    # Two resources of the SAME format for one skill must get distinct ids, or the
    # second would be dropped by ON CONFLICT. The index keeps them apart.
    def make(url):
        return ChosenResource(
            skill_id="s3", title="T", url=url, provider_name="P", delivery_mode="Article",
            duration_minutes=None, level=None, licence_note=None, relevance=None, evidence_note=None,
        )
    s1 = FakeSession([FakeResult([]), FakeResult([Row(provider_id="web-p")])])
    await learning_repo.upsert_filled_resource(s1, make("https://a.com/1"), 1)
    s2 = FakeSession([FakeResult([]), FakeResult([Row(provider_id="web-p")])])
    await learning_repo.upsert_filled_resource(s2, make("https://a.com/2"), 2)
    rid1 = next(c for c in s1.calls if "insert into learning_resource " in c[0].lower())[1]["rid"]
    rid2 = next(c for c in s2.calls if "insert into learning_resource " in c[0].lower())[1]["rid"]
    assert rid1 == "ai-s3-1" and rid2 == "ai-s3-2" and rid1 != rid2
