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


async def test_skills_missing_resources_preserves_order_and_filters_present():
    session = FakeSession([FakeResult([Row(skill_id="b")])])
    out = await learning_repo.skills_missing_resources(session, ["a", "b", "c"])
    assert out == ["a", "c"]


async def test_skills_missing_resources_empty_returns_empty():
    assert await learning_repo.skills_missing_resources(FakeSession([]), []) == []


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
    assert res_call[1]["rid"] == "ai-s1"
    assert res_call[1]["duration"] == timedelta(minutes=90)
    link_call = next(c for c in session.calls if "insert into learning_resource_skill" in c[0].lower())
    assert link_call[1]["sid"] == "s1" and link_call[1]["rid"] == "ai-s1"


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
    assert res_call[1]["duration"] is None
