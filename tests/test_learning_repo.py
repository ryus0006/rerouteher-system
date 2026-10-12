import datetime

import pytest

from app.repositories import learning as learning_repo
from app.repositories import skills as skills_repo

pytestmark = pytest.mark.asyncio


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows

    def first(self):
        return self._rows[0] if self._rows else None


class FakeRow:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeSession:
    def __init__(self, rows):
        self._rows = rows
        self.executed = []

    async def execute(self, stmt, params=None):
        self.executed.append((str(stmt), params))
        return FakeResult(self._rows)


async def test_get_skill_labels_maps_id_to_name_and_definition():
    rows = [
        FakeRow(skill_id="s1", canonical_name="User research", definition="Talking to users."),
        FakeRow(skill_id="s2", canonical_name="Prototyping", definition=None),
    ]
    out = await learning_repo.get_skill_labels(FakeSession(rows), ["s1", "s2"])
    assert out["s1"].name == "User research"
    assert out["s1"].definition == "Talking to users."
    assert out["s2"].definition is None


async def test_get_skill_labels_empty_ids_returns_empty():
    assert await learning_repo.get_skill_labels(FakeSession([]), []) == {}


async def test_get_curated_resources_maps_rows_and_converts_duration():
    rows = [
        FakeRow(
            skill_id="s1",
            resource_id="r1",
            title="Intro to UX research",
            url="https://open.edu/x",
            delivery_mode="Course",
            duration=datetime.timedelta(minutes=90),
            licence_note="CC-BY",
            evidence_note="Covers the interview basics.",
            provider_name="OpenLearn",
        )
    ]
    out = await learning_repo.get_curated_resources(FakeSession(rows), ["s1"])
    assert len(out) == 1
    assert out[0].duration_minutes == 90
    assert out[0].provider_name == "OpenLearn"


async def test_get_curated_resources_null_duration_is_none():
    rows = [
        FakeRow(
            skill_id="s1",
            resource_id="r1",
            title="Article",
            url="https://x",
            delivery_mode="Article",
            duration=None,
            licence_note=None,
            evidence_note=None,
            provider_name="P",
        )
    ]
    out = await learning_repo.get_curated_resources(FakeSession(rows), ["s1"])
    assert out[0].duration_minutes is None


async def test_get_curated_resources_empty_ids_returns_empty():
    assert await learning_repo.get_curated_resources(FakeSession([]), []) == []


async def test_get_curated_resources_uses_resource_id_as_deterministic_tie_breaker():
    session = FakeSession([])
    await learning_repo.get_curated_resources(session, ["s1"])
    sql = session.executed[0][0]
    assert "lrs.relevance DESC NULLS LAST" in sql
    assert "lr.resource_id" in sql


async def test_get_skill_by_id_maps_canonical_taxonomy_row():
    row = FakeRow(
        skill_id="s1",
        canonical_name="SQL",
        skill_type="technical",
        definition="Working with structured data.",
    )

    out = await skills_repo.get_skill_by_id(FakeSession([row]), "s1")

    assert out.skill_id == "s1"
    assert out.canonical_name == "SQL"
    assert out.definition == "Working with structured data."


async def test_find_exact_skills_returns_canonical_rows_for_alias_query():
    row = FakeRow(
        skill_id="s1",
        canonical_name="SQL",
        skill_type="technical",
        definition="Working with structured data.",
    )
    session = FakeSession([row])

    out = await skills_repo.find_exact_skills(session, "database querying", limit=3)

    assert [item.skill_id for item in out] == ["s1"]
    assert "skill_aliases" in session.executed[0][0]
