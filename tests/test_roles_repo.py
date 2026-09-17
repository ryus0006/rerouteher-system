"""roles_repo.get_rerank_texts: maps role ids to short reranking text."""
import pytest

from app.repositories import roles as roles_repo

pytestmark = pytest.mark.asyncio


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class FakeRow:
    def __init__(self, role_id, role_title, rerank_text):
        self.role_id = role_id
        self.role_title = role_title
        self.rerank_text = rerank_text


class FakeSession:
    def __init__(self, rows):
        self._rows = rows
        self.executed = []

    async def execute(self, stmt, params=None):
        self.executed.append((str(stmt), params))
        return FakeResult(self._rows)


class DistinctiveRow:
    def __init__(self, skill_id, skill_name, skill_type, importance):
        self.skill_id = skill_id
        self.skill_name = skill_name
        self.skill_type = skill_type
        self.importance = importance


async def test_get_distinctive_role_skills_threads_params_and_maps_rows():
    rows = [
        DistinctiveRow("s1", "Developing digital content", "digital", 95.0),
        DistinctiveRow("s2", "Copyright and licences", "digital", 80.0),
    ]
    session = FakeSession(rows)
    out = await roles_repo.get_distinctive_role_skills(
        session, "R04", limit=15, exclude_skill_ids=["s9"]
    )
    stmt, params = session.executed[0]
    # ranks by importance * idf, excludes given ids, bounded to the role
    assert "ln((SELECT t FROM total) / df.n)" in stmt
    assert "NOT (rs.skill_id = ANY(:excl))" in stmt
    assert params["rid"] == "R04" and params["lim"] == 15 and params["excl"] == ["s9"]
    assert [r.skill_id for r in out] == ["s1", "s2"]
    assert out[0].skill_name == "Developing digital content"


async def test_get_distinctive_role_skills_empty_exclude_defaults():
    session = FakeSession([])
    await roles_repo.get_distinctive_role_skills(session, "R04", limit=5)
    _, params = session.executed[0]
    assert params["excl"] == []


async def test_get_rerank_texts_maps_id_to_title_plus_summary():
    rows = [
        FakeRow("r1", "Data Analyst", "Data Analyst. Analyses data to inform decisions."),
        FakeRow("r2", "Operations Manager", "Operations Manager. "),
    ]
    session = FakeSession(rows)
    out = await roles_repo.get_rerank_texts(session, ["r1", "r2"])
    assert out["r1"].startswith("Data Analyst. Analyses data")
    assert out["r2"] == "Operations Manager"  # empty summary -> title only


async def test_get_rerank_texts_empty_ids_returns_empty():
    session = FakeSession([])
    assert await roles_repo.get_rerank_texts(session, []) == {}
