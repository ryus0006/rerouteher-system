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
