import json

import pytest

from app.repositories import accounts as repo

pytestmark = pytest.mark.asyncio


class FakeRow:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeResult:
    def __init__(self, row):
        self._row = row

    def first(self):
        return self._row


class FakeSession:
    def __init__(self, row=None):
        self._row = row
        self.calls = []

    async def execute(self, stmt, params=None):
        self.calls.append((str(stmt), params))
        return FakeResult(self._row)


async def test_get_user_maps_row():
    s = FakeSession(FakeRow(username="aisha", password_hash="h", display_name="Aisha"))
    user = await repo.get_user(s, "aisha")
    assert user.username == "aisha" and user.password_hash == "h" and user.display_name == "Aisha"


async def test_get_user_none_when_absent():
    assert await repo.get_user(FakeSession(None), "nobody") is None


async def test_insert_user_binds_params():
    s = FakeSession()
    await repo.insert_user(s, "aisha", "hash", "Aisha")
    _sql, params = s.calls[0]
    assert params == {"u": "aisha", "p": "hash", "d": "Aisha"}


async def test_upsert_plan_serialises_json():
    s = FakeSession()
    await repo.upsert_plan(s, "aisha", {"snapshot": {"x": 1}})
    _sql, params = s.calls[0]
    assert params["u"] == "aisha"
    assert json.loads(params["p"]) == {"snapshot": {"x": 1}}


async def test_get_plan_parses_json_string():
    s = FakeSession(FakeRow(plan_json=json.dumps({"a": 1})))
    assert await repo.get_plan(s, "aisha") == {"a": 1}


async def test_get_plan_none_when_absent():
    assert await repo.get_plan(FakeSession(None), "aisha") is None
