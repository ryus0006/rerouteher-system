import pytest

from app.repositories import companion as companion_repo

pytestmark = pytest.mark.asyncio


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class FakeRow:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class FakeSession:
    def __init__(self, rows):
        self._rows = rows
        self.executed = []

    async def execute(self, stmt, params=None):
        self.executed.append((str(stmt), params))
        return FakeResult(self._rows)


async def test_load_recent_returns_turns_oldest_first():
    rows = [FakeRow("user", "hi"), FakeRow("assistant", "hello")]
    out = await companion_repo.load_recent(FakeSession(rows), "s1", limit=10)
    assert [(t.role, t.content) for t in out] == [("user", "hi"), ("assistant", "hello")]


async def test_save_turn_executes_insert_with_params():
    session = FakeSession([])
    await companion_repo.save_turn(session, "s1", "aisha", "user", "hi")
    stmt, params = session.executed[0]
    assert "INSERT INTO companion_message" in stmt
    assert params["session_id"] == "s1"
    assert params["username"] == "aisha"
    assert params["role"] == "user"
    assert params["tokens_in"] == 0 and params["tokens_out"] == 0  # default


async def test_save_turn_records_token_usage():
    session = FakeSession([])
    await companion_repo.save_turn(session, "s1", None, "assistant", "hi", tokens_in=120, tokens_out=45)
    _, params = session.executed[0]
    assert params["tokens_in"] == 120
    assert params["tokens_out"] == 45
