import pytest

from app.services import account as account_mod
from app.services.account import AccountError, AccountService

pytestmark = pytest.mark.asyncio


class FakeRepo:
    """In-memory stand-in for accounts_repo, patched at source."""

    def __init__(self):
        self.users = {}   # username -> UserRow-like
        self.plans = {}   # username -> dict
        self.seen = []

    async def get_user(self, session, username):
        return self.users.get(username)

    async def insert_user(self, session, username, password_hash, display_name):
        self.users[username] = type(
            "U", (), {"username": username, "password_hash": password_hash, "display_name": display_name}
        )()

    async def touch_last_seen(self, session, username):
        self.seen.append(username)

    async def get_plan(self, session, username):
        return self.plans.get(username)

    async def upsert_plan(self, session, username, plan):
        self.plans[username] = plan


@pytest.fixture
def repo(monkeypatch):
    fake = FakeRepo()
    monkeypatch.setattr(account_mod, "accounts_repo", fake)
    return fake


async def test_create_normalises_username_hashes_and_saves_plan(repo):
    svc = AccountService()
    out = await svc.create(
        object(), username="  Aisha  ", password="password1",
        display_name="Aisha R", plan={"snapshot": {"x": 1}},
    )
    assert out == {"username": "aisha", "display_name": "Aisha R"}
    assert repo.users["aisha"].password_hash != "password1"  # hashed
    assert repo.plans["aisha"] == {"snapshot": {"x": 1}}


async def test_create_display_name_defaults_to_username(repo):
    svc = AccountService()
    out = await svc.create(object(), username="aisha", password="password1", display_name=None, plan={})
    assert out["display_name"] == "aisha"


async def test_create_rejects_duplicate(repo):
    svc = AccountService()
    await svc.create(object(), username="aisha", password="password1", display_name=None, plan={})
    with pytest.raises(AccountError) as ei:
        await svc.create(object(), username="AISHA", password="password1", display_name=None, plan={})
    assert ei.value.kind == "conflict"


async def test_create_rejects_bad_username_and_short_password(repo):
    svc = AccountService()
    with pytest.raises(AccountError) as ei:
        await svc.create(object(), username="a", password="password1", display_name=None, plan={})
    assert ei.value.kind == "validation"
    with pytest.raises(AccountError) as ei:
        await svc.create(object(), username="aisha", password="short", display_name=None, plan={})
    assert ei.value.kind == "validation"


async def test_sign_in_returns_plan_on_correct_credentials(repo):
    svc = AccountService()
    await svc.create(object(), username="aisha", password="password1", display_name="Aisha", plan={"a": 1})
    out = await svc.sign_in(object(), username="Aisha", password="password1")
    assert out == {"username": "aisha", "display_name": "Aisha", "plan": {"a": 1}}
    assert "aisha" in repo.seen


async def test_sign_in_wrong_credentials_raises_auth(repo):
    svc = AccountService()
    await svc.create(object(), username="aisha", password="password1", display_name=None, plan={})
    with pytest.raises(AccountError) as ei:
        await svc.sign_in(object(), username="aisha", password="nope12345")
    assert ei.value.kind == "auth"
    with pytest.raises(AccountError) as ei:
        await svc.sign_in(object(), username="ghost", password="password1")
    assert ei.value.kind == "auth"
