import pytest

from app.schemas.companion import AskRequest
from app.services.companion import CompanionService
from app.services.llm import GenerateResult

pytestmark = pytest.mark.asyncio


class FakeRepo:
    def __init__(self):
        self.saved = []

    async def load_recent(self, session, session_id, limit=10):
        return []

    async def save_turn(self, session, session_id, username, role, content, tokens_in=0, tokens_out=0):
        self.saved.append(
            {
                "session_id": session_id,
                "username": username,
                "role": role,
                "content": content,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
            }
        )


class ScriptedLlm:
    """Returns queued GenerateResults, one per generate() call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def generate(self, *, system_instruction, contents, tools):
        self.calls.append({"system": system_instruction, "contents": contents, "tools": tools})
        return self._responses.pop(0)


def _text(t, tokens_in=0, tokens_out=0):
    return GenerateResult(
        content={"role": "model", "parts": [{"text": t}]},
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )


def _fn(name, args, tokens_in=0, tokens_out=0):
    return GenerateResult(
        content={"role": "model", "parts": [{"functionCall": {"name": name, "args": args}}]},
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )


async def test_plain_answer_saves_two_turns():
    repo = FakeRepo()
    svc = CompanionService(llm=ScriptedLlm([_text("Tell me about your last job.")]), repo=repo)
    resp = await svc.ask(AskRequest(question="hi", session_id="s1"), session=object(), username=None)
    assert resp.answer == "Tell me about your last job."
    assert resp.journey_update is None
    assert [r["role"] for r in repo.saved] == ["user", "assistant"]


async def test_update_profile_tool_becomes_journey_update():
    repo = FakeRepo()
    args = {
        "cv": {"raw_text": "Marketing lead 5y", "experiences": [], "skill_mentions": ["seo"]},
        "break": {"duration_years": 3, "activities": ["caregiving"]},
    }
    llm = ScriptedLlm([_fn("update_profile", args), _text("Saved - you were a Marketing Lead.")])
    svc = CompanionService(llm=llm, repo=repo)
    resp = await svc.ask(
        AskRequest(question="I was a marketing lead, 3y caregiving", session_id="s1"),
        session=object(),
        username="aisha",
    )
    assert resp.answer.startswith("Saved")
    assert resp.journey_update.cv.skill_mentions == ["seo"]
    assert resp.journey_update.break_.activities == ["caregiving"]
    assert len(llm.calls) == 2  # tool round-trip
    assert repo.saved[0]["username"] == "aisha"


async def test_token_usage_summed_across_calls_and_recorded_on_answer():
    repo = FakeRepo()
    llm = ScriptedLlm(
        [
            _fn("update_profile", {"cv": {"raw_text": "x"}}, tokens_in=100, tokens_out=20),
            _text("Saved.", tokens_in=130, tokens_out=15),
        ]
    )
    svc = CompanionService(llm=llm, repo=repo)
    await svc.ask(AskRequest(question="hi", session_id="s1"), session=object(), username=None)
    answer_row = next(r for r in repo.saved if r["role"] == "assistant")
    user_row = next(r for r in repo.saved if r["role"] == "user")
    assert answer_row["tokens_in"] == 230  # 100 + 130 across both calls
    assert answer_row["tokens_out"] == 35  # 20 + 15
    assert user_row["tokens_in"] == 0 and user_row["tokens_out"] == 0


async def test_history_is_loaded_into_contents():
    repo = FakeRepo()

    async def two_turns(session, session_id, limit=10):
        from app.repositories.companion import Turn

        return [Turn("user", "earlier q"), Turn("assistant", "earlier a")]

    repo.load_recent = two_turns
    llm = ScriptedLlm([_text("ok")])
    svc = CompanionService(llm=llm, repo=repo)
    await svc.ask(AskRequest(question="now", session_id="s1"), session=object(), username=None)
    roles = [c["role"] for c in llm.calls[0]["contents"]]
    assert roles == ["user", "model", "user"]  # history mapped + new question


async def test_no_llm_configured_degrades_gracefully():
    svc = CompanionService(llm=None, repo=FakeRepo())
    resp = await svc.ask(AskRequest(question="hi", session_id="s1"), session=object(), username=None)
    assert "not available" in resp.answer.lower()
    assert resp.journey_update is None


class BoomLlm:
    async def generate(self, *, system_instruction, contents, tools):
        from app.services.llm import LlmError

        raise LlmError("503 Service Unavailable")


async def test_llm_error_degrades_gracefully_without_raising():
    svc = CompanionService(llm=BoomLlm(), repo=FakeRepo())
    resp = await svc.ask(AskRequest(question="hi", session_id="s1"), session=object(), username=None)
    assert "trouble" in resp.answer.lower()
    assert resp.journey_update is None
