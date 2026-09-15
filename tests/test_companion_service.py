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


async def test_update_profile_captures_and_filters_priorities():
    repo = FakeRepo()
    args = {
        "cv": {"raw_text": "HR officer", "experiences": [], "skill_mentions": []},
        "employerPriorities": [
            "flexible_work",
            "not_a_real_id",
            "childcare_support",
            "parental_support",
            "returning_to_work",
        ],
    }
    llm = ScriptedLlm([_fn("update_profile", args), _text("Saved.")])
    svc = CompanionService(llm=llm, repo=repo)
    resp = await svc.ask(
        AskRequest(question="I want flexible work and childcare", session_id="s1"),
        session=object(),
        username=None,
    )
    # invalid id dropped, order preserved, capped at 3
    assert resp.journey_update.employerPriorities == [
        "flexible_work",
        "childcare_support",
        "parental_support",
    ]


async def test_update_profile_without_priorities_defaults_empty():
    repo = FakeRepo()
    args = {"cv": {"raw_text": "x", "experiences": [], "skill_mentions": []}}
    llm = ScriptedLlm([_fn("update_profile", args), _text("Saved.")])
    svc = CompanionService(llm=llm, repo=repo)
    resp = await svc.ask(AskRequest(question="hi", session_id="s1"), session=object(), username=None)
    assert resp.journey_update.employerPriorities == []


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


def _fn_step(step, tokens_in=0, tokens_out=0):
    return GenerateResult(
        content={
            "role": "model",
            "parts": [{"functionCall": {"name": "point_to_step", "args": {"step": step}}}],
        },
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )


async def test_point_to_step_returns_learning_cta():
    repo = FakeRepo()
    llm = ScriptedLlm([_fn_step("learning"), _text("SQL is a priority gap; here is your plan.")])
    svc = CompanionService(llm=llm, repo=repo)
    resp = await svc.ask(
        AskRequest(question="tell me about my SQL gap", session_id="s1"),
        session=object(),
        username=None,
    )
    assert resp.cta is not None
    assert resp.cta.to == "/plan/learning"
    assert resp.answer.startswith("SQL")
    assert len(llm.calls) == 2  # tool round-trip


async def test_unknown_step_yields_no_cta():
    repo = FakeRepo()
    llm = ScriptedLlm([_fn_step("nonsense"), _text("Here is some help.")])
    svc = CompanionService(llm=llm, repo=repo)
    resp = await svc.ask(AskRequest(question="hi", session_id="s1"), session=object(), username=None)
    assert resp.cta is None


async def test_results_summary_is_injected_when_present():
    repo = FakeRepo()
    llm = ScriptedLlm([_text("You are 60% ready.")])
    svc = CompanionService(llm=llm, repo=repo)
    await svc.ask(
        AskRequest(
            question="what is my readiness?",
            session_id="s1",
            journey={
                "gapResult": {"readiness": 60, "gaps": [{"skill": "SQL", "uplift": 9}]},
                "selectedRole": {"role": "Data Analyst", "role_id": "R1"},
                "snapshot": {
                    "professional_skills": [{"skill": "Excel"}],
                    "reframed_skills": [{"skill": "Team coordination"}],
                    "previous_occupation": {"role": "Marketing Lead"},
                    "recommended_roles": [{"role": "Data Analyst"}, {"role": "Business Analyst"}],
                },
            },
        ),
        session=object(),
        username=None,
    )
    system = llm.calls[0]["system"]
    assert "60" in system and "Data Analyst" in system and "SQL" in system
    # amendment: snapshot skills + previous occupation are grounded too (8.2.1 skills)
    assert "Excel" in system and "Marketing Lead" in system


async def test_complete_profile_note_tells_it_not_to_push_updates():
    repo = FakeRepo()
    llm = ScriptedLlm([_text("Want to see your snapshot?")])
    svc = CompanionService(llm=llm, repo=repo)
    await svc.ask(
        AskRequest(
            question="what next?",
            session_id="s1",
            journey={
                "cv": {"raw_text": "x", "experiences": [], "skill_mentions": []},
                "break": {"duration_years": 2, "activities": ["caregiving"]},
            },
        ),
        session=object(),
        username=None,
    )
    system = llm.calls[0]["system"].lower()
    assert "already complete" in system
    assert "do not push" in system


async def test_incomplete_profile_has_no_complete_note():
    repo = FakeRepo()
    llm = ScriptedLlm([_text("Tell me about your break.")])
    svc = CompanionService(llm=llm, repo=repo)
    await svc.ask(
        AskRequest(
            question="hi",
            session_id="s1",
            journey={"cv": {"raw_text": "x", "experiences": [], "skill_mentions": []}},
        ),
        session=object(),
        username=None,
    )
    assert "already complete" not in llm.calls[0]["system"].lower()


async def test_no_results_note_when_journey_empty():
    repo = FakeRepo()
    llm = ScriptedLlm([_text("Tell me about your last job.")])
    svc = CompanionService(llm=llm, repo=repo)
    await svc.ask(AskRequest(question="hi", session_id="s1"), session=object(), username=None)
    system = llm.calls[0]["system"]
    assert "readiness is" not in system.lower()


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
