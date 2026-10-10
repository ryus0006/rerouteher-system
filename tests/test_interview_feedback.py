"""Grounded Gemini feedback: only approved inputs reach the model, exactly one
structured tool call is required, and criterion ids/tags/titles are derived from the
database rows rather than trusted from the model output."""
import pytest

from app.repositories.interview import CriterionRow
from app.services.interview_feedback import (
    FeedbackError,
    FeedbackInput,
    InterviewFeedbackService,
)
from app.services.llm import GenerateResult, LlmError

pytestmark = pytest.mark.asyncio

_CRITERIA = [
    CriterionRow(
        criterion_id="EVAL-01", criterion="relevance_to_question",
        positive_feedback_tag="answers_the_question",
        improvement_feedback_tag="focus_on_the_question",
        prohibited_inference="Do not use accent or confidence style as a proxy for relevance.",
    ),
    CriterionRow(
        criterion_id="EVAL-02", criterion="role_connection",
        positive_feedback_tag="connects_experience_to_role",
        improvement_feedback_tag="make_role_connection_clearer",
        prohibited_inference="Do not score prestige of employer, school, credential, or network.",
    ),
    CriterionRow(
        criterion_id="EVAL-09", criterion="growth_from_adversity",
        positive_feedback_tag="shows_growth",
        improvement_feedback_tag="reflect_on_the_setback",
        prohibited_inference="Do not require disclosure of personal or family circumstances.",
    ),
]


class ScriptedLlm:
    def __init__(self, responses, model="gemini-3.1-flash-lite"):
        self._responses = list(responses)
        self.calls = []
        self.model = model

    async def generate(self, *, system_instruction, contents, tools):
        self.calls.append({"system": system_instruction, "contents": contents, "tools": tools})
        return self._responses.pop(0)


class BoomingLlm:
    def __init__(self, exc):
        self._exc = exc
        self.model = "gemini-3.1-flash-lite"

    async def generate(self, **kwargs):
        raise self._exc


def _fn(name, args, tokens_in=10, tokens_out=20):
    return GenerateResult(
        content={"role": "model", "parts": [{"functionCall": {"name": name, "args": args}}]},
        tokens_in=tokens_in, tokens_out=tokens_out,
    )


def _text(t):
    return GenerateResult(content={"role": "model", "parts": [{"text": t}]})


def _input(**over):
    base = dict(
        question_text="Tell me about a time you led a team.",
        category="teamwork_and_collaboration",
        role_title="Data Analyst",
        transcript="I led a team of five on a reporting project and we hit every deadline.",
        criteria=_CRITERIA,
        skill_labels=["Excel", "SQL"],
        interview_method_sources="OPM-STRUCTURED-INTERVIEWS; VA-PBI",
        authoring_method="authored_from_structured_behavioural_interview_patterns",
        answer_framework="STAR",
        answer_guidance="Describe the situation, your task, the action you took, and the result.",
        strong_evidence_signals="Specific context; clear personal contribution; evidenced result.",
        watch_out_for="Generic answer; unclear ownership; no result.",
        follow_up_question="What would you do differently next time?",
    )
    base.update(over)
    return FeedbackInput(**base)


def _valid_call(**over):
    base = dict(
        summary="Clear, relevant answer connected to leadership experience.",
        worked_well=[{"criterion_id": "EVAL-01", "detail": "Directly answered with a concrete example."}],
        what_to_improve=[{"criterion_id": "EVAL-02", "detail": "Could tie the project back to the target role."}],
    )
    base.update(over)
    return base


async def test_general_question_feedback_round_trip():
    llm = ScriptedLlm([_fn("submit_interview_feedback", _valid_call())])
    svc = InterviewFeedbackService(llm)
    result = await svc.evaluate(_input(role_title=None))

    assert result.summary.startswith("Clear")
    assert result.strengths[0].criterion_id == "EVAL-01"
    assert result.strengths[0].tag == "answers_the_question"
    assert result.strengths[0].title == "Relevance To Question"
    assert result.improvements[0].criterion_id == "EVAL-02"
    assert result.model == "gemini-3.1-flash-lite"
    assert result.tokens_in == 10 and result.tokens_out == 20


async def test_role_specific_question_includes_role_in_context():
    llm = ScriptedLlm([_fn("submit_interview_feedback", _valid_call())])
    svc = InterviewFeedbackService(llm)
    await svc.evaluate(_input(role_title="Data Analyst"))
    user_text = llm.calls[0]["contents"][0]["parts"][0]["text"]
    assert "Data Analyst" in user_text


async def test_career_break_question_does_not_require_personal_disclosure():
    llm = ScriptedLlm([_fn("submit_interview_feedback", _valid_call(
        worked_well=[{"criterion_id": "EVAL-09", "detail": "Reframed the break as a planning strength."}],
        what_to_improve=[],
    ))])
    svc = InterviewFeedbackService(llm)
    result = await svc.evaluate(_input(category="career_growth", role_title=None))
    assert result.strengths[0].criterion_id == "EVAL-09"
    assert "personal or family circumstances" in llm.calls[0]["system"]


async def test_return_readiness_question_round_trip():
    llm = ScriptedLlm([_fn("submit_interview_feedback", _valid_call(
        worked_well=[{"criterion_id": "EVAL-01", "detail": "Concrete first-90-days plan."}],
        what_to_improve=[],
    ))])
    svc = InterviewFeedbackService(llm)
    result = await svc.evaluate(_input(category="situational"))
    assert result.improvements == []


async def test_only_approved_grounded_inputs_reach_gemini():
    llm = ScriptedLlm([_fn("submit_interview_feedback", _valid_call())])
    svc = InterviewFeedbackService(llm)
    await svc.evaluate(_input())
    user_text = llm.calls[0]["contents"][0]["parts"][0]["text"]
    system = llm.calls[0]["system"]
    assert "I led a team of five" in user_text  # transcript
    assert "Excel" in user_text and "SQL" in user_text  # skill labels only
    assert "EVAL-01" in user_text and "EVAL-02" in user_text and "EVAL-09" in user_text
    # never a username, raw journey json, or audio reference anywhere in the call
    for blob in (user_text, system):
        assert "username" not in blob.lower()
        assert "plan_json" not in blob.lower()


async def test_gemini_receives_all_seven_question_design_context_fields():
    llm = ScriptedLlm([_fn("submit_interview_feedback", _valid_call())])
    svc = InterviewFeedbackService(llm)
    await svc.evaluate(_input())
    user_text = llm.calls[0]["contents"][0]["parts"][0]["text"]
    assert "OPM-STRUCTURED-INTERVIEWS; VA-PBI" in user_text
    assert "authored_from_structured_behavioural_interview_patterns" in user_text
    assert "STAR" in user_text
    assert "Describe the situation, your task, the action you took, and the result." in user_text
    assert "Specific context; clear personal contribution; evidenced result." in user_text
    assert "Generic answer; unclear ownership; no result." in user_text
    assert "What would you do differently next time?" in user_text


async def test_question_design_context_is_labelled_guidance_not_evidence():
    llm = ScriptedLlm([_fn("submit_interview_feedback", _valid_call())])
    svc = InterviewFeedbackService(llm)
    await svc.evaluate(_input())
    user_text = llm.calls[0]["contents"][0]["parts"][0]["text"]
    assert "not evidence about the candidate" in user_text.lower()
    assert "not required to follow" in user_text.lower()


async def test_system_prompt_forbids_quoting_sources_or_authoring_metadata():
    llm = ScriptedLlm([_fn("submit_interview_feedback", _valid_call())])
    svc = InterviewFeedbackService(llm)
    await svc.evaluate(_input())
    system = llm.calls[0]["system"]
    assert "do not quote" in system.lower()
    assert "interview sources" in system.lower() or "authoring metadata" in system.lower()


async def test_english_output_is_requested():
    llm = ScriptedLlm([_fn("submit_interview_feedback", _valid_call())])
    svc = InterviewFeedbackService(llm)
    await svc.evaluate(_input())
    assert "English" in llm.calls[0]["system"]


async def test_tags_and_titles_come_from_the_database_not_the_model():
    llm = ScriptedLlm([_fn("submit_interview_feedback", _valid_call(
        worked_well=[{"criterion_id": "EVAL-01", "detail": "x"}], what_to_improve=[],
    ))])
    svc = InterviewFeedbackService(llm)
    result = await svc.evaluate(_input())
    assert result.strengths[0].tag == "answers_the_question"  # from CriterionRow, not the model
    assert result.strengths[0].title == "Relevance To Question"


async def test_prohibited_inference_and_non_fabrication_instructions_present():
    llm = ScriptedLlm([_fn("submit_interview_feedback", _valid_call())])
    svc = InterviewFeedbackService(llm)
    await svc.evaluate(_input())
    system = llm.calls[0]["system"]
    assert "Never invent" in system
    assert "accent" in system
    assert "protected characteristics" in system
    assert "no numeric score" in system.lower()


async def test_system_prompt_runs_relevance_check_before_praise():
    llm = ScriptedLlm([_fn("submit_interview_feedback", _valid_call())])
    await InterviewFeedbackService(llm).evaluate(_input())
    system = llm.calls[0]["system"]
    assert "answers_question" in system
    assert "worked_well must be an empty list" in system
    assert "open by recognising what she did well" not in system
    params = llm.calls[0]["tools"][0]["function_declarations"][0]["parameters"]
    assert params["properties"]["answers_question"]["type"] == "boolean"
    assert "answers_question" in params["required"]


async def test_off_topic_answer_drops_any_strengths_the_model_returns():
    llm = ScriptedLlm([_fn("submit_interview_feedback", _valid_call(
        answers_question=False,
        summary="This answer did not address the question yet.",
        worked_well=[{"criterion_id": "EVAL-02", "detail": "You were very concise."}],
        what_to_improve=[{"criterion_id": "EVAL-01", "detail": "Start with one example."}],
    ))])
    result = await InterviewFeedbackService(llm).evaluate(
        _input(transcript="Sup? What's up? What's up?")
    )
    assert result.strengths == []
    assert result.improvements[0].criterion_id == "EVAL-01"


async def test_on_topic_answer_keeps_strengths():
    llm = ScriptedLlm([_fn("submit_interview_feedback", _valid_call(answers_question=True))])
    result = await InterviewFeedbackService(llm).evaluate(_input())
    assert result.strengths[0].criterion_id == "EVAL-01"


async def test_rejects_unknown_criterion_id():
    llm = ScriptedLlm([_fn("submit_interview_feedback", _valid_call(
        worked_well=[{"criterion_id": "EVAL-99", "detail": "x"}],
    ))])
    with pytest.raises(FeedbackError):
        await InterviewFeedbackService(llm).evaluate(_input())


async def test_rejects_non_applicable_criterion_id():
    llm = ScriptedLlm([_fn("submit_interview_feedback", _valid_call(
        worked_well=[{"criterion_id": "EVAL-09", "detail": "x"}],
    ))])
    with pytest.raises(FeedbackError):
        await InterviewFeedbackService(llm).evaluate(_input(criteria=_CRITERIA[:2]))


async def test_rejects_duplicate_criterion_in_same_list():
    llm = ScriptedLlm([_fn("submit_interview_feedback", _valid_call(
        worked_well=[
            {"criterion_id": "EVAL-01", "detail": "x"},
            {"criterion_id": "EVAL-01", "detail": "y"},
        ],
    ))])
    with pytest.raises(FeedbackError):
        await InterviewFeedbackService(llm).evaluate(_input())


async def test_rejects_criterion_conflicting_across_lists():
    llm = ScriptedLlm([_fn("submit_interview_feedback", _valid_call(
        worked_well=[{"criterion_id": "EVAL-01", "detail": "x"}],
        what_to_improve=[{"criterion_id": "EVAL-01", "detail": "y"}],
    ))])
    with pytest.raises(FeedbackError):
        await InterviewFeedbackService(llm).evaluate(_input())


async def test_rejects_item_missing_detail():
    llm = ScriptedLlm([_fn("submit_interview_feedback", _valid_call(
        worked_well=[{"criterion_id": "EVAL-01"}],
    ))])
    with pytest.raises(FeedbackError):
        await InterviewFeedbackService(llm).evaluate(_input())


async def test_rejects_missing_summary():
    llm = ScriptedLlm([_fn("submit_interview_feedback", _valid_call(summary=""))])
    with pytest.raises(FeedbackError):
        await InterviewFeedbackService(llm).evaluate(_input())


async def test_llm_unavailable_raises_feedback_error():
    with pytest.raises(FeedbackError):
        await InterviewFeedbackService(None).evaluate(_input())


async def test_llm_timeout_or_quota_error_raises_feedback_error():
    with pytest.raises(FeedbackError):
        await InterviewFeedbackService(BoomingLlm(LlmError("timeout"))).evaluate(_input())


async def test_malformed_plain_text_response_raises_feedback_error():
    llm = ScriptedLlm([_text("Sorry, I cannot do that.")])
    with pytest.raises(FeedbackError):
        await InterviewFeedbackService(llm).evaluate(_input())


async def test_missing_tool_call_among_multiple_calls_raises_feedback_error():
    content = {
        "role": "model",
        "parts": [
            {"functionCall": {"name": "submit_interview_feedback", "args": _valid_call()}},
            {"functionCall": {"name": "submit_interview_feedback", "args": _valid_call()}},
        ],
    }
    llm = ScriptedLlm([GenerateResult(content=content)])
    with pytest.raises(FeedbackError):
        await InterviewFeedbackService(llm).evaluate(_input())


async def test_wrong_function_name_raises_feedback_error():
    llm = ScriptedLlm([_fn("some_other_tool", {})])
    with pytest.raises(FeedbackError):
        await InterviewFeedbackService(llm).evaluate(_input())


async def test_no_applicable_criteria_raises_feedback_error():
    with pytest.raises(FeedbackError):
        await InterviewFeedbackService(ScriptedLlm([])).evaluate(_input(criteria=[]))


async def test_logs_do_not_contain_prompt_or_response_bodies(caplog):
    import logging

    caplog.set_level(logging.WARNING, logger="rerouteher")
    with pytest.raises(FeedbackError):
        await InterviewFeedbackService(BoomingLlm(LlmError("some network detail"))).evaluate(_input())
    for record in caplog.records:
        assert "I led a team of five" not in record.getMessage()
        assert "some network detail" not in record.getMessage()
