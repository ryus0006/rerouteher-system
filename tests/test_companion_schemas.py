import pytest
from pydantic import ValidationError

from app.schemas.companion import (
    AskRequest,
    AskResponse,
    InterviewContextIn,
    JourneyUpdate,
    ProfileSkillUpdate,
    SkillChoice,
)


def test_ask_request_interview_defaults_none():
    req = AskRequest(question="hi", session_id="s1")
    assert req.interview is None


def test_interview_context_parses_feedback_items():
    req = AskRequest(
        question="How do I answer this?",
        session_id="s1",
        interview={
            "question_id": "GEN-001",
            "question_text": "Tell me about yourself.",
            "kind": "general",
            "transcript": "I led a small team.",
            "feedback_summary": "Clear and relevant.",
            "strengths": [{"title": "Relevance", "detail": "Stayed on topic."}],
            "improvements": [{"title": "Add a result", "detail": "Say what changed."}],
        },
    )
    assert req.interview.question_id == "GEN-001"
    assert req.interview.strengths[0].title == "Relevance"
    assert req.interview.improvements[0].detail == "Say what changed."


def test_interview_context_requires_question_id():
    with pytest.raises(ValidationError):
        InterviewContextIn(question_text="no id")


def test_ask_request_minimal():
    req = AskRequest(question="hi", session_id="s1")
    assert req.current_page is None
    assert req.journey.cv is None


def test_ask_request_reads_journey_cv_and_break_alias():
    req = AskRequest(
        question="hi",
        session_id="s1",
        journey={
            "cv": {"raw_text": "x", "experiences": [], "skill_mentions": []},
            "break": {"duration_years": 3, "activities": ["caregiving"]},
        },
        current_page="cv-upload",
    )
    assert req.journey.cv.raw_text == "x"
    assert req.journey.break_.duration_years == 3


def test_journey_carries_employer_matches():
    from app.schemas.companion import AskRequest

    req = AskRequest(
        question="hi",
        session_id="s1",
        journey={
            "employerMatches": [
                {"name": "Maybank", "met": ["flexible_work"], "unmet": ["parental_support"]}
            ]
        },
    )
    assert req.journey.employerMatches[0]["name"] == "Maybank"


def test_journey_carries_role_skills_offered_flag():
    from app.schemas.companion import AskRequest

    req = AskRequest(
        question="hi",
        session_id="s1",
        journey={"roleSkillsOfferedForRoleId": "R1"},
    )
    assert req.journey.roleSkillsOfferedForRoleId == "R1"


def test_ask_response_carries_skill_choices_role_id():
    from app.schemas.companion import AskResponse, SkillChoice

    resp = AskResponse(
        answer="Here are skills for that role.",
        skill_choices=[SkillChoice(skill_id="s1", skill_name="SQL")],
        skill_choices_role_id="R1",
    )
    assert resp.model_dump()["skill_choices_role_id"] == "R1"
    assert AskResponse(answer="hi").skill_choices_role_id is None


def test_ask_response_carries_skill_choices():
    from app.schemas.companion import AskResponse, SkillChoice

    resp = AskResponse(
        answer="Here are skills common for a Marketing Manager.",
        skill_choices=[SkillChoice(skill_id="s1", skill_name="Campaign Management")],
    )
    d = resp.model_dump()
    assert d["skill_choices"][0]["skill_id"] == "s1"
    assert AskResponse(answer="hi").skill_choices is None


def test_journey_in_carries_confirmed_skills():
    from app.schemas.companion import AskRequest

    req = AskRequest(
        question="hi",
        session_id="s1",
        journey={"confirmedSkills": [{"skill_id": "s1", "skill_name": "SQL"}]},
    )
    assert req.journey.confirmedSkills[0].skill_id == "s1"


def test_ask_response_carries_optional_cta():
    from app.schemas.companion import AskResponse, CtaOut

    resp = AskResponse(
        answer="SQL is a priority gap because it is core to the role.",
        sources=["Your gap result"],
        cta=CtaOut(label="Open your learning plan", to="/plan/learning"),
    )
    d = resp.model_dump()
    assert d["cta"]["to"] == "/plan/learning"
    assert AskResponse(answer="hi").cta is None


def test_ask_response_round_trip_with_journey_update():
    resp = AskResponse(
        answer="done",
        sources=["Your CV"],
        journey_update=JourneyUpdate(
            cv={"raw_text": "x", "experiences": [], "skill_mentions": []},
            break_={"duration_years": 2, "activities": []},
        ),
    )
    d = resp.model_dump(by_alias=True)
    assert d["journey_update"]["break"]["duration_years"] == 2
    assert d["answer"] == "done"


def test_ask_response_carries_skill_matches_and_profile_update():
    resp = AskResponse(
        answer="I found one matching skill.",
        skill_matches=[SkillChoice(skill_id="s1", skill_name="SQL", definition="Query data.")],
        profile_skill_update=ProfileSkillUpdate(
            action="add",
            status="added",
            skill_id="s1",
            skill="SQL",
            snapshot={"professional_skills": [{"skill_id": "s1", "skill": "SQL"}]},
        ),
    )
    data = resp.model_dump()
    assert data["skill_matches"][0]["definition"] == "Query data."
    assert data["profile_skill_update"]["status"] == "added"
    assert AskResponse(answer="hi").skill_matches is None
    assert AskResponse(answer="hi").profile_skill_update is None
