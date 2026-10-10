"""Public contract for the E7 interview schemas: strict enums, explicit
content_expired, independent list defaults, and no leakage of internal fields."""
import pytest
from pydantic import ValidationError

from app.schemas.interview import (
    AreaOut,
    AreasOut,
    AttemptOut,
    CreateSessionRequest,
    ErrorOut,
    FeedbackItemOut,
    InterviewSetupOut,
    QuestionSlotOut,
    RoleOut,
    SessionDetailOut,
    SessionSummaryOut,
)

_NOW = "2026-10-03T10:00:00Z"


def _attempt(**over):
    base = dict(
        response_id=1,
        attempt_no=1, transcript="I led a small team.", feedback_status="ready",
        feedback_summary="Clear and relevant.",
        strengths=[FeedbackItemOut(criterion_id="EVAL-01", title="Relevance", detail="Stayed on topic.")],
        improvements=[],
        detected_language="en", duration_s=12.4, error_code=None,
        created_at=_NOW, updated_at=_NOW, content_expired=False,
    )
    base.update(over)
    return AttemptOut(**base)


def test_create_session_request_rejects_invalid_focus():
    with pytest.raises(ValidationError):
        CreateSessionRequest(role_id="R1", practice_focus="not_a_focus")


def test_session_summary_rejects_invalid_status():
    with pytest.raises(ValidationError):
        SessionSummaryOut(
            session_id="s1", role=RoleOut(role_id="R1", role_title="Data Analyst"),
            practice_focus="general", status="archived", progress=0,
            created_at=_NOW, updated_at=_NOW,
        )


def test_independent_list_defaults_do_not_share_state():
    setup_a = InterviewSetupOut(selected_role=RoleOut(role_id="R1", role_title="Data Analyst"))
    setup_b = InterviewSetupOut(selected_role=RoleOut(role_id="R2", role_title="Operations Manager"))
    setup_a.available_roles.append(RoleOut(role_id="R3", role_title="UX Designer"))
    assert setup_b.available_roles == []

    attempt_a = _attempt()
    attempt_b = _attempt(strengths=[])
    attempt_a.improvements.append(FeedbackItemOut(criterion_id="EVAL-02", title="Role fit", detail="x"))
    assert attempt_b.improvements == []


def test_attempt_carries_response_id_for_feedback_retry():
    attempt = _attempt(response_id=42)
    assert attempt.response_id == 42
    assert attempt.model_dump()["response_id"] == 42


def test_content_expired_is_required_and_explicit():
    with pytest.raises(ValidationError):
        AttemptOut(
            attempt_no=1, transcript=None, feedback_status="pending", feedback_summary=None,
            detected_language=None, duration_s=None, error_code=None,
            created_at=_NOW, updated_at=_NOW,
        )  # content_expired omitted


def test_complete_nested_session_serialisation():
    detail = SessionDetailOut(
        session_id="s1",
        role=RoleOut(role_id="R1", role_title="Data Analyst"),
        practice_focus="role_specific",
        status="active",
        created_at=_NOW,
        updated_at=_NOW,
        questions=[
            QuestionSlotOut(
                sequence_no=1, question_id="GEN-001", question_text="Tell me about yourself.",
                category="introduction_and_background", difficulty="foundation",
                role_id=None, kind="general",
                attempts=[_attempt()],
            )
        ],
    )
    dumped = detail.model_dump()
    assert dumped["questions"][0]["kind"] == "general"
    assert dumped["questions"][0]["role_id"] is None
    assert dumped["questions"][0]["attempts"][0]["feedback_status"] == "ready"
    assert dumped["questions"][0]["attempts"][0]["strengths"][0]["criterion_id"] == "EVAL-01"


def test_question_slot_carries_role_id_and_kind():
    slot = QuestionSlotOut(
        sequence_no=2, question_id="SFT-0001-01", question_text="Walk me through a tricky bug.",
        category="technical_depth", difficulty="intermediate",
        role_id="SFT-0001", kind="role_specific", attempts=[],
    )
    assert slot.role_id == "SFT-0001"
    assert slot.kind == "role_specific"


def test_question_slot_kind_rejects_unknown_value():
    with pytest.raises(ValidationError):
        QuestionSlotOut(
            sequence_no=1, question_id="GEN-001", question_text="Tell me about yourself.",
            category="introduction_and_background", difficulty="foundation",
            role_id=None, kind="warmup", attempts=[],
        )


def test_areas_out_separates_improvements_and_strengths():
    areas = AreasOut(
        improvements=[AreaOut(criterion_id="EVAL-07", title="Problem Solving", response_count=3)],
        strengths=[AreaOut(criterion_id="EVAL-01", title="Relevance", response_count=5)],
    )
    assert areas.improvements[0].response_count == 3
    assert areas.strengths[0].title == "Relevance"


def test_error_out_shape():
    err = ErrorOut(error="interview_resource_not_found")
    assert err.model_dump() == {"error": "interview_resource_not_found"}


def test_public_models_do_not_expose_internal_fields():
    for model_cls in (
        RoleOut, InterviewSetupOut, CreateSessionRequest, SessionSummaryOut,
        FeedbackItemOut, AttemptOut, QuestionSlotOut, SessionDetailOut, AreaOut,
        AreasOut, ErrorOut,
    ):
        forbidden = {"username", "audio", "audio_bytes", "prompt", "journey_json", "plan_json"}
        assert forbidden.isdisjoint(model_cls.model_fields.keys()), model_cls
