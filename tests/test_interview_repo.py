"""Interview repository: ownership-scoped SQL, no commits.

FakeSession returns one canned result per execute() call, consumed in call order, so
tests can drive a repo function that issues several statements in sequence.
"""
import json

import pytest

from app.repositories import interview as repo

pytestmark = pytest.mark.asyncio


class FakeRow:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeResult:
    def __init__(self, rows=None, row=None, rowcount=0):
        self._rows = rows
        self._row = row
        self.rowcount = rowcount

    def all(self):
        return self._rows if self._rows is not None else []

    def first(self):
        return self._row


class FakeSession:
    """Each call to execute() pops the next queued FakeResult."""

    def __init__(self, results: list[FakeResult]):
        self._results = list(results)
        self.calls: list[tuple[str, dict | None]] = []

    async def execute(self, stmt, params=None):
        self.calls.append((str(stmt), params))
        return self._results.pop(0)


def _question_row(**over):
    base = dict(
        question_id="GEN-001", role_id=None, category="introduction_and_background",
        difficulty="foundation", question_text="Tell me about yourself.",
        interview_method_sources="OPM-STRUCTURED-INTERVIEWS; VA-PBI",
        authoring_method="authored_from_structured_behavioural_interview_patterns",
        answer_framework="concise-summary",
        answer_guidance="State your current professional focus and one evidence point.",
        strong_evidence_signals="Specific context; clear personal contribution.",
        watch_out_for="Generic answer; unclear ownership.",
        follow_up_question="Which part of your background would help you contribute most quickly?",
    )
    base.update(over)
    return FakeRow(**base)


def _criterion_row(**over):
    base = dict(
        criterion_id="EVAL-01", criterion="relevance_to_question",
        positive_feedback_tag="answers_the_question",
        improvement_feedback_tag="focus_on_the_question",
        prohibited_inference="Do not use accent as a proxy for relevance.",
    )
    base.update(over)
    return FakeRow(**base)


def _attempt_row(**over):
    base = dict(
        response_id=1, session_id="s1", question_id="GEN-001", sequence_no=1, attempt_no=1,
        question_text="Tell me about yourself.", category="introduction_and_background",
        difficulty="foundation", role_id=None,
        interview_method_sources="OPM-STRUCTURED-INTERVIEWS; VA-PBI",
        authoring_method="authored_from_structured_behavioural_interview_patterns",
        answer_framework="concise-summary",
        answer_guidance="State your current professional focus and one evidence point.",
        strong_evidence_signals="Specific context; clear personal contribution.",
        watch_out_for="Generic answer; unclear ownership.",
        follow_up_question="Which part of your background would help you contribute most quickly?",
        transcript=None, detected_language=None, audio_duration_s=None,
        feedback_status="pending", feedback_summary=None,
        worked_well="[]", what_to_improve="[]",
        strength_tags=[], improvement_tags=[],
        error_code=None, created_at="2026-10-01T00:00:00Z", updated_at="2026-10-01T00:00:00Z",
        content_purged_at=None,
    )
    base.update(over)
    return FakeRow(**base)


# -- roles -------------------------------------------------------------------

async def test_get_roles_by_ids_maps_rows():
    rows = [FakeRow(role_id="R1", role_title="Data Analyst")]
    session = FakeSession([FakeResult(rows=rows)])
    out = await repo.get_roles_by_ids(session, ["R1"])
    assert out[0].role_id == "R1" and out[0].role_title == "Data Analyst"


async def test_get_roles_by_ids_empty_returns_empty_without_query():
    session = FakeSession([])
    assert await repo.get_roles_by_ids(session, []) == []
    assert session.calls == []


# -- question pool / criteria --------------------------------------------------

async def test_get_question_pool_does_not_filter_on_review_status():
    rows = [_question_row(), _question_row(question_id="M31590201", role_id="R1")]
    session = FakeSession([FakeResult(rows=rows)])
    out = await repo.get_question_pool(session, "R1")
    stmt, params = session.calls[0]
    assert "review_status" not in stmt
    assert params["rid"] == "R1"
    assert {q.question_id for q in out} == {"GEN-001", "M31590201"}


async def test_get_question_pool_carries_question_design_context():
    session = FakeSession([FakeResult(rows=[_question_row()])])
    out = await repo.get_question_pool(session, "R1")
    q = out[0]
    assert q.answer_framework == "concise-summary"
    assert q.strong_evidence_signals == "Specific context; clear personal contribution."
    assert q.watch_out_for == "Generic answer; unclear ownership."
    assert q.follow_up_question == "Which part of your background would help you contribute most quickly?"
    assert q.interview_method_sources == "OPM-STRUCTURED-INTERVIEWS; VA-PBI"
    assert q.authoring_method == "authored_from_structured_behavioural_interview_patterns"
    assert q.answer_guidance == "State your current professional focus and one evidence point."


async def test_get_criteria_for_questions_groups_by_question_id():
    rows = [
        FakeRow(question_id="GEN-001", **vars(_criterion_row())),
        FakeRow(question_id="GEN-001", **vars(_criterion_row(criterion_id="EVAL-05", criterion="motivation"))),
        FakeRow(question_id="GEN-002", **vars(_criterion_row())),
    ]
    session = FakeSession([FakeResult(rows=rows)])
    out = await repo.get_criteria_for_questions(session, ["GEN-001", "GEN-002"])
    assert {c.criterion_id for c in out["GEN-001"]} == {"EVAL-01", "EVAL-05"}
    assert len(out["GEN-002"]) == 1


async def test_get_criteria_for_questions_empty_returns_empty():
    session = FakeSession([])
    assert await repo.get_criteria_for_questions(session, []) == {}
    assert session.calls == []


# -- session setup -------------------------------------------------------------

async def test_find_session_returns_none_when_absent():
    session = FakeSession([FakeResult(row=None)])
    assert await repo.find_session(session, "aisha", "R1", "general") is None


async def test_find_session_is_scoped_to_username_role_and_focus():
    row = FakeRow(
        session_id="s1", username="aisha", role_id="R1", practice_focus="general",
        status="active", created_at="t1", updated_at="t1",
    )
    session = FakeSession([FakeResult(row=row)])
    out = await repo.find_session(session, "aisha", "R1", "general")
    stmt, params = session.calls[0]
    assert "username = :u" in stmt and "role_id = :rid" in stmt and "practice_focus = :focus" in stmt
    assert out.session_id == "s1"


async def test_insert_session_returns_new_session_row():
    row = FakeRow(session_id="new-id", status="active", created_at="t1", updated_at="t1")
    session = FakeSession([FakeResult(row=row)])
    out = await repo.insert_session(session, "aisha", "R1", "mixed")
    stmt, params = session.calls[0]
    assert "RETURNING" in stmt
    assert params == {"u": "aisha", "rid": "R1", "focus": "mixed"}
    assert out.session_id == "new-id"


async def test_insert_placeholder_responses_inserts_five_rows_in_order():
    session = FakeSession([FakeResult(rowcount=5)])
    await repo.insert_placeholder_responses(
        session, "s1", ["GEN-001", "GEN-002", "M1", "M2", "M3"]
    )
    _stmt, params = session.calls[0]
    assert len(params) == 5
    assert [p["seq"] for p in params] == [1, 2, 3, 4, 5]
    assert all(p["sid"] == "s1" for p in params)
    assert params[0]["qid"] == "GEN-001"


# -- listing / detail -----------------------------------------------------------

async def test_list_sessions_scoped_to_username_newest_first():
    rows = [
        FakeRow(
            session_id="s2", role_id="R1", role_title="Data Analyst", practice_focus="general",
            status="active", created_at="t2", updated_at="t2", progress=1,
        ),
        FakeRow(
            session_id="s1", role_id="R1", role_title="Data Analyst", practice_focus="mixed",
            status="completed", created_at="t1", updated_at="t1", progress=5,
        ),
    ]
    session = FakeSession([FakeResult(rows=rows)])
    out = await repo.list_sessions(session, "aisha")
    stmt, params = session.calls[0]
    assert "username = :u" in stmt
    assert params == {"u": "aisha"}
    assert [s.session_id for s in out] == ["s2", "s1"]
    assert out[0].progress == 1


async def test_get_session_detail_none_when_not_owned_or_missing():
    session = FakeSession([FakeResult(row=None)])
    assert await repo.get_session_detail(session, "aisha", "missing") is None
    assert len(session.calls) == 1  # second query never runs


async def test_get_session_detail_groups_attempts_under_their_question_slot():
    session_row = FakeRow(
        session_id="s1", role_id="R1", role_title="Data Analyst", practice_focus="general",
        status="active", created_at="t1", updated_at="t1",
    )
    attempt_rows = [
        _attempt_row(response_id=1, sequence_no=1, attempt_no=1),
        _attempt_row(response_id=2, sequence_no=1, attempt_no=2, feedback_status="ready"),
        _attempt_row(response_id=3, sequence_no=2, attempt_no=1, question_id="GEN-002",
                     question_text="What interests you?"),
    ]
    session = FakeSession([FakeResult(row=session_row), FakeResult(rows=attempt_rows)])
    detail = await repo.get_session_detail(session, "aisha", "s1")
    assert detail.session_id == "s1"
    assert detail.role_title == "Data Analyst"
    by_seq = {q.sequence_no: q for q in detail.questions}
    assert len(by_seq[1].attempts) == 2
    assert [a.attempt_no for a in by_seq[1].attempts] == [1, 2]
    assert len(by_seq[2].attempts) == 1
    assert by_seq[2].question_id == "GEN-002"


async def test_get_session_detail_slots_carry_question_design_context():
    session_row = FakeRow(
        session_id="s1", role_id="R1", role_title="Data Analyst", practice_focus="general",
        status="active", created_at="t1", updated_at="t1",
    )
    session = FakeSession([FakeResult(row=session_row), FakeResult(rows=[_attempt_row()])])
    detail = await repo.get_session_detail(session, "aisha", "s1")
    slot = detail.questions[0]
    assert slot.answer_framework == "concise-summary"
    assert slot.strong_evidence_signals == "Specific context; clear personal contribution."
    assert slot.watch_out_for == "Generic answer; unclear ownership."
    assert slot.follow_up_question == "Which part of your background would help you contribute most quickly?"
    assert slot.interview_method_sources == "OPM-STRUCTURED-INTERVIEWS; VA-PBI"
    assert slot.authoring_method == "authored_from_structured_behavioural_interview_patterns"
    assert slot.answer_guidance == "State your current professional focus and one evidence point."


async def test_get_session_detail_slots_carry_role_id():
    session_row = FakeRow(
        session_id="s1", role_id="R1", role_title="Data Analyst", practice_focus="mixed",
        status="active", created_at="t1", updated_at="t1",
    )
    attempt_rows = [
        _attempt_row(response_id=1, sequence_no=1, question_id="GEN-001", role_id=None),
        _attempt_row(response_id=2, sequence_no=2, question_id="R1-int-0", role_id="R1"),
    ]
    session = FakeSession([FakeResult(row=session_row), FakeResult(rows=attempt_rows)])
    detail = await repo.get_session_detail(session, "aisha", "s1")
    by_seq = {q.sequence_no: q for q in detail.questions}
    assert by_seq[1].role_id is None
    assert by_seq[2].role_id == "R1"


# -- attempt lifecycle ------------------------------------------------------------

async def test_lock_latest_attempt_scoped_to_username_and_uses_for_update():
    row = _attempt_row()
    session = FakeSession([FakeResult(row=row)])
    out = await repo.lock_latest_attempt(session, "aisha", "s1", 1)
    stmt, params = session.calls[0]
    assert "FOR UPDATE" in stmt
    assert "username = :u" in stmt
    assert out.response_id == 1


async def test_lock_latest_attempt_none_when_not_owned():
    session = FakeSession([FakeResult(row=None)])
    assert await repo.lock_latest_attempt(session, "aisha", "s1", 1) is None


async def test_insert_attempt_returns_new_response_id():
    session = FakeSession([FakeResult(row=FakeRow(response_id=42))])
    response_id = await repo.insert_attempt(session, "s1", "GEN-001", 1, 2)
    _stmt, params = session.calls[0]
    assert params == {"sid": "s1", "qid": "GEN-001", "seq": 1, "attempt_no": 2}
    assert response_id == 42


async def test_save_transcript_processing_sets_status_and_fields():
    session = FakeSession([FakeResult(rowcount=1)])
    await repo.save_transcript_processing(session, 1, "I led a team.", "en", 12.4)
    _stmt, params = session.calls[0]
    assert params == {"rid": 1, "t": "I led a team.", "lang": "en", "dur": 12.4}


async def test_save_feedback_ready_serialises_json_fields():
    session = FakeSession([FakeResult(rowcount=1)])
    await repo.save_feedback_ready(
        session, 1, "Clear answer.",
        [{"criterion_id": "EVAL-01", "title": "Relevance", "detail": "On topic."}],
        [],
        ["answers_the_question"], [],
        {"model": "gemini-3.1-flash-lite"},
    )
    _stmt, params = session.calls[0]
    assert json.loads(params["ww"])[0]["criterion_id"] == "EVAL-01"
    assert json.loads(params["wti"]) == []
    assert params["stags"] == ["answers_the_question"]
    assert json.loads(params["meta"])["model"] == "gemini-3.1-flash-lite"


async def test_save_feedback_error_sets_error_code():
    session = FakeSession([FakeResult(rowcount=1)])
    await repo.save_feedback_error(session, 1, "feedback_service_unavailable")
    _stmt, params = session.calls[0]
    assert params == {"rid": 1, "code": "feedback_service_unavailable"}


async def test_mark_completed_if_all_ready_is_a_single_statement():
    session = FakeSession([FakeResult(rowcount=1)])
    await repo.mark_completed_if_all_ready(session, "s1")
    stmt, params = session.calls[0]
    assert "completed" in stmt
    assert params == {"sid": "s1"}


async def test_delete_session_scoped_to_owner_returns_true_on_success():
    session = FakeSession([FakeResult(rowcount=1)])
    assert await repo.delete_session(session, "aisha", "s1") is True
    _stmt, params = session.calls[0]
    assert params == {"u": "aisha", "sid": "s1"}


async def test_delete_session_returns_false_when_not_owned():
    session = FakeSession([FakeResult(rowcount=0)])
    assert await repo.delete_session(session, "aisha", "s1") is False


async def test_get_attempt_by_id_scoped_to_username():
    session = FakeSession([FakeResult(row=_attempt_row())])
    out = await repo.get_attempt_by_id(session, "aisha", 1)
    stmt, params = session.calls[0]
    assert "username = :u" in stmt
    assert out.response_id == 1


async def test_get_attempt_by_id_none_when_not_owned():
    session = FakeSession([FakeResult(row=None)])
    assert await repo.get_attempt_by_id(session, "aisha", 1) is None


# -- aggregation / retention --------------------------------------------------

async def test_get_recurring_areas_splits_strength_and_improvement_kinds():
    rows = [
        FakeRow(kind="strength", criterion_id="EVAL-01", title="Relevance To Question", response_count=5),
        FakeRow(kind="improvement", criterion_id="EVAL-07", title="Problem Solving", response_count=2),
    ]
    session = FakeSession([FakeResult(rows=rows)])
    out = await repo.get_recurring_areas(session, "aisha")
    kinds = {a.kind for a in out}
    assert kinds == {"strength", "improvement"}
    _stmt, params = session.calls[0]
    assert params == {"u": "aisha"}


async def test_purge_expired_content_returns_count_and_uses_cutoff():
    session = FakeSession([FakeResult(rowcount=3)])
    count = await repo.purge_expired_content(session, "2026-09-03T00:00:00Z")
    assert count == 3
    _stmt, params = session.calls[0]
    assert params == {"cutoff": "2026-09-03T00:00:00Z"}


async def test_repository_functions_never_commit():
    """Repositories must not call session.commit(); the route owns the transaction."""
    import inspect

    for name, fn in vars(repo).items():
        if inspect.iscoroutinefunction(fn):
            src = inspect.getsource(fn)
            assert "commit" not in src, f"{name} must not commit"


async def test_get_question_coaching_returns_fields_or_none():
    row = FakeRow(
        question_id="GEN-001", role_id=None, question_text="Tell me about yourself.",
        answer_framework="concise-summary",
        answer_guidance="State your current focus and one evidence point.",
        strong_evidence_signals="Specific context; clear personal contribution.",
        watch_out_for="Generic answer; unclear ownership.",
        follow_up_question="Which part of your background helps most?",
    )
    session = FakeSession([FakeResult(row=row)])
    out = await repo.get_question_coaching(session, "GEN-001")
    assert out.answer_framework == "concise-summary"
    assert out.role_id is None
    assert out.follow_up_question.startswith("Which part")

    missing = FakeSession([FakeResult(row=None)])
    assert await repo.get_question_coaching(missing, "NOPE") is None
