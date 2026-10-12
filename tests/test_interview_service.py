"""E7 interview orchestration: setup/role allow-list, question selection, session
lifecycle, recording/retry, areas aggregation, and retention. Parakeet, FFmpeg, Gemini,
clock and random selection are all fakes; question selection is also tested directly
as a pure function.
"""
import random
from datetime import datetime, timedelta, timezone

import pytest

from app.repositories.interview import (
    AttemptRow,
    CriterionRow,
    QuestionRow,
    QuestionSlotDetail,
    RoleRow,
    SessionDetail,
    SessionRow,
    SessionSummaryRow,
)
from app.services.interview import (
    InterviewError,
    InterviewService,
    _question_slot_to_schema,
    _select_questions,
)
from app.services.interview_feedback import FeedbackError, FeedbackItem, FeedbackResult
from app.services.transcription import AudioInput, TranscriptionError, TranscriptionResult

# -- fixtures -----------------------------------------------------------------------

def _q(question_id, role_id, difficulty, category="general_cat"):
    return QuestionRow(
        question_id, role_id, category, difficulty, f"Question {question_id}?",
        interview_method_sources="OPM-STRUCTURED-INTERVIEWS; VA-PBI",
        authoring_method="authored_from_structured_behavioural_interview_patterns",
        answer_framework="concise-summary",
        answer_guidance="State your current professional focus and one evidence point.",
        strong_evidence_signals="Specific context; clear personal contribution.",
        watch_out_for="Generic answer; unclear ownership.",
        follow_up_question=f"Follow-up for {question_id}?",
    )


def _full_pool(role_id="R1"):
    """4 general + 4 role-specific questions per difficulty, matching the real data
    shape (every role has exactly 4/4/4; general has enough too)."""
    pool = []
    for d in ("foundation", "intermediate", "advanced"):
        for i in range(4):
            pool.append(_q(f"GEN-{d[:3]}-{i}", None, d, category=f"gen_{d}_{i % 2}"))
        for i in range(4):
            pool.append(_q(f"{role_id}-{d[:3]}-{i}", role_id, d, category=f"role_{d}_{i % 2}"))
    return pool


def _tech_pool(role_id="R1"):
    """Role pool matching the real technical spread: 0 technical at foundation,
    1 at intermediate, 2 at advanced (3 technical total), the rest non-technical."""
    pool = []
    for c in ("introduction", "motivation", "career_growth", "questions_for_interviewer"):
        pool.append(_q(f"{role_id}-fnd-{c}", role_id, "foundation", category=c))
    pool.append(_q(f"{role_id}-int-tech", role_id, "intermediate", category="technical"))
    for c in ("behavioural_a", "behavioural_b", "situational"):
        pool.append(_q(f"{role_id}-int-{c}", role_id, "intermediate", category=c))
    for n in (1, 2):
        pool.append(_q(f"{role_id}-adv-tech{n}", role_id, "advanced", category="technical"))
    for c in ("behavioural", "situational"):
        pool.append(_q(f"{role_id}-adv-{c}", role_id, "advanced", category=c))
    return pool


def _mixed_tech_pool(role_id="R1"):
    """Role tech pool plus general (role_id None) questions, for exercising mixed focus."""
    pool = list(_tech_pool(role_id))
    for d in ("foundation", "intermediate", "advanced"):
        for i in range(4):
            pool.append(_q(f"GEN-{d[:3]}-{i}", None, d, category=f"gen_{d}_{i % 2}"))
    return pool


def _slot_detail(role_id):
    return QuestionSlotDetail(
        1, "GEN-001" if role_id is None else "R1-int-0", role_id, "Question?", "cat", "foundation",
        "OPM-STRUCTURED-INTERVIEWS", "authored", "concise-summary", "guidance",
        "signals", "watch out", "follow-up?",
    )


def test_question_slot_to_schema_general_kind_when_role_id_null():
    out = _question_slot_to_schema(_slot_detail(None))
    assert out.role_id is None
    assert out.kind == "general"


def test_question_slot_to_schema_role_specific_kind_when_role_id_present():
    out = _question_slot_to_schema(_slot_detail("R1"))
    assert out.role_id == "R1"
    assert out.kind == "role_specific"


class FakeAccounts:
    def __init__(self, plan=None):
        self.plan = plan

    async def get_plan(self, session, username):
        return self.plan


class FakeInterviewRepo:
    def __init__(self, questions=None, roles=None, criteria=None):
        self.questions = questions or []
        self.roles = roles or {}
        self.criteria = criteria or {}
        self.sessions: dict[str, dict] = {}
        self.responses: dict[int, dict] = {}
        self._next_session = 1
        self._next_response = 1
        self.areas = []
        self.purge_calls = []

    async def get_roles_by_ids(self, session, role_ids):
        return [self.roles[r] for r in role_ids if r in self.roles]

    async def get_question_pool(self, session, role_id):
        return [q for q in self.questions if q.role_id is None or q.role_id == role_id]

    async def get_criteria_for_questions(self, session, question_ids):
        return {qid: self.criteria.get(qid, []) for qid in question_ids if qid in self.criteria}

    async def find_session(self, session, username, role_id, practice_focus):
        for sid, s in self.sessions.items():
            if s["username"] == username and s["role_id"] == role_id and s["practice_focus"] == practice_focus:
                return SessionRow(sid, s["status"], s["created_at"], s["updated_at"])
        return None

    async def insert_session(self, session, username, role_id, practice_focus):
        sid = f"s{self._next_session}"
        self._next_session += 1
        now = datetime(2026, 10, 3, tzinfo=timezone.utc)
        self.sessions[sid] = dict(
            username=username, role_id=role_id, practice_focus=practice_focus,
            status="active", created_at=now, updated_at=now,
        )
        return SessionRow(sid, "active", now, now)

    async def insert_placeholder_responses(self, session, session_id, question_ids):
        for i, qid in enumerate(question_ids, start=1):
            self._insert_response(session_id, qid, i, 1)

    def _insert_response(self, session_id, question_id, sequence_no, attempt_no):
        rid = self._next_response
        self._next_response += 1
        now = datetime(2026, 10, 3, tzinfo=timezone.utc)
        self.responses[rid] = dict(
            response_id=rid, session_id=session_id, question_id=question_id,
            sequence_no=sequence_no, attempt_no=attempt_no,
            transcript=None, detected_language=None, audio_duration_s=None,
            feedback_status="pending", feedback_summary=None,
            worked_well=[], what_to_improve=[], strength_tags=[], improvement_tags=[],
            error_code=None, created_at=now, updated_at=now, content_purged_at=None,
        )
        return rid

    async def list_sessions(self, session, username):
        out = []
        for sid, s in self.sessions.items():
            if s["username"] != username:
                continue
            ready = {
                r["sequence_no"] for r in self.responses.values()
                if r["session_id"] == sid and r["feedback_status"] == "ready"
            }
            role = self.roles[s["role_id"]]
            out.append(
                SessionSummaryRow(
                    sid, role.role_id, role.role_title, s["practice_focus"], s["status"],
                    len(ready), s["created_at"], s["updated_at"],
                )
            )
        return sorted(out, key=lambda r: r.created_at, reverse=True)

    def _attempt_row(self, r):
        return AttemptRow(
            r["response_id"], r["session_id"], r["question_id"], r["sequence_no"], r["attempt_no"],
            r["transcript"], r["detected_language"], r["audio_duration_s"], r["feedback_status"],
            r["feedback_summary"], r["worked_well"], r["what_to_improve"],
            r["strength_tags"], r["improvement_tags"], r["error_code"],
            r["created_at"], r["updated_at"], r["content_purged_at"],
        )

    async def get_session_detail(self, session, username, session_id):
        s = self.sessions.get(session_id)
        if s is None or s["username"] != username:
            return None
        role = self.roles[s["role_id"]]
        rows = sorted(
            (r for r in self.responses.values() if r["session_id"] == session_id),
            key=lambda r: (r["sequence_no"], r["attempt_no"]),
        )
        slots: dict[int, QuestionSlotDetail] = {}
        for r in rows:
            slot = slots.get(r["sequence_no"])
            if slot is None:
                q = next(q for q in self.questions if q.question_id == r["question_id"])
                slot = QuestionSlotDetail(
                    r["sequence_no"], q.question_id, q.role_id, q.question_text, q.category,
                    q.difficulty,
                    q.interview_method_sources, q.authoring_method, q.answer_framework,
                    q.answer_guidance, q.strong_evidence_signals, q.watch_out_for, q.follow_up_question,
                )
                slots[r["sequence_no"]] = slot
            slot.attempts.append(self._attempt_row(r))
        return SessionDetail(
            session_id, role.role_id, role.role_title, s["practice_focus"], s["status"],
            s["created_at"], s["updated_at"], [slots[k] for k in sorted(slots)],
        )

    async def lock_latest_attempt(self, session, username, session_id, sequence_no):
        s = self.sessions.get(session_id)
        if s is None or s["username"] != username:
            return None
        candidates = [
            r for r in self.responses.values()
            if r["session_id"] == session_id and r["sequence_no"] == sequence_no
        ]
        if not candidates:
            return None
        return self._attempt_row(max(candidates, key=lambda r: r["attempt_no"]))

    async def insert_attempt(self, session, session_id, question_id, sequence_no, attempt_no):
        return self._insert_response(session_id, question_id, sequence_no, attempt_no)

    async def save_transcript_processing(self, session, response_id, transcript, detected_language, audio_duration_s):
        r = self.responses[response_id]
        r.update(
            transcript=transcript, detected_language=detected_language,
            audio_duration_s=audio_duration_s, feedback_status="processing",
        )

    async def save_feedback_ready(
        self, session, response_id, feedback_summary, worked_well, what_to_improve,
        strength_tags, improvement_tags, model_metadata,
    ):
        r = self.responses[response_id]
        r.update(
            feedback_status="ready", feedback_summary=feedback_summary,
            worked_well=worked_well, what_to_improve=what_to_improve,
            strength_tags=strength_tags, improvement_tags=improvement_tags,
        )

    async def save_feedback_error(self, session, response_id, error_code):
        self.responses[response_id].update(feedback_status="error", error_code=error_code)

    async def mark_completed_if_all_ready(self, session, session_id):
        ready = {
            r["sequence_no"] for r in self.responses.values()
            if r["session_id"] == session_id and r["feedback_status"] == "ready"
        }
        if len(ready) == 5:
            self.sessions[session_id]["status"] = "completed"

    async def delete_session(self, session, username, session_id):
        s = self.sessions.get(session_id)
        if s is None or s["username"] != username:
            return False
        del self.sessions[session_id]
        for rid in [rid for rid, r in self.responses.items() if r["session_id"] == session_id]:
            del self.responses[rid]
        return True

    async def get_attempt_by_id(self, session, username, response_id):
        r = self.responses.get(response_id)
        if r is None:
            return None
        s = self.sessions.get(r["session_id"])
        if s is None or s["username"] != username:
            return None
        return self._attempt_row(r)

    async def get_recurring_areas(self, session, username):
        return self.areas

    async def purge_expired_content(self, session, cutoff):
        self.purge_calls.append(cutoff)
        return 0


class FakeTranscriber:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    async def transcribe(self, audio):
        self.calls.append(audio)
        if self.error:
            raise self.error
        return self.result


class FakeRedactor:
    def __init__(self):
        self.calls = []

    def redact(self, text):
        self.calls.append(text)
        return text.replace("secret-detail", "[redacted]")


class FakeFeedback:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    async def evaluate(self, data):
        self.calls.append(data)
        if self.error:
            raise self.error
        return self.result


_CRITERIA = {
    "GEN-fou-0": [
        CriterionRow("EVAL-01", "relevance_to_question", "answers_the_question", "focus_on_the_question", "no accent bias"),
    ],
}


def _plan(role_id="R1", recommended=()):
    return {
        "selectedRole": {"role": "Data Analyst", "role_id": role_id},
        "snapshot": {
            "recommended_roles": [{"role_id": r} for r in recommended],
            "professional_skills": [{"skill": "Excel"}],
            "reframed_skills": [{"skill": "Budgeting"}],
        },
    }


def _service(repo, accounts, transcriber=None, redactor=None, feedback=None, rng=None):
    return InterviewService(
        transcriber=transcriber or FakeTranscriber(),
        redactor=redactor or FakeRedactor(),
        feedback_service=feedback or FakeFeedback(),
        rng=rng or random.Random(1),
        repo=repo,
        accounts=accounts,
    )


def _feedback_result(summary="Clear answer.", strengths=None, improvements=None):
    return FeedbackResult(
        summary=summary,
        strengths=strengths or [FeedbackItem("EVAL-01", "Relevance To Question", "answers_the_question", "On topic.")],
        improvements=improvements or [],
        model="gemini-3.1-flash-lite", tokens_in=10, tokens_out=20,
    )


# -- pure question selection ---------------------------------------------------------

def test_general_focus_selects_two_foundation_two_intermediate_one_advanced():
    pool = _full_pool()
    picked = _select_questions(random.Random(1), pool, "R1", "general", set())
    assert len(picked) == 5
    assert all(q.role_id is None for q in picked)
    diffs = [q.difficulty for q in picked]
    assert diffs == ["foundation", "foundation", "intermediate", "intermediate", "advanced"]


def test_role_specific_focus_selects_from_role_pool_only():
    pool = _full_pool()
    picked = _select_questions(random.Random(2), pool, "R1", "role_specific", set())
    assert len(picked) == 5
    assert all(q.role_id == "R1" for q in picked)


def test_role_specific_focus_is_technical_weighted():
    pool = _tech_pool()
    for seed in range(20):
        picked = _select_questions(random.Random(seed), pool, "R1", "role_specific", set())
        assert len(picked) == 5
        # all available technical questions are included
        assert len([q for q in picked if q.category == "technical"]) == 3
        # a gentle foundation opener is kept
        assert any(q.difficulty == "foundation" for q in picked)
        # presented foundation -> intermediate -> advanced
        order = ("foundation", "intermediate", "advanced")
        idx = [order.index(q.difficulty) for q in picked]
        assert idx == sorted(idx)


def test_mixed_focus_weights_technical_in_role_specific_slots():
    pool = _mixed_tech_pool()
    for seed in range(20):
        picked = _select_questions(random.Random(seed), pool, "R1", "mixed", set())
        assert len(picked) == 5
        # the role-specific intermediate and advanced slots are the technical ones
        # (foundation has no technical question); the general slots stay non-technical.
        role_tech = [q for q in picked if q.role_id == "R1" and q.category == "technical"]
        assert len(role_tech) == 2


def test_mixed_focus_splits_general_and_role_specific_per_difficulty():
    pool = _full_pool()
    picked = _select_questions(random.Random(3), pool, "R1", "mixed", set())
    assert len(picked) == 5
    foundation = [q for q in picked if q.difficulty == "foundation"]
    intermediate = [q for q in picked if q.difficulty == "intermediate"]
    advanced = [q for q in picked if q.difficulty == "advanced"]
    assert len(foundation) == 2 and {q.role_id for q in foundation} == {None, "R1"}
    assert len(intermediate) == 2 and {q.role_id for q in intermediate} == {None, "R1"}
    assert len(advanced) == 1 and advanced[0].role_id == "R1"


def test_presentation_order_is_foundation_then_intermediate_then_advanced():
    pool = _full_pool()
    picked = _select_questions(random.Random(4), pool, "R1", "general", set())
    order = [q.difficulty for q in picked]
    assert order.index("foundation") < order.index("intermediate") < order.index("advanced")


def test_no_question_id_repeats_within_a_session():
    pool = _full_pool()
    for seed in range(20):
        picked = _select_questions(random.Random(seed), pool, "R1", "mixed", set())
        ids = [q.question_id for q in picked]
        assert len(ids) == len(set(ids))


def test_category_diversity_is_preferred_when_possible():
    # two foundation categories available, four questions -> picking 2 should favour distinct categories
    pool = [
        _q("g1", None, "foundation", "cat_a"), _q("g2", None, "foundation", "cat_a"),
        _q("g3", None, "foundation", "cat_b"), _q("g4", None, "foundation", "cat_b"),
    ] + _full_pool()[4:]  # fill in the rest of the buckets from the standard pool
    picked = _select_questions(random.Random(5), pool, "R1", "general", set())
    foundation = [q for q in picked if q.difficulty == "foundation"]
    assert {q.category for q in foundation} == {"cat_a", "cat_b"}


def test_category_diversity_never_overrides_difficulty_requirement():
    # only one category available per difficulty bucket -- diversity is impossible,
    # but the difficulty counts must still be met.
    pool = []
    for d, count in (("foundation", 2), ("intermediate", 2), ("advanced", 1)):
        for i in range(4):
            pool.append(_q(f"gen-{d}-{i}", None, d, category="only_category"))
        for i in range(4):
            pool.append(_q(f"role-{d}-{i}", "R1", d, category="only_category"))
    picked = _select_questions(random.Random(6), pool, "R1", "general", set())
    assert len(picked) == 5


def test_refresh_excludes_all_five_old_ids_when_alternatives_exist():
    pool = _full_pool()
    first = _select_questions(random.Random(7), pool, "R1", "general", set())
    old_ids = {q.question_id for q in first}
    second = _select_questions(random.Random(8), pool, "R1", "general", old_ids)
    assert old_ids.isdisjoint({q.question_id for q in second})


def test_pool_exhaustion_allows_reuse_only_for_the_exhausted_pool():
    # general-foundation pool has only 2 questions total -- excluding both exhausts it,
    # but the (unrelated) general-intermediate pool must still honour the exclusion.
    pool = [
        _q("gf1", None, "foundation", "a"), _q("gf2", None, "foundation", "b"),
    ]
    pool += [_q(f"gi{i}", None, "intermediate", f"c{i}") for i in range(4)]
    pool += [_q(f"ga{i}", None, "advanced", f"d{i}") for i in range(4)]
    pool += [_q(f"rf{i}", "R1", "foundation", f"e{i}") for i in range(4)]
    pool += [_q(f"ri{i}", "R1", "intermediate", f"f{i}") for i in range(4)]
    pool += [_q(f"ra{i}", "R1", "advanced", f"g{i}") for i in range(4)]

    exclude = {"gf1", "gf2", "gi0"}  # exhausts the 2-item foundation pool; intermediate has plenty
    picked = _select_questions(random.Random(9), pool, "R1", "general", exclude)
    foundation_ids = {q.question_id for q in picked if q.difficulty == "foundation"}
    intermediate_ids = {q.question_id for q in picked if q.difficulty == "intermediate"}
    assert foundation_ids == {"gf1", "gf2"}  # reused: this pool was exhausted
    assert "gi0" not in intermediate_ids  # not exhausted: exclusion still honoured


# -- setup / role allow-list ----------------------------------------------------------

_ROLES = {
    "R1": RoleRow("R1", "Data Analyst"),
    "R2": RoleRow("R2", "Operations Manager"),
}


async def test_get_setup_returns_selected_and_available_roles_in_journey_order():
    repo = FakeInterviewRepo(roles=_ROLES)
    accounts = FakeAccounts(_plan(role_id="R1", recommended=["R2", "R1"]))
    svc = _service(repo, accounts)
    setup = await svc.get_setup(object(), "aisha")
    assert setup.selected_role.role_id == "R1"
    assert [r.role_id for r in setup.available_roles] == ["R1", "R2"]  # deduped, order preserved


async def test_get_setup_raises_when_selected_role_missing():
    repo = FakeInterviewRepo(roles=_ROLES)
    accounts = FakeAccounts({"selectedRole": {}, "snapshot": {}})
    svc = _service(repo, accounts)
    with pytest.raises(InterviewError) as exc:
        await svc.get_setup(object(), "aisha")
    assert exc.value.code == "journey_prerequisite_incomplete"


async def test_get_setup_raises_when_no_plan_at_all():
    repo = FakeInterviewRepo(roles=_ROLES)
    accounts = FakeAccounts(None)
    svc = _service(repo, accounts)
    with pytest.raises(InterviewError) as exc:
        await svc.get_setup(object(), "aisha")
    assert exc.value.code == "journey_prerequisite_incomplete"


async def test_create_session_rejects_role_outside_matched_set():
    repo = FakeInterviewRepo(questions=_full_pool("R2"), roles=_ROLES)
    accounts = FakeAccounts(_plan(role_id="R1", recommended=[]))
    svc = _service(repo, accounts)
    with pytest.raises(InterviewError) as exc:
        await svc.create_session(object(), "aisha", "R2", "general")
    assert exc.value.code == "invalid_setup"


# -- session lifecycle ----------------------------------------------------------------

async def test_create_session_is_idempotent_for_same_role_and_focus():
    repo = FakeInterviewRepo(questions=_full_pool("R1"), roles=_ROLES)
    accounts = FakeAccounts(_plan(role_id="R1"))
    svc = _service(repo, accounts)

    detail1, created1 = await svc.create_session(object(), "aisha", "R1", "general")
    detail2, created2 = await svc.create_session(object(), "aisha", "R1", "general")

    assert created1 is True and created2 is False
    assert detail1.session_id == detail2.session_id
    assert len(repo.sessions) == 1


async def test_create_session_different_focus_creates_a_separate_session():
    repo = FakeInterviewRepo(questions=_full_pool("R1"), roles=_ROLES)
    accounts = FakeAccounts(_plan(role_id="R1"))
    svc = _service(repo, accounts)

    detail1, _ = await svc.create_session(object(), "aisha", "R1", "general")
    detail2, created2 = await svc.create_session(object(), "aisha", "R1", "mixed")

    assert created2 is True
    assert detail1.session_id != detail2.session_id


async def test_new_session_has_five_question_slots_in_order():
    repo = FakeInterviewRepo(questions=_full_pool("R1"), roles=_ROLES)
    accounts = FakeAccounts(_plan(role_id="R1"))
    svc = _service(repo, accounts)
    detail, _ = await svc.create_session(object(), "aisha", "R1", "general")
    assert [q.sequence_no for q in detail.questions] == [1, 2, 3, 4, 5]


async def test_list_sessions_scoped_to_username():
    repo = FakeInterviewRepo(questions=_full_pool("R1"), roles=_ROLES)
    accounts = FakeAccounts(_plan(role_id="R1"))
    svc = _service(repo, accounts)
    await svc.create_session(object(), "aisha", "R1", "general")
    out = await svc.list_sessions(object(), "aisha")
    assert len(out) == 1
    assert (await svc.list_sessions(object(), "siti")) == []


async def test_get_session_raises_not_found_when_missing_or_not_owned():
    repo = FakeInterviewRepo(questions=_full_pool("R1"), roles=_ROLES)
    accounts = FakeAccounts(_plan(role_id="R1"))
    svc = _service(repo, accounts)
    detail, _ = await svc.create_session(object(), "aisha", "R1", "general")

    with pytest.raises(InterviewError) as exc:
        await svc.get_session(object(), "siti", detail.session_id)
    assert exc.value.code == "interview_resource_not_found"

    with pytest.raises(InterviewError):
        await svc.get_session(object(), "aisha", "does-not-exist")


async def test_refresh_replaces_session_and_excludes_old_questions():
    repo = FakeInterviewRepo(questions=_full_pool("R1"), roles=_ROLES)
    accounts = FakeAccounts(_plan(role_id="R1"))
    svc = _service(repo, accounts)
    old, _ = await svc.create_session(object(), "aisha", "R1", "general")
    old_ids = {q.question_id for q in old.questions}

    new = await svc.refresh_session(object(), "aisha", old.session_id)

    assert new.session_id != old.session_id
    assert old.session_id not in repo.sessions  # old session gone
    new_ids = {q.question_id for q in new.questions}
    assert old_ids.isdisjoint(new_ids)


async def test_refresh_raises_not_found_for_unowned_session():
    repo = FakeInterviewRepo(questions=_full_pool("R1"), roles=_ROLES)
    accounts = FakeAccounts(_plan(role_id="R1"))
    svc = _service(repo, accounts)
    detail, _ = await svc.create_session(object(), "aisha", "R1", "general")
    with pytest.raises(InterviewError):
        await svc.refresh_session(object(), "siti", detail.session_id)


async def test_delete_session_removes_it_and_raises_if_repeated():
    repo = FakeInterviewRepo(questions=_full_pool("R1"), roles=_ROLES)
    accounts = FakeAccounts(_plan(role_id="R1"))
    svc = _service(repo, accounts)
    detail, _ = await svc.create_session(object(), "aisha", "R1", "general")
    await svc.delete_session(object(), "aisha", detail.session_id)
    assert detail.session_id not in repo.sessions
    with pytest.raises(InterviewError):
        await svc.delete_session(object(), "aisha", detail.session_id)


# -- recording / feedback --------------------------------------------------------------

async def _create(svc):
    return await svc.create_session(object(), "aisha", "R1", "general")


def _criteria_for_pool():
    return {
        q.question_id: [
            CriterionRow(
                "EVAL-01", "relevance_to_question", "answers_the_question",
                "focus_on_the_question", "no accent bias",
            )
        ]
        for q in _full_pool("R1")
    }


async def test_first_recording_saves_transcript_and_ready_feedback():
    repo = FakeInterviewRepo(questions=_full_pool("R1"), roles=_ROLES, criteria=_criteria_for_pool())
    accounts = FakeAccounts(_plan(role_id="R1"))
    transcriber = FakeTranscriber(result=TranscriptionResult("I led a team.", "en", 5.0))
    redactor = FakeRedactor()
    feedback = FakeFeedback(result=_feedback_result())
    svc = _service(repo, accounts, transcriber, redactor, feedback)

    detail, _ = await _create(svc)
    outcome = await svc.process_recording(
        object(), "aisha", detail.session_id, 1, AudioInput(b"x", "a.wav", "audio/wav")
    )

    assert outcome.status == "ready"
    assert outcome.attempt.attempt_no == 1
    assert outcome.attempt.transcript == "I led a team."
    sent = feedback.calls[0]
    assert sent.answer_framework == "concise-summary"
    assert sent.strong_evidence_signals == "Specific context; clear personal contribution."
    assert sent.watch_out_for == "Generic answer; unclear ownership."
    assert outcome.attempt.feedback_status == "ready"
    assert len(transcriber.calls) == 1


async def test_retry_inserts_a_new_attempt_with_unlimited_retries():
    repo = FakeInterviewRepo(questions=_full_pool("R1"), roles=_ROLES, criteria=_criteria_for_pool())
    accounts = FakeAccounts(_plan(role_id="R1"))
    transcriber = FakeTranscriber(result=TranscriptionResult("First try.", "en", 4.0))
    svc = _service(repo, accounts, transcriber, FakeRedactor(), FakeFeedback(result=_feedback_result()))
    detail, _ = await _create(svc)

    first = await svc.process_recording(object(), "aisha", detail.session_id, 1, AudioInput(b"x", "a.wav", "audio/wav"))
    transcriber.result = TranscriptionResult("Second try.", "en", 4.0)
    second = await svc.process_recording(object(), "aisha", detail.session_id, 1, AudioInput(b"y", "a.wav", "audio/wav"))
    transcriber.result = TranscriptionResult("Third try.", "en", 4.0)
    third = await svc.process_recording(object(), "aisha", detail.session_id, 1, AudioInput(b"z", "a.wav", "audio/wav"))

    assert [first.attempt.attempt_no, second.attempt.attempt_no, third.attempt.attempt_no] == [1, 2, 3]
    full = await svc.get_session(object(), "aisha", detail.session_id)
    slot = next(q for q in full.questions if q.sequence_no == 1)
    assert len(slot.attempts) == 3  # all attempts remain available


async def test_no_speech_detected_does_not_consume_an_attempt():
    repo = FakeInterviewRepo(questions=_full_pool("R1"), roles=_ROLES, criteria=_criteria_for_pool())
    accounts = FakeAccounts(_plan(role_id="R1"))
    transcriber = FakeTranscriber(error=TranscriptionError("no_speech_detected"))
    svc = _service(repo, accounts, transcriber, FakeRedactor(), FakeFeedback())
    detail, _ = await _create(svc)

    with pytest.raises(InterviewError) as exc:
        await svc.process_recording(object(), "aisha", detail.session_id, 1, AudioInput(b"x", "a.wav", "audio/wav"))
    assert exc.value.code == "no_speech_detected"

    full = await svc.get_session(object(), "aisha", detail.session_id)
    slot = next(q for q in full.questions if q.sequence_no == 1)
    assert len(slot.attempts) == 1  # still just the untouched placeholder
    assert slot.attempts[0].transcript is None


async def test_redaction_runs_before_repository_persistence_or_gemini_access():
    repo = FakeInterviewRepo(questions=_full_pool("R1"), roles=_ROLES, criteria=_criteria_for_pool())
    accounts = FakeAccounts(_plan(role_id="R1"))
    transcriber = FakeTranscriber(result=TranscriptionResult("my secret-detail here", "en", 3.0))
    redactor = FakeRedactor()
    feedback = FakeFeedback(result=_feedback_result())
    svc = _service(repo, accounts, transcriber, redactor, feedback)
    detail, _ = await _create(svc)

    outcome = await svc.process_recording(
        object(), "aisha", detail.session_id, 1, AudioInput(b"x", "a.wav", "audio/wav")
    )

    assert redactor.calls == ["my secret-detail here"]
    assert outcome.attempt.transcript == "my [redacted] here"
    sent_transcript = feedback.calls[0].transcript
    assert sent_transcript == "my [redacted] here"  # Gemini never sees the raw transcript
    assert repo.responses[1]["transcript"] == "my [redacted] here"  # persisted form is redacted too


async def test_transcript_is_preserved_when_feedback_fails():
    repo = FakeInterviewRepo(questions=_full_pool("R1"), roles=_ROLES, criteria=_criteria_for_pool())
    accounts = FakeAccounts(_plan(role_id="R1"))
    transcriber = FakeTranscriber(result=TranscriptionResult("I led a team.", "en", 5.0))
    feedback = FakeFeedback(error=FeedbackError("llm_error"))
    svc = _service(repo, accounts, transcriber, FakeRedactor(), feedback)
    detail, _ = await _create(svc)

    outcome = await svc.process_recording(
        object(), "aisha", detail.session_id, 1, AudioInput(b"x", "a.wav", "audio/wav")
    )

    assert outcome.status == "feedback_error"
    assert outcome.attempt.feedback_status == "error"
    assert outcome.attempt.error_code == "feedback_service_unavailable"
    assert outcome.attempt.transcript == "I led a team."  # transcript kept, not discarded


async def test_retry_feedback_does_not_call_the_transcriber():
    repo = FakeInterviewRepo(questions=_full_pool("R1"), roles=_ROLES, criteria=_criteria_for_pool())
    accounts = FakeAccounts(_plan(role_id="R1"))
    transcriber = FakeTranscriber(result=TranscriptionResult("I led a team.", "en", 5.0))
    feedback = FakeFeedback(error=FeedbackError("llm_error"))
    svc = _service(repo, accounts, transcriber, FakeRedactor(), feedback)
    detail, _ = await _create(svc)
    first = await svc.process_recording(object(), "aisha", detail.session_id, 1, AudioInput(b"x", "a.wav", "audio/wav"))
    assert first.status == "feedback_error"

    feedback.error = None
    feedback.result = _feedback_result()
    response_id = list(repo.responses.keys())[0]
    retried = await svc.retry_feedback(object(), "aisha", response_id)

    assert retried.status == "ready"
    assert len(transcriber.calls) == 1  # unchanged -- retry never re-transcribes


async def test_retry_feedback_rejects_after_content_expiry():
    repo = FakeInterviewRepo(questions=_full_pool("R1"), roles=_ROLES, criteria=_criteria_for_pool())
    accounts = FakeAccounts(_plan(role_id="R1"))
    transcriber = FakeTranscriber(result=TranscriptionResult("I led a team.", "en", 5.0))
    svc = _service(repo, accounts, transcriber, FakeRedactor(), FakeFeedback(error=FeedbackError("x")))
    detail, _ = await _create(svc)
    await svc.process_recording(object(), "aisha", detail.session_id, 1, AudioInput(b"x", "a.wav", "audio/wav"))
    response_id = list(repo.responses.keys())[0]
    repo.responses[response_id]["content_purged_at"] = datetime(2026, 11, 1, tzinfo=timezone.utc)

    with pytest.raises(InterviewError) as exc:
        await svc.retry_feedback(object(), "aisha", response_id)
    assert exc.value.code == "interview_resource_not_found"


async def test_completion_flips_after_all_five_positions_ready():
    repo = FakeInterviewRepo(questions=_full_pool("R1"), roles=_ROLES, criteria=_criteria_for_pool())
    accounts = FakeAccounts(_plan(role_id="R1"))
    transcriber = FakeTranscriber(result=TranscriptionResult("Answer.", "en", 3.0))
    svc = _service(repo, accounts, transcriber, FakeRedactor(), FakeFeedback(result=_feedback_result()))
    detail, _ = await _create(svc)

    for seq in range(1, 6):
        await svc.process_recording(object(), "aisha", detail.session_id, seq, AudioInput(b"x", "a.wav", "audio/wav"))

    final = await svc.get_session(object(), "aisha", detail.session_id)
    assert final.status == "completed"


# -- areas / retention -----------------------------------------------------------------

async def test_areas_separates_strengths_and_improvements():
    from app.repositories.interview import AreaRow

    repo = FakeInterviewRepo(roles=_ROLES)
    repo.areas = [
        AreaRow("strength", "EVAL-01", "Relevance To Question", 5),
        AreaRow("improvement", "EVAL-07", "Problem Solving", 2),
    ]
    svc = _service(repo, FakeAccounts())
    areas = await svc.get_areas(object(), "aisha")
    assert areas.strengths[0].criterion_id == "EVAL-01"
    assert areas.improvements[0].criterion_id == "EVAL-07"


async def test_purge_uses_injected_clock_and_configured_retention_days():
    repo = FakeInterviewRepo(roles=_ROLES)
    fixed_now = datetime(2026, 11, 1, tzinfo=timezone.utc)
    svc = InterviewService(
        transcriber=FakeTranscriber(), redactor=FakeRedactor(), feedback_service=FakeFeedback(),
        clock=lambda: fixed_now, retention_days=30, repo=repo, accounts=FakeAccounts(),
    )
    await svc.purge_expired_content(object())
    assert repo.purge_calls == [fixed_now - timedelta(days=30)]
