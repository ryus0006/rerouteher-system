"""E7 AI Interview Coach orchestration: setup, session lifecycle, recording,
feedback (retry), recurring areas, and retention. Pure: no FastAPI import. The router
maps InterviewError to HTTP and owns the request transaction (see app/api/interview.py).

Question selection, transcription and feedback are kept as separate injected
collaborators (transcriber, redactor, feedback_service) so this module stays testable
with fakes and so each concern (audio, privacy, grounding) keeps its own tests.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.repositories import accounts as accounts_repo
from app.repositories import interview as interview_repo
from app.schemas.interview import (
    AreaOut,
    AreasOut,
    AttemptOut,
    FeedbackItemOut,
    InterviewSetupOut,
    QuestionSlotOut,
    RoleOut,
    SessionDetailOut,
    SessionSummaryOut,
)
from app.services.interview_feedback import FeedbackError, FeedbackInput
from app.services.transcription import TranscriptionError

_DIFFICULTIES = ("foundation", "intermediate", "advanced")
_COUNTS_BY_DIFFICULTY = {"foundation": 2, "intermediate": 2, "advanced": 1}


class InterviewError(Exception):
    """code is one of the approved E7 error codes; the API never exposes internals."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass
class RecordingOutcome:
    status: str  # "ready" or "feedback_error"
    attempt: AttemptOut


# -- question selection (pure, independently testable) ----------------------------

def _bucket(pool, role_id, difficulty):
    return [q for q in pool if q.role_id == role_id and q.difficulty == difficulty]


def _pick(rng: random.Random, candidates: list, count: int, exclude_ids: set[str]) -> list:
    """Picks `count` distinct questions, preferring category diversity. Excludes
    exclude_ids unless doing so would leave too few, in which case this call's pool is
    treated as exhausted and reuse is allowed for it alone."""
    pool = [q for q in candidates if q.question_id not in exclude_ids]
    if len(pool) < count:
        pool = list(candidates)
    pool = list(pool)
    rng.shuffle(pool)

    picked, leftovers, seen_categories = [], [], set()
    for q in pool:
        if len(picked) >= count:
            break
        if q.category not in seen_categories:
            picked.append(q)
            seen_categories.add(q.category)
        else:
            leftovers.append(q)
    for q in leftovers:
        if len(picked) >= count:
            break
        picked.append(q)
    return picked[:count]


def _select_questions(
    rng: random.Random, pool: list, role_id: str, practice_focus: str, exclude_ids: set[str]
) -> list:
    general = {d: _bucket(pool, None, d) for d in _DIFFICULTIES}
    role_specific = {d: _bucket(pool, role_id, d) for d in _DIFFICULTIES}

    if practice_focus == "general":
        plan = [(general, d, _COUNTS_BY_DIFFICULTY[d]) for d in _DIFFICULTIES]
    elif practice_focus == "role_specific":
        plan = [(role_specific, d, _COUNTS_BY_DIFFICULTY[d]) for d in _DIFFICULTIES]
    else:  # mixed: one general + one role-specific per foundation/intermediate, role-specific advanced
        plan = [
            (general, "foundation", 1), (role_specific, "foundation", 1),
            (general, "intermediate", 1), (role_specific, "intermediate", 1),
            (role_specific, "advanced", 1),
        ]

    by_difficulty: dict[str, list] = {d: [] for d in _DIFFICULTIES}
    for bucket_map, difficulty, count in plan:
        by_difficulty[difficulty].extend(_pick(rng, bucket_map[difficulty], count, exclude_ids))

    ordered = []
    for difficulty in _DIFFICULTIES:
        group = by_difficulty[difficulty]
        rng.shuffle(group)  # same-difficulty order may be shuffled
        ordered.extend(group)
    return ordered


# -- repo-row to public-schema mapping --------------------------------------------

def _feedback_items_to_schema(items: list[dict]) -> list[FeedbackItemOut]:
    return [
        FeedbackItemOut(criterion_id=i["criterion_id"], title=i["title"], detail=i["detail"])
        for i in items
    ]


def _attempt_to_schema(row) -> AttemptOut:
    return AttemptOut(
        response_id=row.response_id,
        attempt_no=row.attempt_no,
        transcript=row.transcript,
        feedback_status=row.feedback_status,
        feedback_summary=row.feedback_summary,
        strengths=_feedback_items_to_schema(row.worked_well),
        improvements=_feedback_items_to_schema(row.what_to_improve),
        detected_language=row.detected_language,
        duration_s=float(row.audio_duration_s) if row.audio_duration_s is not None else None,
        error_code=row.error_code,
        created_at=row.created_at,
        updated_at=row.updated_at,
        content_expired=row.content_purged_at is not None,
    )


def _question_slot_to_schema(slot) -> QuestionSlotOut:
    return QuestionSlotOut(
        sequence_no=slot.sequence_no, question_id=slot.question_id,
        question_text=slot.question_text, category=slot.category, difficulty=slot.difficulty,
        role_id=slot.role_id,
        kind="general" if slot.role_id is None else "role_specific",
        attempts=[_attempt_to_schema(a) for a in slot.attempts],
    )


def _session_detail_to_schema(detail) -> SessionDetailOut:
    return SessionDetailOut(
        session_id=detail.session_id,
        role=RoleOut(role_id=detail.role_id, role_title=detail.role_title),
        practice_focus=detail.practice_focus,
        status=detail.status,
        created_at=detail.created_at,
        updated_at=detail.updated_at,
        questions=[_question_slot_to_schema(q) for q in detail.questions],
    )


def _summary_to_schema(row) -> SessionSummaryOut:
    return SessionSummaryOut(
        session_id=row.session_id,
        role=RoleOut(role_id=row.role_id, role_title=row.role_title),
        practice_focus=row.practice_focus,
        status=row.status,
        progress=row.progress,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _feedback_item_dicts(items) -> list[dict]:
    return [{"criterion_id": i.criterion_id, "title": i.title, "detail": i.detail} for i in items]


def _skill_labels_from_plan(plan: dict) -> list[str]:
    snap = (plan or {}).get("snapshot") or {}
    pool = (snap.get("professional_skills") or []) + (snap.get("reframed_skills") or [])
    return [s["skill"] for s in pool if s.get("skill")]


class InterviewService:
    def __init__(
        self,
        *,
        transcriber,
        redactor,
        feedback_service,
        rng: random.Random | None = None,
        clock=None,
        retention_days: int = 30,
        repo=interview_repo,
        accounts=accounts_repo,
    ) -> None:
        self._transcriber = transcriber
        self._redactor = redactor
        self._feedback_service = feedback_service
        self._rng = rng or random.Random()
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._retention_days = retention_days
        self._repo = repo
        self._accounts = accounts

    # -- saved-journey prerequisite --------------------------------------------

    async def _load_allowed_roles(self, session, username):
        plan = await self._accounts.get_plan(session, username) or {}
        selected = plan.get("selectedRole") or {}
        selected_role_id = selected.get("role_id")
        if not selected_role_id:
            raise InterviewError("journey_prerequisite_incomplete")

        snap = plan.get("snapshot") or {}
        recommended_ids = [
            r.get("role_id") for r in (snap.get("recommended_roles") or []) if r.get("role_id")
        ]
        ordered_ids, seen = [], set()
        for rid in [selected_role_id, *recommended_ids]:
            if rid not in seen:
                seen.add(rid)
                ordered_ids.append(rid)

        roles = await self._repo.get_roles_by_ids(session, ordered_ids)
        role_map = {r.role_id: r for r in roles}
        available = [role_map[rid] for rid in ordered_ids if rid in role_map]
        selected_role = role_map.get(selected_role_id)
        if selected_role is None:
            raise InterviewError("journey_prerequisite_incomplete")
        return selected_role, available

    async def get_setup(self, session, username: str) -> InterviewSetupOut:
        selected_role, available = await self._load_allowed_roles(session, username)
        return InterviewSetupOut(
            selected_role=RoleOut(role_id=selected_role.role_id, role_title=selected_role.role_title),
            available_roles=[RoleOut(role_id=r.role_id, role_title=r.role_title) for r in available],
        )

    # -- session lifecycle ------------------------------------------------------

    async def create_session(
        self, session, username: str, role_id: str, practice_focus: str
    ) -> tuple[SessionDetailOut, bool]:
        _selected, available = await self._load_allowed_roles(session, username)
        if role_id not in {r.role_id for r in available}:
            raise InterviewError("invalid_setup")

        existing = await self._repo.find_session(session, username, role_id, practice_focus)
        if existing is not None:
            detail = await self._repo.get_session_detail(session, username, existing.session_id)
            return _session_detail_to_schema(detail), False

        pool = await self._repo.get_question_pool(session, role_id)
        questions = _select_questions(self._rng, pool, role_id, practice_focus, exclude_ids=set())
        new_session = await self._repo.insert_session(session, username, role_id, practice_focus)
        await self._repo.insert_placeholder_responses(
            session, new_session.session_id, [q.question_id for q in questions]
        )
        detail = await self._repo.get_session_detail(session, username, new_session.session_id)
        return _session_detail_to_schema(detail), True

    async def list_sessions(self, session, username: str) -> list[SessionSummaryOut]:
        rows = await self._repo.list_sessions(session, username)
        return [_summary_to_schema(r) for r in rows]

    async def get_session(self, session, username: str, session_id: str) -> SessionDetailOut:
        detail = await self._repo.get_session_detail(session, username, session_id)
        if detail is None:
            raise InterviewError("interview_resource_not_found")
        return _session_detail_to_schema(detail)

    async def refresh_session(self, session, username: str, session_id: str) -> SessionDetailOut:
        old = await self._repo.get_session_detail(session, username, session_id)
        if old is None:
            raise InterviewError("interview_resource_not_found")
        old_question_ids = {q.question_id for q in old.questions}

        deleted = await self._repo.delete_session(session, username, session_id)
        if not deleted:
            raise InterviewError("interview_resource_not_found")

        pool = await self._repo.get_question_pool(session, old.role_id)
        questions = _select_questions(
            self._rng, pool, old.role_id, old.practice_focus, exclude_ids=old_question_ids
        )
        new_session = await self._repo.insert_session(session, username, old.role_id, old.practice_focus)
        await self._repo.insert_placeholder_responses(
            session, new_session.session_id, [q.question_id for q in questions]
        )
        detail = await self._repo.get_session_detail(session, username, new_session.session_id)
        return _session_detail_to_schema(detail)

    async def delete_session(self, session, username: str, session_id: str) -> None:
        deleted = await self._repo.delete_session(session, username, session_id)
        if not deleted:
            raise InterviewError("interview_resource_not_found")

    # -- recording / feedback ----------------------------------------------------

    async def process_recording(
        self, session, username: str, session_id: str, sequence_no: int, audio,
    ) -> RecordingOutcome:
        detail = await self._repo.get_session_detail(session, username, session_id)
        if detail is None:
            raise InterviewError("interview_resource_not_found")
        slot = next((q for q in detail.questions if q.sequence_no == sequence_no), None)
        if slot is None:
            raise InterviewError("interview_resource_not_found")

        try:
            transcription = await self._transcriber.transcribe(audio)
        except TranscriptionError as exc:
            raise InterviewError(exc.code) from exc

        redacted = self._redactor.redact(transcription.transcript)

        locked = await self._repo.lock_latest_attempt(session, username, session_id, sequence_no)
        if locked is None:
            raise InterviewError("interview_resource_not_found")
        if locked.transcript is None:
            response_id = locked.response_id
        else:
            response_id = await self._repo.insert_attempt(
                session, session_id, slot.question_id, sequence_no, locked.attempt_no + 1
            )

        await self._repo.save_transcript_processing(
            session, response_id, redacted, transcription.detected_language, transcription.duration_s
        )

        return await self._run_feedback(
            session, username, session_id, response_id, slot, role_title=detail.role_title
        )

    async def retry_feedback(self, session, username: str, response_id: int) -> RecordingOutcome:
        attempt = await self._repo.get_attempt_by_id(session, username, response_id)
        if attempt is None or attempt.transcript is None or attempt.content_purged_at is not None:
            raise InterviewError("interview_resource_not_found")

        detail = await self._repo.get_session_detail(session, username, attempt.session_id)
        if detail is None:
            raise InterviewError("interview_resource_not_found")
        slot = next((q for q in detail.questions if q.sequence_no == attempt.sequence_no), None)
        if slot is None:
            raise InterviewError("interview_resource_not_found")

        return await self._run_feedback(
            session, username, attempt.session_id, response_id, slot, role_title=detail.role_title
        )

    async def _run_feedback(
        self, session, username, session_id, response_id, slot, *, role_title: str | None,
    ) -> RecordingOutcome:
        attempt_row = await self._repo.get_attempt_by_id(session, username, response_id)
        criteria_map = await self._repo.get_criteria_for_questions(session, [slot.question_id])
        plan = await self._accounts.get_plan(session, username) or {}

        feedback_input = FeedbackInput(
            question_text=slot.question_text,
            category=slot.category,
            role_title=role_title,
            transcript=attempt_row.transcript,
            criteria=criteria_map.get(slot.question_id, []),
            skill_labels=_skill_labels_from_plan(plan),
            interview_method_sources=slot.interview_method_sources,
            authoring_method=slot.authoring_method,
            answer_framework=slot.answer_framework,
            answer_guidance=slot.answer_guidance,
            strong_evidence_signals=slot.strong_evidence_signals,
            watch_out_for=slot.watch_out_for,
            follow_up_question=slot.follow_up_question,
        )

        try:
            result = await self._feedback_service.evaluate(feedback_input)
        except FeedbackError:
            await self._repo.save_feedback_error(session, response_id, "feedback_service_unavailable")
            updated = await self._repo.get_attempt_by_id(session, username, response_id)
            return RecordingOutcome(status="feedback_error", attempt=_attempt_to_schema(updated))

        await self._repo.save_feedback_ready(
            session, response_id, result.summary,
            _feedback_item_dicts(result.strengths), _feedback_item_dicts(result.improvements),
            [i.tag for i in result.strengths], [i.tag for i in result.improvements],
            {"model": result.model, "tokens_in": result.tokens_in, "tokens_out": result.tokens_out},
        )
        await self._repo.mark_completed_if_all_ready(session, session_id)
        updated = await self._repo.get_attempt_by_id(session, username, response_id)
        return RecordingOutcome(status="ready", attempt=_attempt_to_schema(updated))

    # -- areas / retention --------------------------------------------------------

    async def get_areas(self, session, username: str) -> AreasOut:
        rows = await self._repo.get_recurring_areas(session, username)
        return AreasOut(
            improvements=[
                AreaOut(criterion_id=r.criterion_id, title=r.title, response_count=r.response_count)
                for r in rows if r.kind == "improvement"
            ],
            strengths=[
                AreaOut(criterion_id=r.criterion_id, title=r.title, response_count=r.response_count)
                for r in rows if r.kind == "strength"
            ],
        )

    async def purge_expired_content(self, session) -> int:
        cutoff = self._clock() - timedelta(days=self._retention_days)
        return await self._repo.purge_expired_content(session, cutoff)
