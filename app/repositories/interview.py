"""SQL for the E7 AI Interview Coach. Repositories run SQL; the router owns the
transaction (see app/api/interview.py). Every single-resource query is scoped to the
owning username so a missing row and a cross-user row are indistinguishable to the
caller (both come back as None / False), which the API layer turns into a single
404 interview_resource_not_found.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class RoleRow:
    role_id: str
    role_title: str


@dataclass
class QuestionRow:
    question_id: str
    role_id: str | None
    category: str
    difficulty: str
    question_text: str
    interview_method_sources: str
    authoring_method: str
    answer_framework: str
    answer_guidance: str
    strong_evidence_signals: str
    watch_out_for: str
    follow_up_question: str


@dataclass
class CriterionRow:
    criterion_id: str
    criterion: str
    positive_feedback_tag: str
    improvement_feedback_tag: str
    prohibited_inference: str
    what_ai_checks: str = ""


@dataclass
class SessionRow:
    session_id: str
    status: str
    created_at: object
    updated_at: object


@dataclass
class SessionSummaryRow:
    session_id: str
    role_id: str
    role_title: str
    practice_focus: str
    status: str
    progress: int
    created_at: object
    updated_at: object


@dataclass
class AttemptRow:
    response_id: int
    session_id: str
    question_id: str
    sequence_no: int
    attempt_no: int
    transcript: str | None
    detected_language: str | None
    audio_duration_s: float | None
    feedback_status: str
    feedback_summary: str | None
    worked_well: list
    what_to_improve: list
    strength_tags: list
    improvement_tags: list
    error_code: str | None
    created_at: object
    updated_at: object
    content_purged_at: object | None


@dataclass
class QuestionSlotDetail:
    sequence_no: int
    question_id: str
    role_id: str | None
    question_text: str
    category: str
    difficulty: str
    interview_method_sources: str
    authoring_method: str
    answer_framework: str
    answer_guidance: str
    strong_evidence_signals: str
    watch_out_for: str
    follow_up_question: str
    attempts: list[AttemptRow] = field(default_factory=list)


@dataclass
class SessionDetail:
    session_id: str
    role_id: str
    role_title: str
    practice_focus: str
    status: str
    created_at: object
    updated_at: object
    questions: list[QuestionSlotDetail] = field(default_factory=list)


@dataclass
class AreaRow:
    kind: str  # "strength" or "improvement"
    criterion_id: str
    title: str
    response_count: int


def _as_list(value) -> list:
    if value is None:
        return []
    return json.loads(value) if isinstance(value, str) else value


def _attempt_row_from(r) -> AttemptRow:
    return AttemptRow(
        response_id=r.response_id,
        session_id=r.session_id if hasattr(r, "session_id") else None,
        question_id=r.question_id,
        sequence_no=r.sequence_no,
        attempt_no=r.attempt_no,
        transcript=r.transcript,
        detected_language=r.detected_language,
        audio_duration_s=r.audio_duration_s,
        feedback_status=r.feedback_status,
        feedback_summary=r.feedback_summary,
        worked_well=_as_list(r.worked_well),
        what_to_improve=_as_list(r.what_to_improve),
        strength_tags=list(r.strength_tags or []),
        improvement_tags=list(r.improvement_tags or []),
        error_code=r.error_code,
        created_at=r.created_at,
        updated_at=r.updated_at,
        content_purged_at=r.content_purged_at,
    )


# -- roles ---------------------------------------------------------------------

async def get_roles_by_ids(session: AsyncSession, role_ids: list[str]) -> list[RoleRow]:
    if not role_ids:
        return []
    rows = (
        await session.execute(
            text("SELECT role_id, role_title FROM roles WHERE role_id = ANY(:ids)"),
            {"ids": role_ids},
        )
    ).all()
    return [RoleRow(r.role_id, r.role_title) for r in rows]


# -- question pool / criteria ----------------------------------------------------

async def get_question_pool(session: AsyncSession, role_id: str) -> list[QuestionRow]:
    """Every general question (role_id IS NULL) plus every question for role_id.

    review_status is informational and is never filtered here (confirmed decision).
    """
    rows = (
        await session.execute(
            text(
                "SELECT question_id, role_id, category, difficulty, question_text, "
                "interview_method_sources, authoring_method, answer_framework, "
                "answer_guidance, strong_evidence_signals, watch_out_for, follow_up_question "
                "FROM interview_question WHERE role_id IS NULL OR role_id = :rid"
            ),
            {"rid": role_id},
        )
    ).all()
    return [
        QuestionRow(
            r.question_id, r.role_id, r.category, r.difficulty, r.question_text,
            r.interview_method_sources, r.authoring_method, r.answer_framework,
            r.answer_guidance, r.strong_evidence_signals, r.watch_out_for, r.follow_up_question,
        )
        for r in rows
    ]


async def get_criteria_for_questions(
    session: AsyncSession, question_ids: list[str]
) -> dict[str, list[CriterionRow]]:
    if not question_ids:
        return {}
    rows = (
        await session.execute(
            text(
                "SELECT qr.question_id, r.criterion_id, r.criterion, "
                "r.positive_feedback_tag, r.improvement_feedback_tag, r.prohibited_inference, "
                "r.what_ai_checks "
                "FROM interview_question_rubric qr "
                "JOIN ai_evaluation_rubric r ON r.criterion_id = qr.criterion_id "
                "WHERE qr.question_id = ANY(:ids)"
            ),
            {"ids": question_ids},
        )
    ).all()
    out: dict[str, list[CriterionRow]] = {}
    for r in rows:
        out.setdefault(r.question_id, []).append(
            CriterionRow(
                r.criterion_id, r.criterion, r.positive_feedback_tag,
                r.improvement_feedback_tag, r.prohibited_inference, r.what_ai_checks,
            )
        )
    return out


# -- session setup ---------------------------------------------------------------

async def find_session(
    session: AsyncSession, username: str, role_id: str, practice_focus: str
) -> SessionRow | None:
    row = (
        await session.execute(
            text(
                "SELECT session_id, status, created_at, updated_at FROM interview_session "
                "WHERE username = :u AND role_id = :rid AND practice_focus = :focus"
            ),
            {"u": username, "rid": role_id, "focus": practice_focus},
        )
    ).first()
    return SessionRow(row.session_id, row.status, row.created_at, row.updated_at) if row else None


async def insert_session(
    session: AsyncSession, username: str, role_id: str, practice_focus: str
) -> SessionRow:
    row = (
        await session.execute(
            text(
                "INSERT INTO interview_session (username, role_id, practice_focus) "
                "VALUES (:u, :rid, :focus) "
                "RETURNING session_id, status, created_at, updated_at"
            ),
            {"u": username, "rid": role_id, "focus": practice_focus},
        )
    ).first()
    return SessionRow(row.session_id, row.status, row.created_at, row.updated_at)


async def insert_placeholder_responses(
    session: AsyncSession, session_id: str, question_ids: list[str]
) -> None:
    """One pending attempt-1 row per question, in presentation order (sequence_no 1-5)."""
    params = [
        {"sid": session_id, "qid": qid, "seq": i}
        for i, qid in enumerate(question_ids, start=1)
    ]
    await session.execute(
        text(
            "INSERT INTO interview_response (session_id, question_id, sequence_no, attempt_no, feedback_status) "
            "VALUES (:sid, :qid, :seq, 1, 'pending')"
        ),
        params,
    )


# -- listing / detail --------------------------------------------------------------

async def list_sessions(session: AsyncSession, username: str) -> list[SessionSummaryRow]:
    rows = (
        await session.execute(
            text(
                "SELECT s.session_id, s.role_id, r.role_title, s.practice_focus, s.status, "
                "s.created_at, s.updated_at, "
                "(SELECT count(DISTINCT ir.sequence_no) FROM interview_response ir "
                " WHERE ir.session_id = s.session_id AND ir.feedback_status = 'ready') AS progress "
                "FROM interview_session s JOIN roles r ON r.role_id = s.role_id "
                "WHERE s.username = :u ORDER BY s.created_at DESC"
            ),
            {"u": username},
        )
    ).all()
    return [
        SessionSummaryRow(
            r.session_id, r.role_id, r.role_title, r.practice_focus, r.status,
            r.progress, r.created_at, r.updated_at,
        )
        for r in rows
    ]


async def get_session_detail(
    session: AsyncSession, username: str, session_id: str
) -> SessionDetail | None:
    session_row = (
        await session.execute(
            text(
                "SELECT s.session_id, s.role_id, r.role_title, s.practice_focus, s.status, "
                "s.created_at, s.updated_at "
                "FROM interview_session s JOIN roles r ON r.role_id = s.role_id "
                "WHERE s.username = :u AND s.session_id = :sid"
            ),
            {"u": username, "sid": session_id},
        )
    ).first()
    if session_row is None:
        return None

    attempt_rows = (
        await session.execute(
            text(
                "SELECT ir.response_id, ir.sequence_no, ir.attempt_no, ir.question_id, "
                "q.role_id, q.question_text, q.category, q.difficulty, "
                "q.interview_method_sources, q.authoring_method, q.answer_framework, "
                "q.answer_guidance, q.strong_evidence_signals, q.watch_out_for, q.follow_up_question, "
                "ir.transcript, ir.detected_language, ir.audio_duration_s, "
                "ir.feedback_status, ir.feedback_summary, ir.worked_well, ir.what_to_improve, "
                "ir.strength_tags, ir.improvement_tags, ir.error_code, "
                "ir.created_at, ir.updated_at, ir.content_purged_at "
                "FROM interview_response ir JOIN interview_question q ON q.question_id = ir.question_id "
                "WHERE ir.session_id = :sid ORDER BY ir.sequence_no ASC, ir.attempt_no ASC"
            ),
            {"sid": session_id},
        )
    ).all()

    questions: dict[int, QuestionSlotDetail] = {}
    for r in attempt_rows:
        slot = questions.get(r.sequence_no)
        if slot is None:
            slot = QuestionSlotDetail(
                sequence_no=r.sequence_no, question_id=r.question_id, role_id=r.role_id,
                question_text=r.question_text, category=r.category, difficulty=r.difficulty,
                interview_method_sources=r.interview_method_sources,
                authoring_method=r.authoring_method, answer_framework=r.answer_framework,
                answer_guidance=r.answer_guidance, strong_evidence_signals=r.strong_evidence_signals,
                watch_out_for=r.watch_out_for, follow_up_question=r.follow_up_question,
            )
            questions[r.sequence_no] = slot
        slot.attempts.append(_attempt_row_from(r))

    return SessionDetail(
        session_id=session_row.session_id,
        role_id=session_row.role_id,
        role_title=session_row.role_title,
        practice_focus=session_row.practice_focus,
        status=session_row.status,
        created_at=session_row.created_at,
        updated_at=session_row.updated_at,
        questions=[questions[seq] for seq in sorted(questions)],
    )


# -- attempt lifecycle --------------------------------------------------------------

async def lock_latest_attempt(
    session: AsyncSession, username: str, session_id: str, sequence_no: int
) -> AttemptRow | None:
    """The newest attempt for this question slot, row-locked so a concurrent retry on
    the same slot cannot race the attempt-number decision."""
    row = (
        await session.execute(
            text(
                "SELECT ir.response_id, ir.session_id, ir.question_id, ir.sequence_no, ir.attempt_no, "
                "ir.transcript, ir.detected_language, ir.audio_duration_s, "
                "ir.feedback_status, ir.feedback_summary, ir.worked_well, ir.what_to_improve, "
                "ir.strength_tags, ir.improvement_tags, ir.error_code, "
                "ir.created_at, ir.updated_at, ir.content_purged_at "
                "FROM interview_response ir "
                "JOIN interview_session s ON s.session_id = ir.session_id "
                "WHERE s.username = :u AND ir.session_id = :sid AND ir.sequence_no = :seq "
                "ORDER BY ir.attempt_no DESC LIMIT 1 FOR UPDATE OF ir"
            ),
            {"u": username, "sid": session_id, "seq": sequence_no},
        )
    ).first()
    return _attempt_row_from(row) if row else None


async def insert_attempt(
    session: AsyncSession, session_id: str, question_id: str, sequence_no: int, attempt_no: int
) -> int:
    row = (
        await session.execute(
            text(
                "INSERT INTO interview_response (session_id, question_id, sequence_no, attempt_no, feedback_status) "
                "VALUES (:sid, :qid, :seq, :attempt_no, 'pending') "
                "RETURNING response_id"
            ),
            {"sid": session_id, "qid": question_id, "seq": sequence_no, "attempt_no": attempt_no},
        )
    ).first()
    return row.response_id


async def save_transcript_processing(
    session: AsyncSession, response_id: int, transcript: str,
    detected_language: str, audio_duration_s: float,
) -> None:
    await session.execute(
        text(
            "UPDATE interview_response SET transcript = :t, detected_language = :lang, "
            "audio_duration_s = :dur, feedback_status = 'processing', updated_at = now() "
            "WHERE response_id = :rid"
        ),
        {"rid": response_id, "t": transcript, "lang": detected_language, "dur": audio_duration_s},
    )


async def save_feedback_ready(
    session: AsyncSession, response_id: int, feedback_summary: str,
    worked_well: list[dict], what_to_improve: list[dict],
    strength_tags: list[str], improvement_tags: list[str], model_metadata: dict,
) -> None:
    await session.execute(
        text(
            "UPDATE interview_response SET feedback_status = 'ready', "
            "feedback_summary = :summary, worked_well = cast(:ww as jsonb), "
            "what_to_improve = cast(:wti as jsonb), strength_tags = :stags, "
            "improvement_tags = :itags, model_metadata = cast(:meta as jsonb), "
            "processed_at = now(), updated_at = now() "
            "WHERE response_id = :rid"
        ),
        {
            "rid": response_id, "summary": feedback_summary,
            "ww": json.dumps(worked_well), "wti": json.dumps(what_to_improve),
            "stags": strength_tags, "itags": improvement_tags,
            "meta": json.dumps(model_metadata),
        },
    )


async def save_feedback_error(session: AsyncSession, response_id: int, error_code: str) -> None:
    await session.execute(
        text(
            "UPDATE interview_response SET feedback_status = 'error', error_code = :code, "
            "processed_at = now(), updated_at = now() WHERE response_id = :rid"
        ),
        {"rid": response_id, "code": error_code},
    )


async def mark_completed_if_all_ready(session: AsyncSession, session_id: str) -> None:
    """Flip to completed only when all five question positions have a ready attempt.
    A no-op (and safe to call after every feedback save) until that is true."""
    await session.execute(
        text(
            "UPDATE interview_session s SET status = 'completed', updated_at = now() "
            "WHERE s.session_id = :sid AND ("
            "  SELECT count(DISTINCT ir.sequence_no) FROM interview_response ir "
            "  WHERE ir.session_id = :sid AND ir.feedback_status = 'ready'"
            ") = 5"
        ),
        {"sid": session_id},
    )


async def delete_session(session: AsyncSession, username: str, session_id: str) -> bool:
    result = await session.execute(
        text("DELETE FROM interview_session WHERE session_id = :sid AND username = :u"),
        {"u": username, "sid": session_id},
    )
    return result.rowcount > 0


async def get_attempt_by_id(
    session: AsyncSession, username: str, response_id: int
) -> AttemptRow | None:
    row = (
        await session.execute(
            text(
                "SELECT ir.response_id, ir.session_id, ir.question_id, ir.sequence_no, ir.attempt_no, "
                "ir.transcript, ir.detected_language, ir.audio_duration_s, "
                "ir.feedback_status, ir.feedback_summary, ir.worked_well, ir.what_to_improve, "
                "ir.strength_tags, ir.improvement_tags, ir.error_code, "
                "ir.created_at, ir.updated_at, ir.content_purged_at "
                "FROM interview_response ir "
                "JOIN interview_session s ON s.session_id = ir.session_id "
                "WHERE s.username = :u AND ir.response_id = :rid"
            ),
            {"u": username, "rid": response_id},
        )
    ).first()
    return _attempt_row_from(row) if row else None


# -- aggregation / retention -------------------------------------------------------

async def get_recurring_areas(session: AsyncSession, username: str) -> list[AreaRow]:
    """The latest ready attempt per (session, question) across all of the user's
    current sessions, with its tags unnested and counted once each, split by kind."""
    rows = (
        await session.execute(
            text(
                "WITH latest_ready AS ("
                "  SELECT DISTINCT ON (ir.session_id, ir.sequence_no) "
                "    ir.strength_tags, ir.improvement_tags "
                "  FROM interview_response ir "
                "  JOIN interview_session s ON s.session_id = ir.session_id "
                "  WHERE s.username = :u AND ir.feedback_status = 'ready' "
                "  ORDER BY ir.session_id, ir.sequence_no, ir.attempt_no DESC"
                "), strength_tags AS ("
                "  SELECT unnest(strength_tags) AS tag FROM latest_ready"
                "), improvement_tags AS ("
                "  SELECT unnest(improvement_tags) AS tag FROM latest_ready"
                ") "
                "SELECT 'strength' AS kind, r.criterion_id, "
                "initcap(replace(r.criterion, '_', ' ')) AS title, count(*) AS response_count "
                "FROM strength_tags st JOIN ai_evaluation_rubric r ON r.positive_feedback_tag = st.tag "
                "GROUP BY r.criterion_id, r.criterion "
                "UNION ALL "
                "SELECT 'improvement' AS kind, r.criterion_id, "
                "initcap(replace(r.criterion, '_', ' ')) AS title, count(*) AS response_count "
                "FROM improvement_tags it JOIN ai_evaluation_rubric r ON r.improvement_feedback_tag = it.tag "
                "GROUP BY r.criterion_id, r.criterion "
                "ORDER BY response_count DESC, title ASC"
            ),
            {"u": username},
        )
    ).all()
    return [AreaRow(r.kind, r.criterion_id, r.title, r.response_count) for r in rows]


async def purge_expired_content(session: AsyncSession, cutoff) -> int:
    """Clears transcript/feedback detail for responses created before cutoff that have
    not already been purged. Session metadata, question ids, attempt numbering,
    statuses, and normalized tags are untouched."""
    result = await session.execute(
        text(
            "UPDATE interview_response SET transcript = NULL, feedback_summary = NULL, "
            "worked_well = '[]'::jsonb, what_to_improve = '[]'::jsonb, "
            "model_metadata = '{}'::jsonb, content_purged_at = now(), updated_at = now() "
            "WHERE created_at < :cutoff AND content_purged_at IS NULL"
        ),
        {"cutoff": cutoff},
    )
    return result.rowcount


@dataclass
class QuestionCoaching:
    question_id: str
    role_id: str | None
    question_text: str
    answer_framework: str
    answer_guidance: str
    strong_evidence_signals: str
    watch_out_for: str
    follow_up_question: str


async def get_question_coaching(
    session: AsyncSession, question_id: str
) -> QuestionCoaching | None:
    """Rubric-authored coaching for one question, used to ground Hera server-side.

    Authoring metadata (sources, authoring_method) is intentionally not selected.
    """
    row = (
        await session.execute(
            text(
                "SELECT question_id, role_id, question_text, answer_framework, "
                "answer_guidance, strong_evidence_signals, watch_out_for, follow_up_question "
                "FROM interview_question WHERE question_id = :qid"
            ),
            {"qid": question_id},
        )
    ).first()
    if row is None:
        return None
    return QuestionCoaching(
        question_id=row.question_id,
        role_id=row.role_id,
        question_text=row.question_text,
        answer_framework=row.answer_framework,
        answer_guidance=row.answer_guidance,
        strong_evidence_signals=row.strong_evidence_signals,
        watch_out_for=row.watch_out_for,
        follow_up_question=row.follow_up_question,
    )
