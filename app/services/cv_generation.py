"""Grounded refreshed-CV generation and wording refinement."""

from __future__ import annotations

import copy
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.repositories import accounts as accounts_repo
from app.services.llm import LlmError

logger = logging.getLogger("rerouteher")

_DEFAULT_PERSONAL = {
    "name": "",
    "email": "",
    "phone": "",
    "location": "",
}

_BANNED_OUTPUT_TERMS = (
    "caregiv",
    "motherhood",
    "mother",
    "family",
    "childcare",
    "career break",
    "stay-at-home",
)

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")
_NRIC_RE = re.compile(r"(?<!\w)\d{6}[-\s]?\d{2}[-\s]?\d{4}(?!\w)")
_ADDRESS_RE = re.compile(
    r"(?im)^\s*(?:address|alamat|street|jalan|road|lorong|no\.)\s*[:#-]?.*$"
)


class CvGenerationError(RuntimeError):
    """Stable service error mapped by the API layer."""

    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        self.message = message or code
        super().__init__(self.message)


@dataclass(frozen=True)
class CvGenerateResult:
    role_id: str
    status: str
    generated_at: datetime
    draft: dict[str, Any]


@dataclass(frozen=True)
class CvImproveResult:
    section: str
    experience_index: int | None
    suggestion: str
    evidence: str


_DRAFT_TOOL = {
    "function_declarations": [
        {
            "name": "submit_cv_draft",
            "description": (
                "Return the grounded wording for a professional CV. Do not change "
                "source metadata, invent facts, or mention protected personal context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "skills": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "experiences": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_index": {"type": "integer"},
                                "bullets": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "text": {"type": "string"},
                                            "evidence": {"type": "string"},
                                        },
                                        "required": ["text", "evidence"],
                                    },
                                },
                            },
                            "required": ["source_index", "bullets"],
                        },
                    },
                },
                "required": ["summary", "skills", "experiences"],
            },
        }
    ]
}

_IMPROVE_TOOL = {
    "function_declarations": [
        {
            "name": "submit_cv_improvement",
            "description": (
                "Return one grounded wording suggestion and the exact source excerpt "
                "supporting it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "suggestion": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["suggestion", "evidence"],
            },
        }
    ]
}

_DRAFT_SYSTEM_PROMPT = (
    "You are a professional CV editor. Write concise, ATS-friendly wording for the "
    "selected target role using only the supplied journey evidence. Return exactly one "
    "submit_cv_draft tool call. Write a neutral summary of one to three short sentences, "
    "use action-led experience bullets, and select only exact supplied skill labels. "
    "Do not use first-person language. Do not invent employers, dates, achievements, "
    "metrics, responsibilities, qualifications, or skills. Do not mention caregiving, "
    "motherhood, family, childcare, a career break, or other protected personal context. "
    "Every experience bullet must include its source index and an exact evidence excerpt."
)

_IMPROVE_SYSTEM_PROMPT = (
    "You are a professional CV editor. Improve only the supplied section using only "
    "the supplied journey evidence. Return exactly one submit_cv_improvement tool call. "
    "Do not invent facts, metrics, employers, dates, achievements, or skills. Do not use "
    "first-person language or mention caregiving, motherhood, family, childcare, a career "
    "break, or other protected personal context. Include the exact source excerpt that "
    "supports the suggestion."
)


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _redact_text(value: Any) -> str:
    """Keep useful experience wording while excluding common direct identifiers."""
    text = _safe_text(value)
    text = _EMAIL_RE.sub("[redacted email]", text)
    text = _PHONE_RE.sub("[redacted phone]", text)
    text = _NRIC_RE.sub("[redacted identifier]", text)
    return _ADDRESS_RE.sub("[redacted address]", text)


def _label(value: Any) -> str:
    if isinstance(value, str):
        return _safe_text(value)
    if isinstance(value, dict):
        for key in ("skill", "skill_name", "name", "label", "canonical_name", "role"):
            if value.get(key):
                return _safe_text(value[key])
    return ""


def _role_id(role: Any) -> str:
    if isinstance(role, str):
        return role.strip()
    if isinstance(role, dict):
        return _safe_text(role.get("role_id") or role.get("roleId") or role.get("id"))
    return ""


def _role_name(role: Any) -> str:
    if isinstance(role, str):
        return role.strip()
    if isinstance(role, dict):
        return _safe_text(role.get("role") or role.get("name") or role.get("title"))
    return ""


def _roles_from_plan(plan: dict[str, Any]) -> list[dict[str, str]]:
    snapshot = plan.get("snapshot") or {}
    candidates = [plan.get("selectedRole")]
    candidates.extend(snapshot.get("recommended_roles") or snapshot.get("recommendedRoles") or [])

    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for candidate in candidates:
        role_id = _role_id(candidate)
        role_name = _role_name(candidate)
        if role_id and role_id not in seen:
            result.append({"role_id": role_id, "role": role_name})
            seen.add(role_id)
    return result


def _skills_from_plan(plan: dict[str, Any]) -> list[str]:
    snapshot = plan.get("snapshot") or {}
    values: list[str] = []
    for key in (
        "professional_skills",
        "professionalSkills",
        "confirmed_skills",
        "confirmedSkills",
        "reframed_skills",
        "reframedSkills",
    ):
        raw_values = snapshot.get(key) or plan.get(key) or []
        for value in raw_values:
            label = _label(value)
            if label and label not in values:
                values.append(label)

    return values


def _experience_sources(plan: dict[str, Any]) -> list[dict[str, str]]:
    cv = plan.get("cv") or {}
    sources: list[dict[str, str]] = []
    for raw in cv.get("experiences") or []:
        if not isinstance(raw, dict):
            continue
        source = {
            "title": _redact_text(raw.get("title")),
            "organisation": _redact_text(raw.get("organisation") or raw.get("organization")),
            "start": _redact_text(raw.get("start")),
            "end": _redact_text(raw.get("end")),
            "description": _redact_text(raw.get("description")),
        }
        if any(source.values()):
            sources.append(source)
    return sources


def _activity_labels(plan: dict[str, Any]) -> list[str]:
    raw_break = plan.get("break") or {}
    labels: list[str] = []
    for activity in raw_break.get("activities") or []:
        label = _label(activity)
        if label and label not in labels:
            labels.append(label)
    return labels


def _source_context(
    plan: dict[str, Any],
    role: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, str]], list[str]]:
    experiences = _experience_sources(plan)
    skills = _skills_from_plan(plan)
    context = {
        "target_role": role,
        "experiences": experiences,
        "skills": skills,
        "additional_activity_labels": _activity_labels(plan),
    }
    return context, experiences, skills


def _draft_book(plan: dict[str, Any]) -> dict[str, Any]:
    book = plan.get("cvDraft")
    return copy.deepcopy(book) if isinstance(book, dict) else {}


def _existing_draft(plan: dict[str, Any], role_id: str) -> dict[str, Any] | None:
    book = _draft_book(plan)
    drafts = book.get("drafts")
    if not isinstance(drafts, dict):
        return None
    draft = drafts.get(role_id)
    return copy.deepcopy(draft) if isinstance(draft, dict) and draft else None


def _personal_fields(plan: dict[str, Any], role_id: str) -> dict[str, Any]:
    existing = _existing_draft(plan, role_id)
    if existing and isinstance(existing.get("personal"), dict):
        return copy.deepcopy(existing["personal"])

    book = _draft_book(plan)
    personal = book.get("personal")
    if isinstance(personal, dict):
        return copy.deepcopy(personal)
    return copy.deepcopy(_DEFAULT_PERSONAL)


def _validate_setup(
    plan: Any,
    requested_role_id: str | None,
) -> tuple[dict[str, str], list[dict[str, str]], list[str], dict[str, Any]]:
    if not isinstance(plan, dict):
        raise CvGenerationError(
            "journey_prerequisite_incomplete",
            "Complete your saved journey before generating a CV.",
        )

    snapshot = plan.get("snapshot")
    roles = _roles_from_plan(plan)
    selected = plan.get("selectedRole")
    if not plan.get("cvParsed") or not isinstance(snapshot, dict) or not _role_id(selected) or not roles:
        raise CvGenerationError(
            "journey_prerequisite_incomplete",
            "Complete your skill snapshot and select a matched target role first.",
        )

    role_by_id = {role["role_id"]: role for role in roles}
    active_id = requested_role_id or _role_id(selected)
    if active_id not in role_by_id:
        raise CvGenerationError(
            "invalid_cv_setup",
            "The selected target role is not available in your saved journey.",
        )

    role = role_by_id[active_id]
    context, experiences, skills = _source_context(plan, role)
    if not experiences and not skills:
        raise CvGenerationError(
            "journey_prerequisite_incomplete",
            "Add at least one experience or skill before generating a CV.",
        )
    return role, experiences, skills, context


def _function_calls(content: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for part in content.get("parts") or []:
        call = part.get("functionCall") or part.get("function_call")
        if isinstance(call, dict):
            calls.append(call)
    return calls


def _one_tool_args(result: Any, expected_name: str) -> dict[str, Any]:
    content = getattr(result, "content", None)
    if not isinstance(content, dict):
        raise CvGenerationError("invalid_cv_content", "no content dict in response")

    calls = _function_calls(content)
    if len(calls) != 1 or calls[0].get("name") != expected_name:
        raise CvGenerationError(
            "invalid_cv_content",
            f"expected exactly one {expected_name} call, got {len(calls)}",
        )

    args = calls[0].get("args") or calls[0].get("arguments")
    if not isinstance(args, dict):
        raise CvGenerationError("invalid_cv_content", "tool call args missing")
    return args


def _contains_banned_term(text: str) -> bool:
    lowered = text.casefold()
    return any(term in lowered for term in _BANNED_OUTPUT_TERMS)


def _validate_summary(value: Any) -> str:
    summary = _safe_text(value)
    if not summary:
        raise CvGenerationError("invalid_cv_content", "summary empty")
    if len(summary) > 800:
        raise CvGenerationError("invalid_cv_content", "summary too long")
    if _contains_banned_term(summary):
        raise CvGenerationError("invalid_cv_content", "summary has banned term")
    if re.search(r"\b(i|me|my|we|our)\b", summary, re.IGNORECASE):
        raise CvGenerationError("invalid_cv_content", "summary has first-person pronoun")
    return summary


def _validate_bullet(value: Any) -> str:
    text = _safe_text(value)
    if not text:
        raise CvGenerationError("invalid_cv_content", "bullet empty")
    if len(text) > 400:
        raise CvGenerationError("invalid_cv_content", "bullet too long")
    if "\n" in text:
        raise CvGenerationError("invalid_cv_content", "bullet has newline")
    if _contains_banned_term(text):
        raise CvGenerationError("invalid_cv_content", "bullet has banned term")
    if re.search(r"\b(i|me|my|we|our)\b", text, re.IGNORECASE):
        raise CvGenerationError("invalid_cv_content", "bullet has first-person pronoun")
    return text


def _source_text(source: dict[str, str]) -> str:
    return "\n".join(value for value in source.values() if value)


def _normalise_draft(
    args: dict[str, Any],
    *,
    role: dict[str, str],
    experiences: list[dict[str, str]],
    allowed_skills: list[str],
    personal: dict[str, Any],
) -> dict[str, Any]:
    summary = _validate_summary(args.get("summary"))
    raw_skills = args.get("skills")
    if not isinstance(raw_skills, list):
        raise CvGenerationError("invalid_cv_content", "skills not a list")

    skills: list[str] = []
    for raw_skill in raw_skills:
        skill = _safe_text(raw_skill)
        if not skill:
            raise CvGenerationError("invalid_cv_content", "empty skill label")
        if skill not in allowed_skills:
            raise CvGenerationError("invalid_cv_content", "skill not in allowlist")
        if skill in skills:
            raise CvGenerationError("invalid_cv_content", "duplicate skill")
        skills.append(skill)

    raw_experiences = args.get("experiences")
    if not isinstance(raw_experiences, list):
        raise CvGenerationError("invalid_cv_content", "experiences not a list")

    bullets_by_index: dict[int, list[str]] = {}
    for item in raw_experiences:
        if not isinstance(item, dict) or isinstance(item.get("source_index"), bool):
            raise CvGenerationError("invalid_cv_content", "experience item not an object")
        index = item.get("source_index")
        if not isinstance(index, int) or index < 0 or index >= len(experiences):
            raise CvGenerationError("invalid_cv_content", "source_index out of range")
        if index in bullets_by_index:
            raise CvGenerationError("invalid_cv_content", "duplicate source_index")

        raw_bullets = item.get("bullets")
        if not isinstance(raw_bullets, list):
            raise CvGenerationError("invalid_cv_content", "bullets not a list")
        if len(raw_bullets) > 4:
            raise CvGenerationError("invalid_cv_content", "too many bullets")

        source = experiences[index]
        source_text = _source_text(source)
        bullets: list[str] = []
        for raw_bullet in raw_bullets:
            if not isinstance(raw_bullet, dict):
                raise CvGenerationError("invalid_cv_content", "bullet not an object")
            text = _validate_bullet(raw_bullet.get("text"))
            evidence = _safe_text(raw_bullet.get("evidence"))
            if not evidence:
                raise CvGenerationError("invalid_cv_content", "bullet evidence empty")
            if evidence not in source_text:
                raise CvGenerationError("invalid_cv_content", "bullet evidence not an exact source excerpt")
            bullets.append(text)
        bullets_by_index[index] = bullets

    normalised_experiences: list[dict[str, str]] = []
    for index, source in enumerate(experiences):
        normalised_experiences.append(
            {
                "title": source["title"],
                "organisation": source["organisation"],
                "start": source["start"],
                "end": source["end"],
                "description": "\n".join(f"- {bullet}" for bullet in bullets_by_index.get(index, [])),
            }
        )

    return {
        "version": 3,
        "roleId": role["role_id"],
        "personal": copy.deepcopy(personal),
        "summary": summary,
        "skills": skills,
        "experiences": normalised_experiences,
        "careerBreak": None,
    }


def _prompt_for_draft(context: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "parts": [
                {
                    "text": (
                        "Create the refreshed CV draft from this allowlisted journey context. "
                        "The target role, source metadata, skill labels, and evidence are the "
                        "only facts available:\n"
                        + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
                    )
                }
            ],
        }
    ]


def _prompt_for_improvement(
    context: dict[str, Any],
    *,
    section: str,
    experience_index: int | None,
    current_text: str,
    previous_suggestions: list[str],
) -> list[dict[str, Any]]:
    request = {
        "section": section,
        "experience_index": experience_index,
        "current_text": _redact_text(current_text),
        "previous_suggestions": [_redact_text(value) for value in previous_suggestions],
        "journey_context": context,
    }
    return [
        {
            "role": "user",
            "parts": [
                {
                    "text": (
                        "Improve only this CV section using the supplied evidence:\n"
                        + json.dumps(request, ensure_ascii=False, separators=(",", ":"))
                    )
                }
            ],
        }
    ]


class CvGenerationService:
    def __init__(self, llm: Any, repo=None) -> None:
        self._llm = llm
        self._repo = repo or accounts_repo

    async def generate(
        self,
        session: Any,
        username: str,
        role_id: str | None = None,
        regenerate: bool = False,
    ) -> CvGenerateResult:
        plan = await self._repo.get_plan(session, username)
        role, experiences, skills, context = _validate_setup(plan, role_id)
        existing = _existing_draft(plan, role["role_id"])

        if existing is not None and not regenerate:
            return CvGenerateResult(
                role_id=role["role_id"],
                status="existing",
                generated_at=datetime.now(timezone.utc),
                draft=existing,
            )

        if self._llm is None:
            raise CvGenerationError(
                "cv_generation_unavailable",
                "The CV service is temporarily unavailable. Please try again later.",
            )

        try:
            result = await self._llm.generate(
                system_instruction=_DRAFT_SYSTEM_PROMPT,
                contents=_prompt_for_draft(context),
                tools=[_DRAFT_TOOL],
            )
            args = _one_tool_args(result, "submit_cv_draft")
            draft = _normalise_draft(
                args,
                role=role,
                experiences=experiences,
                allowed_skills=skills,
                personal=_personal_fields(plan, role["role_id"]),
            )
        except CvGenerationError as exc:
            logger.warning(
                "cv generation rejected: user=%s role_id=%s code=%s reason=%s",
                username,
                role["role_id"],
                exc.code,
                exc.message,
            )
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "cv generation unavailable: user=%s role_id=%s error_type=%s",
                username,
                role["role_id"],
                type(exc).__name__,
            )
            raise CvGenerationError(
                "cv_generation_unavailable",
                "The CV service is temporarily unavailable. Please try again later.",
            ) from exc

        updated_plan = copy.deepcopy(plan)
        book = _draft_book(updated_plan)
        book.setdefault("version", 3)
        book.setdefault("personal", copy.deepcopy(draft["personal"]))
        book.setdefault("drafts", {})
        book["activeRoleId"] = role["role_id"]
        book["drafts"][role["role_id"]] = copy.deepcopy(draft)
        updated_plan["cvDraft"] = book

        try:
            await self._repo.upsert_plan(session, username, updated_plan)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "cv generation persistence unavailable: user=%s role_id=%s error_type=%s",
                username,
                role["role_id"],
                type(exc).__name__,
            )
            raise CvGenerationError(
                "cv_generation_unavailable",
                "The CV service is temporarily unavailable. Please try again later.",
            ) from exc

        logger.info(
            "cv generated: user=%s role_id=%s experiences=%d skills=%d",
            username,
            role["role_id"],
            len(draft["experiences"]),
            len(draft["skills"]),
        )
        return CvGenerateResult(
            role_id=role["role_id"],
            status="generated",
            generated_at=datetime.now(timezone.utc),
            draft=draft,
        )

    async def improve(
        self,
        session: Any,
        username: str,
        role_id: str,
        section: str,
        experience_index: int | None = None,
        current_text: str = "",
        previous_suggestions: list[str] | None = None,
    ) -> CvImproveResult:
        plan = await self._repo.get_plan(session, username)
        role, experiences, _skills, context = _validate_setup(plan, role_id)

        if section not in {"summary", "experience"}:
            raise CvGenerationError("invalid_cv_setup", "The CV section is invalid.")

        if section == "experience":
            if (
                experience_index is None
                or experience_index < 0
                or experience_index >= len(experiences)
            ):
                raise CvGenerationError("invalid_cv_setup", "The CV experience is invalid.")
            evidence_sources = [experiences[experience_index]]
        else:
            evidence_sources = experiences

        if self._llm is None:
            raise CvGenerationError(
                "cv_generation_unavailable",
                "The CV service is temporarily unavailable. Please try again later.",
            )

        try:
            result = await self._llm.generate(
                system_instruction=_IMPROVE_SYSTEM_PROMPT,
                contents=_prompt_for_improvement(
                    context,
                    section=section,
                    experience_index=experience_index,
                    current_text=current_text,
                    previous_suggestions=previous_suggestions or [],
                ),
                tools=[_IMPROVE_TOOL],
            )
            args = _one_tool_args(result, "submit_cv_improvement")
            suggestion = _validate_bullet(args.get("suggestion"))
            evidence = _safe_text(args.get("evidence"))
            if not evidence or not any(evidence in _source_text(source) for source in evidence_sources):
                raise CvGenerationError("invalid_cv_content", "improve evidence not an exact source excerpt")
        except CvGenerationError as exc:
            logger.warning(
                "cv refinement rejected: user=%s role_id=%s section=%s code=%s reason=%s",
                username,
                role["role_id"],
                section,
                exc.code,
                exc.message,
            )
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "cv refinement unavailable: user=%s role_id=%s section=%s error_type=%s",
                username,
                role["role_id"],
                section,
                type(exc).__name__,
            )
            raise CvGenerationError(
                "cv_generation_unavailable",
                "The CV service is temporarily unavailable. Please try again later.",
            ) from exc

        logger.info(
            "cv refinement generated: user=%s role_id=%s section=%s",
            username,
            role["role_id"],
            section,
        )
        return CvImproveResult(
            section=section,
            experience_index=experience_index,
            suggestion=suggestion,
            evidence=evidence,
        )
