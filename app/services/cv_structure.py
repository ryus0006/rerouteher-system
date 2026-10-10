"""LLM structuring for CV parsing.

The deterministic extractor produces clean, PII-masked text; this turns that text into
experiences and skills with one Gemini structured call, which is far more robust across
differing CV layouts than regex segmentation. Returns None to let the caller fall back to
the deterministic parse when the LLM is unavailable or returns nothing usable. Only the
already-masked text is sent - names and addresses are masked and email/phone removed before
this runs.
"""
from __future__ import annotations

import logging
from typing import Any

from app.schemas.cv import Experience
from app.services.llm import LlmError

logger = logging.getLogger("rerouteher")

_SYSTEM_PROMPT = (
    "You extract structured data from the plain text of one person's CV. The candidate's "
    "name and address have already been masked with asterisks and their email and phone "
    "removed - do not try to recover them.\n\n"
    "Call submit_cv exactly once with two fields:\n"
    "- experiences: real paid work roles only, most recent first. For each role give:\n"
    "  title: the job role as a short noun phrase (e.g. 'Senior Software Developer'). It is "
    "NEVER a full sentence or a bullet point - if a line reads like a responsibility "
    "('Developed front-end components...'), that is a description, not a title.\n"
    "  organisation: the employer's company name (e.g. 'Axiata Digital Labs'). It is NEVER a "
    "technology, tool, methodology, certification, or acronym - 'CI/CD', 'UI/UX', 'QA', "
    "'Agile', 'ISO 9001', 'HPLC', 'Angular' are not employers. If the employer is not clearly "
    "stated for a role, leave organisation empty rather than guessing.\n"
    "  start and end: exactly as written in the CV (e.g. 'Feb 2013', 'Nov 2019', 'Present').\n"
    "  description: the role's bullet points joined into a short paragraph.\n"
    "  Exclude education, references, projects, and career breaks - a line such as 'Took a "
    "career break to raise children' is NOT a work experience.\n"
    "- skills: concrete professional skills and tools named in the CV (e.g. 'SQL', 'digital "
    "marketing', 'stakeholder management').\n\n"
    "Use only what is written in the text. Never invent roles, employers, dates, or skills. "
    "Leave a field empty if it is not present."
)

_SUBMIT_CV_TOOL = {
    "function_declarations": [
        {
            "name": "submit_cv",
            "description": "Return the experiences and skills extracted from the CV text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "experiences": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "organisation": {"type": "string"},
                                "start": {"type": "string"},
                                "end": {"type": "string"},
                                "description": {"type": "string"},
                            },
                        },
                    },
                    "skills": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["experiences", "skills"],
            },
        }
    ]
}


def _submit_cv_args(content: Any) -> dict | None:
    if not isinstance(content, dict):
        return None
    for part in content.get("parts") or []:
        call = part.get("functionCall") or part.get("function_call") if isinstance(part, dict) else None
        if isinstance(call, dict) and call.get("name") == "submit_cv":
            args = call.get("args") or call.get("arguments")
            return args if isinstance(args, dict) else None
    return None


class CvStructureService:
    """Structures masked CV text into experiences and skills via the LLM."""

    def __init__(self, llm=None) -> None:
        self._llm = llm

    @property
    def available(self) -> bool:
        return self._llm is not None

    async def structure(self, raw_text: str) -> tuple[list[Experience], list[str]] | None:
        if self._llm is None or not (raw_text or "").strip():
            return None
        try:
            result = await self._llm.generate(
                system_instruction=_SYSTEM_PROMPT,
                contents=[{"role": "user", "parts": [{"text": raw_text}]}],
                tools=[_SUBMIT_CV_TOOL],
            )
        except LlmError as exc:
            logger.warning("cv structure LLM error: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("cv structure failed: %s", type(exc).__name__)
            return None

        args = _submit_cv_args(getattr(result, "content", None))
        if args is None:
            logger.warning("cv structure: no submit_cv call in response")
            return None
        experiences = self._experiences(args.get("experiences"))
        skills = self._skills(args.get("skills"))
        if not experiences and not skills:
            return None
        logger.info("cv structured by llm: experiences=%d skills=%d", len(experiences), len(skills))
        return experiences, skills

    @staticmethod
    def _clean(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        value = value.strip()
        return value or None

    @classmethod
    def _experiences(cls, raw: Any) -> list[Experience]:
        if not isinstance(raw, list):
            return []
        experiences: list[Experience] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            title = cls._clean(item.get("title"))
            organisation = cls._clean(item.get("organisation"))
            if not title and not organisation:
                continue  # need at least a role or employer to be a real entry
            experiences.append(
                Experience(
                    title=title,
                    organisation=organisation,
                    start=cls._clean(item.get("start")),
                    end=cls._clean(item.get("end")),
                    description=cls._clean(item.get("description")),
                )
            )
        return experiences

    @classmethod
    def _skills(cls, raw: Any) -> list[str]:
        if not isinstance(raw, list):
            return []
        seen: dict[str, str] = {}
        for item in raw:
            skill = cls._clean(item)
            if skill and skill.lower() not in seen:
                seen[skill.lower()] = skill
        return list(seen.values())
