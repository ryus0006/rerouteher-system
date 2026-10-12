"""Keeps only job openings that are for the target role.

Job boards search loosely ("Systems Programmer" also returns "Programme
Coordinator"), so every opening at a tracked employer is checked before it is shown
as hiring for her role. One Gemini call per search judges all titles at once, so
synonyms ("Software Engineer") pass while word-only overlaps do not. When Gemini is
unavailable or returns unusable output, a stricter whole-word title check decides.
Only the role title and job titles are sent; no user data.
"""
from __future__ import annotations

import json
import logging
import re

from app.services.job_sources import JobCandidate

logger = logging.getLogger("rerouteher")

_SYSTEM_INSTRUCTION = (
    "You check whether job postings are for a target occupation. You are given the "
    "target role and a numbered list of job titles. Return a JSON array of the numbers "
    "whose job is the same occupation or a close equivalent doing the same core work, "
    "at any seniority. Exclude jobs that only share a word with the role (for example "
    "'Programme Coordinator' is not 'Programmer'). Return only the JSON array."
)


def _word_stems(value: str) -> set[str]:
    # Whole words, with a trailing plural "s" dropped so "System" matches "Systems".
    words = re.sub(r"[^a-z0-9]+", " ", (value or "").casefold()).split()
    return {word[:-1] if len(word) > 3 and word.endswith("s") else word for word in words}


def title_matches(role_title: str, job_title: str) -> bool:
    """Every word of the role title appears as a whole word in the job title."""
    role_words = _word_stems(role_title)
    return bool(role_words) and role_words <= _word_stems(job_title)


class JobRelevanceFilter:
    def __init__(self, llm) -> None:
        self._llm = llm

    async def keep(self, role_title: str, candidates: list[JobCandidate]) -> list[JobCandidate]:
        if not candidates:
            return []
        kept = await self._judge(role_title, candidates)
        method = "gemini"
        if kept is None:
            kept = [c for c in candidates if title_matches(role_title, c.title)]
            method = "title_words"
        dropped = [c.title for c in candidates if c not in kept]
        logger.info(
            "job relevance | role=%r | method=%s | kept=%d | dropped=%s",
            role_title, method, len(kept), dropped,
        )
        return kept

    async def _judge(
        self, role_title: str, candidates: list[JobCandidate]
    ) -> list[JobCandidate] | None:
        # None means "no usable verdict", so the caller falls back to the title check.
        if self._llm is None:
            return None
        prompt = json.dumps(
            {
                "target_role": role_title,
                "jobs": [{"n": i, "title": c.title} for i, c in enumerate(candidates)],
            }
        )
        try:
            generated = await self._llm.generate(
                system_instruction=_SYSTEM_INSTRUCTION,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                tools=None,
            )
            text = "".join(
                part.get("text", "")
                for part in generated.content.get("parts", [])
                if isinstance(part, dict)
            )
            match = re.search(r"\[[\s\S]*?\]", text)
            picked = json.loads(match.group(0)) if match else None
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "job relevance gemini failed | error=%s", type(exc).__name__, exc_info=True
            )
            return None
        if not isinstance(picked, list) or not all(isinstance(n, int) for n in picked):
            logger.warning("job relevance gemini returned unusable output")
            return None
        keep = {n for n in picked if 0 <= n < len(candidates)}
        return [c for i, c in enumerate(candidates) if i in keep]
