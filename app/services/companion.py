"""E8 AI Companion agent (US8.1: profile building).

One Gemini agent, one system prompt, one history, tool-calling. Guest-usable.
The update_profile tool lets the model commit a structured {cv, break} profile,
which is returned as journey_update for the frontend to apply to its store.
Snapshot generation is a later story; this only builds the profile.
"""
import logging

from app.repositories import companion as companion_repo
from app.schemas.companion import AskRequest, AskResponse, CtaOut, JourneyUpdate
from app.services.llm import LlmError

logger = logging.getLogger("rerouteher")

SYSTEM_PROMPT = (
    "You are ReRouteHer's re-entry companion, Hera, for Malaysian mothers returning to "
    "work. Be warm, concrete and brief; write in short plain sentences and split a long "
    "answer into two short messages rather than one long block.\n\n"
    "If she has no results yet, help her build a career profile by chatting: her most "
    "recent occupation, her skills, and her career break (how long and what she did - "
    "caregiving counts as real experience, never a blank gap). Ask one clear follow-up "
    "when something important is missing; do not guess. While she is still building, if "
    "her CV's most recent experience looks years out of date, note it once and invite her "
    "to add anything recent such as courses, volunteering, freelance or caregiving; once "
    "her profile is complete do not raise it again. If she mentions what she "
    "most wants from an employer (flexible or remote work, childcare support, parental "
    "support, a return-to-work programme, or an inclusive workplace), capture up to three "
    "as employerPriorities. When you have enough, call the update_profile tool with a cv "
    "object (raw_text, experiences, skill_mentions), a break object (duration_years, "
    "activities) and employerPriorities if she gave any, then tell her you have drafted "
    "her profile and invite her to review and confirm it or ask for a change.\n\n"
    "If her results are provided below, answer her questions about her skills, readiness "
    "and priority gaps grounded only in those results, in plain language that relates them "
    "to her experience and target role. When she asks about a specific gap, explain "
    "briefly why it is a gap and offer her learning plan by calling the point_to_step tool "
    "with step 'learning'. If she asks something her results do not cover, say you cannot "
    "answer that from her results rather than inventing a figure.\n\n"
    "Stay within career re-entry support: decline medical, legal, financial or other "
    "out-of-scope advice and steer back. Use only what she has told you or what is in her "
    "journey; never invent facts."
)

# Fixed workplace-priority pick-list. Source of truth is the frontend config
# (src/config/employerPriorities.js); mirrored here for the tool enum and filter.
_VALID_PRIORITY_IDS = [
    "flexible_work",
    "childcare_support",
    "parental_support",
    "returning_to_work",
    "inclusive_workplace",
]
_MAX_PRIORITIES = 3

UPDATE_PROFILE_TOOL = {
    "function_declarations": [
        {
            "name": "update_profile",
            "description": "Commit the mother's structured career profile built from the conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cv": {
                        "type": "object",
                        "properties": {
                            "raw_text": {"type": "string"},
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
                            "skill_mentions": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["raw_text"],
                    },
                    "break": {
                        "type": "object",
                        "properties": {
                            "duration_years": {"type": "number"},
                            "activities": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                    "employerPriorities": {
                        "type": "array",
                        "description": (
                            "Up to three things she most wants from an employer, chosen from the "
                            "fixed list."
                        ),
                        "items": {"type": "string", "enum": _VALID_PRIORITY_IDS},
                    },
                },
            },
        }
    ]
}

POINT_TO_STEP_TOOL = {
    "function_declarations": [
        {
            "name": "point_to_step",
            "description": (
                "Offer the mother an optional link to a step in her journey that helps her "
                "next action, such as her learning plan for a skill gap. Use sparingly, only "
                "when it genuinely helps."
            ),
            "parameters": {
                "type": "object",
                "properties": {"step": {"type": "string", "enum": ["learning"]}},
                "required": ["step"],
            },
        }
    ]
}

# Maps an allowed step to the CTA the frontend renders (button -> client-side nav).
_STEP_CTAS = {"learning": {"label": "Open your learning plan", "to": "/plan/learning"}}

_TOOLS = [UPDATE_PROFILE_TOOL, POINT_TO_STEP_TOOL]

_NOT_AVAILABLE = "The companion is not available right now. Please try again later."
_TROUBLE = "I am having a little trouble right now. Please try again in a moment."


def _first_text(content: dict) -> str:
    for part in content.get("parts", []):
        if "text" in part:
            return part["text"]
    return ""


def _first_function_call(content: dict) -> dict | None:
    for part in content.get("parts", []):
        if "functionCall" in part:
            return part["functionCall"]
    return None


class CompanionService:
    def __init__(self, llm=None, repo=companion_repo) -> None:
        self._llm = llm
        self._repo = repo

    def _journey_note(self, req: AskRequest) -> str:
        j = req.journey
        bits = []
        if req.current_page:
            bits.append(
                f"She is currently on the '{req.current_page}' page; assume that is her focus "
                "unless she asks about something else."
            )
        if j.cv is not None:
            n = len(j.cv.experiences)
            bits.append(f"A parsed CV is already in her journey ({n} experience entries).")
        if j.break_ is not None and j.break_.duration_years:
            bits.append(f"Her recorded break is about {j.break_.duration_years} years.")
        # A CV and a break together mean she has already built a profile (mid-draft the
        # store still holds the old values). Do not nag a returning mother whose CV is
        # naturally out of date to keep updating a profile she has finished (AC 8.1.9).
        if j.cv is not None and j.break_ is not None:
            bits.append(
                "Her profile is already complete (a CV and a career break are saved). Do not push "
                "her to update it and do not repeat the out-of-date note; only change it if she "
                "explicitly asks. Otherwise help her move on to her skill snapshot."
            )
        return " ".join(bits)

    def _results_note(self, req: AskRequest) -> str:
        # Grounds Q&A (US8.2) in her own results by context-injection. Skills, roles
        # and readiness only - never any PII. Kept compact.
        j = req.journey
        bits = []
        role = (j.selectedRole or {}).get("role") if j.selectedRole else None
        gap = j.gapResult or {}
        if gap.get("readiness") is not None:
            bits.append(f"Her readiness is {gap['readiness']}%.")
        if role:
            bits.append(f"Her target role is {role}.")
        gaps = gap.get("gaps") or []
        top = ", ".join(
            f"{g.get('skill')} (+{g.get('uplift')}% if learned)"
            for g in gaps[:3]
            if g.get("skill")
        )
        if top:
            bits.append(f"Her priority skill gaps: {top}.")
        snap = j.snapshot or {}
        prev = (snap.get("previous_occupation") or {}).get("role")
        if prev:
            bits.append(f"Her most recent occupation was {prev}.")
        skills = [
            s.get("skill")
            for s in (snap.get("professional_skills") or []) + (snap.get("reframed_skills") or [])
            if s.get("skill")
        ]
        if skills:
            bits.append(f"Her snapshot skills include: {', '.join(skills[:12])}.")
        recs = [r.get("role") for r in (snap.get("recommended_roles") or []) if r.get("role")]
        if recs:
            bits.append(f"Roles recommended for her: {', '.join(recs[:5])}.")
        return " ".join(bits)

    async def ask(self, req: AskRequest, session, username: str | None) -> AskResponse:
        if self._llm is None:
            return AskResponse(answer=_NOT_AVAILABLE, sources=[], journey_update=None)

        history = await self._repo.load_recent(session, req.session_id, 10)
        contents: list[dict] = [
            {"role": "model" if t.role == "assistant" else "user", "parts": [{"text": t.content}]}
            for t in history
        ]
        contents.append({"role": "user", "parts": [{"text": req.question}]})

        system = SYSTEM_PROMPT
        extras = " ".join(p for p in (self._journey_note(req), self._results_note(req)) if p)
        if extras:
            system = f"{system}\n\nContext: {extras}"

        journey_update: JourneyUpdate | None = None
        cta: CtaOut | None = None
        sources: list[str] = []
        tokens_in = 0
        tokens_out = 0

        # A transient LLM failure (timeout, 5xx, parse) degrades to a calm message
        # rather than a 500 - which would also lose CORS headers and read as a
        # "failed to fetch" in the browser.
        try:
            result = await self._llm.generate(
                system_instruction=system, contents=contents, tools=_TOOLS
            )
            tokens_in += result.tokens_in
            tokens_out += result.tokens_out
            content = result.content
            call = _first_function_call(content)
            if call:
                name = call.get("name")
                args = call.get("args") or {}
                if name == "update_profile":
                    journey_update = JourneyUpdate.model_validate(args)
                    # Defend the fixed pick-list even though the tool enum constrains it:
                    # drop anything unknown, keep order, cap at three.
                    journey_update.employerPriorities = [
                        p for p in journey_update.employerPriorities if p in _VALID_PRIORITY_IDS
                    ][:_MAX_PRIORITIES]
                    if req.journey.cv is not None:
                        sources.append("Your CV")
                elif name == "point_to_step":
                    mapped = _STEP_CTAS.get(args.get("step"))
                    if mapped:
                        cta = CtaOut(**mapped)
                # feed the tool result back so the model gives a natural reply
                contents.append(content)
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {"functionResponse": {"name": name, "response": {"status": "ok"}}}
                        ],
                    }
                )
                result = await self._llm.generate(
                    system_instruction=system, contents=contents, tools=_TOOLS
                )
                tokens_in += result.tokens_in
                tokens_out += result.tokens_out
                content = result.content
        except LlmError as exc:
            logger.warning("companion LLM error: %s", exc)
            return AskResponse(answer=_TROUBLE, sources=[], journey_update=None)

        answer = _first_text(content) or "Got it."

        await self._repo.save_turn(session, req.session_id, username, "user", req.question)
        # Token usage for the whole turn (both model calls) is recorded on the answer.
        await self._repo.save_turn(
            session,
            req.session_id,
            username,
            "assistant",
            answer,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

        logger.info(
            "companion: session=%s page=%s tool=%s user=%s tokens_in=%d tokens_out=%d",
            req.session_id,
            req.current_page,
            journey_update is not None,
            username or "-",
            tokens_in,
            tokens_out,
        )
        return AskResponse(answer=answer, sources=sources, journey_update=journey_update, cta=cta)
