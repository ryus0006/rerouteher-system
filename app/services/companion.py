"""E8 AI Companion agent (US8.1: profile building).

One Gemini agent, one system prompt, one history, tool-calling. Guest-usable.
The update_profile tool lets the model commit a structured {cv, break} profile,
which is returned as journey_update for the frontend to apply to its store.
Snapshot generation is a later story; this only builds the profile.
"""
import logging

from app.repositories import companion as companion_repo
from app.schemas.companion import AskRequest, AskResponse, JourneyUpdate
from app.services.llm import LlmError

logger = logging.getLogger("rerouteher")

SYSTEM_PROMPT = (
    "You are ReRouteHer's re-entry companion for Malaysian mothers returning to work. "
    "Be warm, concrete and brief. Your job in this step is to help her build a career "
    "profile by chatting: her most recent occupation, her skills, and her career break "
    "(how long, and what she did - caregiving counts as real experience, never a blank "
    "gap). Ask one clear follow-up question when something important is missing or "
    "unclear; do not guess. When you have enough to form or update her profile, call the "
    "update_profile tool with a cv object (raw_text, experiences, skill_mentions) and a "
    "break object (duration_years, activities), then tell her you have drafted her profile "
    "and invite her to review it below and confirm it or ask for a change. "
    "If her provided CV's most recent experience looks years out of date, note "
    "it and invite her to add anything recent (courses, volunteering, freelance, "
    "caregiving). Stay within career re-entry support: if asked for medical, legal, "
    "financial or other out-of-scope advice, say that is outside what you can help with "
    "and steer back. Only use what she has told you or what is in her journey; never "
    "invent facts."
)

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
                },
            },
        }
    ]
}

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
        note = self._journey_note(req)
        if note:
            system = f"{system}\n\nContext: {note}"

        journey_update: JourneyUpdate | None = None
        sources: list[str] = []
        tokens_in = 0
        tokens_out = 0

        # A transient LLM failure (timeout, 5xx, parse) degrades to a calm message
        # rather than a 500 - which would also lose CORS headers and read as a
        # "failed to fetch" in the browser.
        try:
            result = await self._llm.generate(
                system_instruction=system, contents=contents, tools=[UPDATE_PROFILE_TOOL]
            )
            tokens_in += result.tokens_in
            tokens_out += result.tokens_out
            content = result.content
            call = _first_function_call(content)
            if call and call.get("name") == "update_profile":
                args = call.get("args") or {}
                journey_update = JourneyUpdate.model_validate(args)
                if req.journey.cv is not None:
                    sources.append("Your CV")
                # feed the tool result back so the model gives a natural confirmation
                contents.append(content)
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": "update_profile",
                                    "response": {"status": "saved"},
                                }
                            }
                        ],
                    }
                )
                result = await self._llm.generate(
                    system_instruction=system, contents=contents, tools=[UPDATE_PROFILE_TOOL]
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
        return AskResponse(answer=answer, sources=sources, journey_update=journey_update)
