"""E8 AI Companion agent (US8.1: profile building).

One Gemini agent, one system prompt, one history, tool-calling. Guest-usable.
The update_profile tool lets the model commit a structured {cv, break} profile,
which is returned as journey_update for the frontend to apply to its store.
Snapshot generation is a later story; this only builds the profile.
"""
import logging

from app.repositories import companion as companion_repo
from app.repositories import roles as roles_repo
from app.schemas.companion import AskRequest, AskResponse, CtaOut, JourneyUpdate, SkillChoice
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
    "as employerPriorities. Map what she did during her break to the closest of our fixed "
    "activity ids (childcare, running the household, caring for elderly or sick family, "
    "day-to-day coordination, managing schedules, event planning, paperwork and records, "
    "budgeting, home repairs and contractors, negotiation, teaching or tutoring, "
    "volunteering) and pass them as break.activities, so her break is recognised as real "
    "experience, never a blank gap. When you have enough, call the update_profile tool with "
    "a cv object (raw_text, experiences, skill_mentions), a break object (duration_years, "
    "activities) and employerPriorities if she gave any, then tell her you have drafted "
    "her profile and invite her to review and confirm it or ask for a change. Once you know "
    "her most recent occupation, call offer_role_skills with it so she can tick the skills "
    "she already has from that role. Only describe actions you have actually taken this turn "
    "- do not say you have shown her a checklist or saved anything unless you called the "
    "matching tool.\n\n"
    "If her results are provided below, answer her questions about her skills, readiness "
    "and priority gaps grounded only in those results, in plain language that relates them "
    "to her experience and target role. When she asks about a specific gap, explain "
    "briefly why it is a gap and offer her learning plan by calling the point_to_step tool "
    "with step 'learning'. If she asks something her results do not cover, say you cannot "
    "answer that from her results rather than inventing a figure.\n\n"
    "If her employer matches are provided below, answer why an employer matches her by "
    "relating what it discloses to her stated priorities, and cite the source report by name "
    "with its link so she can open it. If she asks about a detail the data does not cover, or "
    "a priority the employer does not disclose, say it is not disclosed rather than guessing.\n\n"
    "If she asks for help with where she is or what to do next, use the page she is on and "
    "what her journey already has to explain the current step briefly and point her to the "
    "next step by calling point_to_step with the matching step.\n\n"
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

# Fixed career-break activity taxonomy (source of truth = frontend activityTaxonomy.js,
# == the 12 distinct caregiving_map.activity_id values). The companion maps her free-text
# break to these ids so the Career Break page highlights them and caregiving_map reframes them.
_VALID_ACTIVITY_IDS = [
    "care_household.cared_for_children",
    "care_household.ran_household",
    "care_household.cared_for_elderly_sick_family",
    "planning.organised_family_logistics",
    "planning.managed_multiple_schedules",
    "planning.planned_events_gatherings",
    "planning.kept_household_records",
    "finance.managed_budget_finances",
    "finance.managed_home_repairs_vendors",
    "finance.handled_disputes_negotiations",
    "learning.taught_tutored_children",
    "learning.volunteered_community_roles",
]

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
                            "activities": {
                                "type": "array",
                                "description": (
                                    "The things she did during her break, each mapped to the "
                                    "closest of our fixed activity ids. Only use ids from the list."
                                ),
                                "items": {"type": "string", "enum": _VALID_ACTIVITY_IDS},
                            },
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
                "Offer her an optional button to a step in her journey that helps her next "
                "action - her skill snapshot, readiness and gaps, learning plan, or employer "
                "matches. Use sparingly, only when it genuinely helps."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "step": {
                        "type": "string",
                        "enum": ["learning", "snapshot", "gap", "employers"],
                    }
                },
                "required": ["step"],
            },
        }
    ]
}

# Maps an allowed step to the CTA the frontend renders (button -> client-side nav).
_STEP_CTAS = {
    "learning": {"label": "Open your learning plan", "to": "/plan/learning"},
    "snapshot": {"label": "See my skill snapshot", "to": "/diagnostic/snapshot"},
    "gap": {"label": "See my readiness & gaps", "to": "/diagnostic/gap"},
    "employers": {"label": "See my employer matches", "to": "/plan/employers/matches"},
}

OFFER_ROLE_SKILLS_TOOL = {
    "function_declarations": [
        {
            "name": "offer_role_skills",
            "description": (
                "Offer her a checklist of skills common to her most recent occupation so she "
                "can tick the ones she already has. Call it once you know her occupation."
            ),
            "parameters": {
                "type": "object",
                "properties": {"occupation": {"type": "string"}},
                "required": ["occupation"],
            },
        }
    ]
}
_MAX_ROLE_SKILL_CHOICES = 15

_TOOLS = [UPDATE_PROFILE_TOOL, POINT_TO_STEP_TOOL, OFFER_ROLE_SKILLS_TOOL]

# Cap on tool-calling rounds per turn, so a model that keeps calling tools cannot loop
# unbounded; on the cap we force one final answer with no tools.
_MAX_TOOL_ITERS = 4

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


def _function_calls(content: dict) -> list[dict]:
    return [p["functionCall"] for p in content.get("parts", []) if "functionCall" in p]


class CompanionService:
    def __init__(self, llm=None, repo=companion_repo, role_resolver=None) -> None:
        self._llm = llm
        self._repo = repo
        # anything exposing async resolve_role(title, skill_names, session) -> role|None;
        # the snapshot service provides it so the checklist role matches the derived one.
        self._role_resolver = role_resolver

    async def _role_skill_choices(self, occupation, req, session):
        """Resolve her occupation to a role and return (role_id, distinctive skill choices),
        excluding ones she already confirmed. None when nothing resolves, no new choices, or
        the role was already offered this session (gate on the resolved role_id so a genuine
        role change still re-offers)."""
        if not occupation or self._role_resolver is None:
            return None
        role = await self._role_resolver.resolve_role(occupation, [], session)
        if role is None:
            return None
        if role.role_id == req.journey.roleSkillsOfferedForRoleId:
            return None  # already offered for this role; do not re-offer
        confirmed = [c.skill_id for c in req.journey.confirmedSkills]
        skills = await roles_repo.get_distinctive_role_skills(
            session, role.role_id, _MAX_ROLE_SKILL_CHOICES, confirmed
        )
        choices = [SkillChoice(skill_id=s.skill_id, skill_name=s.skill_name) for s in skills]
        if not choices:
            return None
        return role.role_id, choices

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

    @staticmethod
    def _pretty_priority(token: str) -> str:
        return token.replace("_", " ").strip()

    def _employer_note(self, req: AskRequest) -> str:
        # Grounds employer Q&A (US8.3) in her own matches: employer names, the priorities
        # each meets vs does not disclose, and the report behind the claim. No PII.
        matches = req.journey.employerMatches or []
        bits = []
        for m in matches[:5]:
            name = m.get("name")
            if not name:
                continue
            met = [self._pretty_priority(p) for p in (m.get("met") or [])]
            unmet = [self._pretty_priority(p) for p in (m.get("unmet") or [])]
            parts = [name]
            if met:
                parts.append(f"meets {', '.join(met)}")
            if unmet:
                parts.append(f"does not disclose {', '.join(unmet)}")
            report = m.get("report") or {}
            if report.get("label") and report.get("url"):
                parts.append(f"source: {report['label']} ({report['url']})")
            bits.append("; ".join(parts) + ".")
        return ("Her employer matches: " + " ".join(bits)) if bits else ""

    async def ask(self, req: AskRequest, session, username: str | None) -> AskResponse:
        if self._llm is None:
            return AskResponse(answer=_NOT_AVAILABLE, sources=[], journey_update=None)

        history = await self._repo.load_recent(session, req.session_id, username, 10)
        contents: list[dict] = [
            {"role": "model" if t.role == "assistant" else "user", "parts": [{"text": t.content}]}
            for t in history
        ]
        contents.append({"role": "user", "parts": [{"text": req.question}]})

        system = SYSTEM_PROMPT
        extras = " ".join(
            p
            for p in (self._journey_note(req), self._results_note(req), self._employer_note(req))
            if p
        )
        if extras:
            system = f"{system}\n\nContext: {extras}"

        journey_update: JourneyUpdate | None = None
        cta: CtaOut | None = None
        skill_choices = None
        skill_choices_role_id = None
        sources: list[str] = []
        tokens_in = 0
        tokens_out = 0

        # Bounded multi-tool agentic loop: execute EVERY tool the model calls, feed all
        # results back, and let it chain for a few rounds until it returns plain text. This
        # is what lets one turn both draft a profile and open a checklist, and stops the
        # model narrating a tool it did not actually call.
        # A transient LLM failure degrades to a calm message rather than a 500 - which would
        # also lose CORS headers and read as a "failed to fetch" in the browser.
        try:
            content = None
            for _ in range(_MAX_TOOL_ITERS):
                result = await self._llm.generate(
                    system_instruction=system, contents=contents, tools=_TOOLS
                )
                tokens_in += result.tokens_in
                tokens_out += result.tokens_out
                content = result.content
                calls = _function_calls(content)
                if not calls:
                    break
                contents.append(content)
                response_parts = []
                for call in calls:
                    name = call.get("name")
                    args = call.get("args") or {}
                    status = {"status": "ok"}
                    if name == "update_profile":
                        journey_update = JourneyUpdate.model_validate(args)
                        # Defend the fixed pick-list even though the tool enum constrains it:
                        # drop anything unknown, keep order, cap at three.
                        journey_update.employerPriorities = [
                            p for p in journey_update.employerPriorities if p in _VALID_PRIORITY_IDS
                        ][:_MAX_PRIORITIES]
                        # Defend the fixed activity taxonomy: keep only ids the CareerBreak page
                        # and caregiving_map recognise (so both can render/reframe them).
                        if journey_update.break_ is not None:
                            journey_update.break_.activities = [
                                a
                                for a in journey_update.break_.activities
                                if a in _VALID_ACTIVITY_IDS
                            ]
                        if req.journey.cv is not None:
                            sources.append("Your CV")
                        status = {"status": "saved"}
                    elif name == "point_to_step":
                        mapped = _STEP_CTAS.get(args.get("step"))
                        if mapped:
                            cta = CtaOut(**mapped)
                    elif name == "offer_role_skills":
                        offered = await self._role_skill_choices(
                            args.get("occupation") or "", req, session
                        )
                        if offered:
                            skill_choices_role_id, skill_choices = offered
                            status = {"status": "shown", "count": len(skill_choices)}
                        else:
                            status = {"status": "already_shown"}
                    response_parts.append(
                        {"functionResponse": {"name": name, "response": status}}
                    )
                contents.append({"role": "user", "parts": response_parts})
            else:
                # cap reached while still tool-calling: force one clean text answer
                result = await self._llm.generate(
                    system_instruction=system, contents=contents, tools=None
                )
                tokens_in += result.tokens_in
                tokens_out += result.tokens_out
                content = result.content
        except LlmError as exc:
            logger.warning("companion LLM error: %s", exc)
            return AskResponse(answer=_TROUBLE, sources=[], journey_update=None)

        answer = _first_text(content or {}) or "Got it."

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
        return AskResponse(
            answer=answer,
            sources=sources,
            journey_update=journey_update,
            cta=cta,
            skill_choices=skill_choices,
            skill_choices_role_id=skill_choices_role_id,
        )
