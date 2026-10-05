"""Taxonomy-backed professional-skill mutations shared by account and Hera flows."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import anyio

from app.repositories import accounts as accounts_repo
from app.repositories import roles as roles_repo
from app.repositories import skills as skills_repo
from app.schemas.gap import GapRequest


class ProfileSkillError(Exception):
    """User-safe profile-skill error with a router-facing category."""

    def __init__(self, message: str, kind: str) -> None:
        super().__init__(message)
        self.message = message
        self.kind = kind


@dataclass
class ProfileSkillMutation:
    status: str
    skill_id: str
    skill: str | None
    definition: str | None
    plan: dict[str, Any]
    snapshot: dict[str, Any]
    gap_result: dict[str, Any] | None
    learned_skills: list[dict[str, Any]]


class ProfileSkillService:
    """Read taxonomy skills and mutate only the signed-in user's saved plan."""

    def __init__(self, *, embedder=None, gap_service=None, embedding_threshold: float = 0.35):
        self._embedder = embedder
        self._gap_service = gap_service
        self._embedding_threshold = embedding_threshold

    async def match_skills(self, session, query: str, limit: int = 3):
        text = (query or "").strip()
        if not text:
            return []

        exact = await skills_repo.find_exact_skills(session, text, limit)
        if exact:
            return exact[:limit]
        if self._embedder is None:
            return []

        vector = await anyio.to_thread.run_sync(self._embedder.encode_one, text)
        return await skills_repo.match_by_embedding(
            session,
            vector,
            k=limit,
            threshold=self._embedding_threshold,
        )

    async def add_skill(self, session, username: str, skill_id: str) -> ProfileSkillMutation:
        plan = await self._load_plan(session, username)
        skill = await skills_repo.get_skill_by_id(session, skill_id)
        if skill is None:
            raise ProfileSkillError("That skill was not found.", "not_found")

        snapshot = copy.deepcopy(plan["snapshot"])
        professional = list(snapshot.get("professional_skills") or [])
        if any(self._skill_id(item) == skill.skill_id for item in professional):
            return self._result("already_present", skill, plan, snapshot)

        entry = {"skill_id": skill.skill_id, "skill": skill.canonical_name}
        if skill.definition:
            entry["definition"] = skill.definition
        professional.append(entry)
        snapshot["professional_skills"] = professional
        plan["snapshot"] = snapshot

        gap_result = copy.deepcopy(plan.get("gapResult"))
        gap = self._find_gap(gap_result, skill.skill_id)
        if gap is not None:
            gap_result["readiness"] = min(
                100,
                round(float(gap_result.get("readiness") or 0) + float(gap.get("uplift") or 0), 1),
            )
            have = list(gap_result.get("skills_have") or [])
            if skill.canonical_name not in have:
                have.append(skill.canonical_name)
            gap_result["skills_have"] = have
            gap_result["gaps"] = [
                item for item in gap_result.get("gaps") or [] if self._skill_id(item) != skill.skill_id
            ]
            plan["gapResult"] = gap_result

        await self._persist(session, username, plan)
        return self._result("added", skill, plan, snapshot)

    async def remove_skill(self, session, username: str, skill_id: str) -> ProfileSkillMutation:
        plan = await self._load_plan(session, username)
        skill = await skills_repo.get_skill_by_id(session, skill_id)
        if skill is None:
            raise ProfileSkillError("That skill was not found.", "not_found")

        snapshot = copy.deepcopy(plan["snapshot"])
        professional = list(snapshot.get("professional_skills") or [])
        remaining = [item for item in professional if self._skill_id(item) != skill.skill_id]
        if len(remaining) == len(professional):
            return self._result("not_present", skill, plan, snapshot)

        snapshot["professional_skills"] = remaining
        plan["snapshot"] = snapshot

        selected_role = plan.get("selectedRole") or {}
        role_id = selected_role.get("role_id")
        gap_result = copy.deepcopy(plan.get("gapResult"))
        if role_id and self._gap_service is not None:
            role = await roles_repo.get_role_with_skills_by_id(session, role_id)
            required = role and any(item.skill_id == skill.skill_id for item in role.skills)
            if required:
                request = GapRequest(
                    skill_ids=self._skill_ids(snapshot),
                    target_role_id=role_id,
                    target_role=selected_role.get("role"),
                )
                computed = await self._gap_service.compute(request, session)
                gap_result = self._dump(computed)
                plan["gapResult"] = gap_result

        await self._persist(session, username, plan)
        return self._result("removed", skill, plan, snapshot)

    async def _persist(self, session, username: str, plan: dict[str, Any]) -> None:
        try:
            await accounts_repo.upsert_plan(session, username, plan)
        except Exception as exc:
            raise ProfileSkillError(
                "Profile skill is temporarily unavailable.",
                "internal",
            ) from exc

    async def _load_plan(self, session, username: str) -> dict[str, Any]:
        reader = getattr(accounts_repo, "get_plan_for_update", accounts_repo.get_plan)
        plan = await reader(session, username)
        if not isinstance(plan, dict) or not isinstance(plan.get("snapshot"), dict):
            raise ProfileSkillError(
                "Complete your skill snapshot before changing professional skills.",
                "journey",
            )
        return copy.deepcopy(plan)

    @staticmethod
    def _skill_id(value: Any) -> str | None:
        return value.get("skill_id") if isinstance(value, dict) else getattr(value, "skill_id", None)

    @classmethod
    def _find_gap(cls, gap_result: dict[str, Any] | None, skill_id: str) -> dict[str, Any] | None:
        if not isinstance(gap_result, dict):
            return None
        return next(
            (item for item in gap_result.get("gaps") or [] if cls._skill_id(item) == skill_id),
            None,
        )

    @staticmethod
    def _skill_ids(snapshot: dict[str, Any]) -> list[str]:
        values = list(snapshot.get("professional_skills") or []) + list(
            snapshot.get("reframed_skills") or []
        )
        return [item["skill_id"] for item in values if isinstance(item, dict) and item.get("skill_id")]

    @staticmethod
    def _dump(value):
        return value.model_dump() if hasattr(value, "model_dump") else copy.deepcopy(value)

    def _result(self, status, skill, plan, snapshot):
        return ProfileSkillMutation(
            status=status,
            skill_id=skill.skill_id,
            skill=skill.canonical_name,
            definition=skill.definition,
            plan=copy.deepcopy(plan),
            snapshot=copy.deepcopy(snapshot),
            gap_result=copy.deepcopy(plan.get("gapResult")),
            learned_skills=copy.deepcopy(plan.get("learnedSkills") or []),
        )
