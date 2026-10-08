from copy import deepcopy

import pytest

from app.services import cv_generation as cv_generation_mod
from app.services.cv_generation import (
    CvGenerationError,
    CvGenerationService,
)
from app.services.llm import GenerateResult, LlmError

pytestmark = pytest.mark.asyncio


def _plan():
    return {
        "cvParsed": True,
        "cv": {
            "raw_text": "PRIVATE RAW CV text aisha@example.com +60123456789",
            "experiences": [
                {
                    "title": "Operations Coordinator",
                    "organisation": "Acme Sdn Bhd",
                    "start": "2018-01",
                    "end": "2021-06",
                    "description": "Coordinated delivery across teams and maintained project schedules.",
                },
                {
                    "title": "Office Assistant",
                    "organisation": "Beta Services",
                    "start": "2016",
                    "end": "2017",
                    "description": "",
                },
            ],
            "skill_mentions": ["coordination"],
        },
        "break": {
            "duration_years": 2,
            "activities": [
                "care_household.cared_for_children",
                "finance.managed_budget_finances",
            ],
        },
        "snapshot": {
            "professional_skills": [
                {"skill": "Project coordination"},
                {"skill": "Scheduling"},
            ],
            "reframed_skills": [{"skill": "Planning"}],
            "recommended_roles": [
                {"role": "Project Manager", "role_id": "role-project"},
                {"role": "Operations Manager", "role_id": "role-operations"},
            ],
        },
        "selectedRole": {"role": "Project Manager", "role_id": "role-project"},
        "cvDraft": {
            "version": 3,
            "activeRoleId": None,
            "personal": {
                "name": "",
                "email": "",
                "phone": "",
                "location": "",
            },
            "drafts": {},
            "gaps": {},
        },
        "unrelated": {"must": "survive"},
    }


def _function_result(*, summary="Operations professional ready for project management.", skills=None):
    return GenerateResult(
        content={
            "parts": [
                {
                    "functionCall": {
                        "name": "submit_cv_draft",
                        "args": {
                            "summary": summary,
                            "skills": skills or ["Project coordination", "Planning"],
                            "experiences": [
                                {
                                    "source_index": 0,
                                    "bullets": [
                                        {
                                            "text": "Coordinated delivery across teams and maintained project schedules.",
                                            "evidence": "Coordinated delivery across teams and maintained project schedules.",
                                        }
                                    ],
                                }
                            ],
                        },
                    }
                }
            ]
        },
        tokens_in=10,
        tokens_out=20,
    )


def _improve_result():
    return GenerateResult(
        content={
            "parts": [
                {
                    "functionCall": {
                        "name": "submit_cv_improvement",
                        "args": {
                            "suggestion": "Coordinated cross-team delivery and maintained project schedules.",
                            "evidence": "Coordinated delivery across teams and maintained project schedules.",
                        },
                    }
                }
            ]
        },
        tokens_in=10,
        tokens_out=20,
    )


class FakeRepo:
    def __init__(self, plan):
        self.plans = {"aisha": deepcopy(plan)}
        self.writes = []

    async def get_plan(self, session, username):
        return deepcopy(self.plans.get(username))

    async def upsert_plan(self, session, username, plan):
        self.writes.append((username, deepcopy(plan)))
        self.plans[username] = deepcopy(plan)


class FakeLlm:
    model = "fake-gemini"

    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.result


@pytest.fixture
def repo(monkeypatch):
    fake = FakeRepo(_plan())
    monkeypatch.setattr(cv_generation_mod, "accounts_repo", fake)
    return fake


async def test_generate_sends_only_allowlisted_context_and_persists_normalized_draft(repo):
    llm = FakeLlm(result=_function_result())
    service = CvGenerationService(llm)

    result = await service.generate(object(), "aisha")

    assert result.status == "generated"
    assert result.role_id == "role-project"
    assert result.draft["personal"] == {"name": "", "email": "", "phone": "", "location": ""}
    assert result.draft["skills"] == ["Project coordination", "Planning"]
    assert result.draft["experiences"][0]["title"] == "Operations Coordinator"
    assert result.draft["experiences"][0]["organisation"] == "Acme Sdn Bhd"
    assert result.draft["experiences"][0]["description"].startswith("- ")
    assert result.draft["experiences"][1]["description"] == ""
    assert result.draft["careerBreak"] == {
        "duration": "About 2 years",
        "description": (
            "- Took a career break to care for children.\n- Managed the family budget."
        ),
    }

    prompt = llm.calls[0]["contents"][0]["parts"][0]["text"]
    assert "PRIVATE RAW CV" not in prompt
    assert "aisha@example.com" not in prompt
    assert "+60123456789" not in prompt
    assert "Operations Coordinator" in prompt
    assert "Project coordination" in prompt
    assert "Planning" in prompt
    assert repo.plans["aisha"]["cvDraft"]["drafts"]["role-project"] == result.draft
    assert repo.plans["aisha"]["unrelated"] == {"must": "survive"}


async def test_regenerate_keeps_learned_skills(repo):
    # A learned skill (from a finished focus area) is client-owned, not in the snapshot.
    # It must be in the allowlist so a regenerate does not drop it.
    repo.plans["aisha"]["learnedSkills"] = [{"skill": "Data Visualisation"}]
    llm = FakeLlm(result=_function_result(skills=["Project coordination", "Data Visualisation"]))
    service = CvGenerationService(llm)

    result = await service.generate(object(), "aisha", regenerate=True)

    assert "Data Visualisation" in result.draft["skills"]
    assert "Project coordination" in result.draft["skills"]


async def test_existing_role_draft_is_returned_without_calling_gemini(repo):
    existing = {
        "version": 3,
        "roleId": "role-project",
        "personal": {"name": "Aisha", "email": "", "phone": "", "location": ""},
        "summary": "My saved summary.",
        "skills": ["Scheduling"],
        "experiences": [],
        "careerBreak": None,
    }
    repo.plans["aisha"]["cvDraft"]["drafts"]["role-project"] = existing
    llm = FakeLlm(result=_function_result(summary="must not be used"))

    result = await CvGenerationService(llm).generate(object(), "aisha")

    assert result.status == "existing"
    assert result.draft == existing
    assert llm.calls == []


async def test_regeneration_preserves_personal_fields_and_replaces_only_after_success(repo):
    old = {
        "version": 3,
        "roleId": "role-project",
        "personal": {"name": "Aisha", "email": "aisha@mail.test", "phone": "", "location": ""},
        "summary": "Old generated wording.",
        "skills": ["Scheduling"],
        "experiences": [],
        "careerBreak": None,
    }
    repo.plans["aisha"]["cvDraft"]["drafts"]["role-project"] = old
    llm = FakeLlm(result=_function_result(summary="Fresh generated wording."))

    result = await CvGenerationService(llm).generate(object(), "aisha", regenerate=True)

    assert result.draft["summary"] == "Fresh generated wording."
    assert result.draft["personal"] == old["personal"]
    assert repo.plans["aisha"]["cvDraft"]["drafts"]["role-project"]["summary"] == "Fresh generated wording."


async def test_generation_failure_preserves_existing_draft(repo):
    old = {
        "version": 3,
        "roleId": "role-project",
        "personal": {"name": "Aisha", "email": "", "phone": "", "location": ""},
        "summary": "Keep this.",
        "skills": [],
        "experiences": [],
        "careerBreak": None,
    }
    repo.plans["aisha"]["cvDraft"]["drafts"]["role-project"] = old
    llm = FakeLlm(error=LlmError("quota"))

    with pytest.raises(CvGenerationError) as exc_info:
        await CvGenerationService(llm).generate(object(), "aisha", regenerate=True)

    assert exc_info.value.code == "cv_generation_unavailable"
    assert repo.plans["aisha"]["cvDraft"]["drafts"]["role-project"] == old
    assert repo.writes == []


async def test_unmatched_role_is_rejected_before_calling_gemini(repo):
    llm = FakeLlm(result=_function_result())

    with pytest.raises(CvGenerationError) as exc_info:
        await CvGenerationService(llm).generate(object(), "aisha", role_id="role-other")

    assert exc_info.value.code == "invalid_cv_setup"
    assert llm.calls == []


async def test_grounding_rejects_unknown_skill_and_bad_evidence(repo):
    invalid = _function_result(skills=["Invented leadership"])
    llm = FakeLlm(result=invalid)

    with pytest.raises(CvGenerationError) as exc_info:
        await CvGenerationService(llm).generate(object(), "aisha", regenerate=True)

    assert exc_info.value.code == "invalid_cv_content"
    assert repo.writes == []


async def test_improve_returns_suggestion_without_persisting(repo):
    llm = FakeLlm(result=_improve_result())
    service = CvGenerationService(llm)

    result = await service.improve(
        object(),
        "aisha",
        role_id="role-project",
        section="experience",
        experience_index=0,
        current_text="- Coordinated delivery across teams and maintained project schedules.",
    )

    assert result.section == "experience"
    assert result.suggestion == "- Coordinated cross-team delivery and maintained project schedules."
    assert repo.writes == []


def _improve_summary_result():
    return GenerateResult(
        content={
            "parts": [
                {
                    "functionCall": {
                        "name": "submit_cv_improvement",
                        "args": {
                            "suggestion": (
                                "Operations coordinator with team delivery and "
                                "scheduling experience, aiming for a project management role."
                            ),
                            "evidence": "Coordinated delivery and scheduling across teams",
                        },
                    }
                }
            ]
        },
        tokens_in=10,
        tokens_out=20,
    )


async def test_improve_summary_accepts_synthesised_evidence(repo):
    # A summary improvement must not require evidence to be a verbatim experience excerpt.
    llm = FakeLlm(result=_improve_summary_result())
    service = CvGenerationService(llm)

    result = await service.improve(
        object(),
        "aisha",
        role_id="role-project",
        section="summary",
        current_text="Operations professional ready for project management.",
    )

    assert result.section == "summary"
    assert result.suggestion.startswith("Operations coordinator")
    assert repo.writes == []


def _improve_result_multi_bullet():
    return GenerateResult(
        content={
            "parts": [
                {
                    "functionCall": {
                        "name": "submit_cv_improvement",
                        "args": {
                            "suggestion": (
                                "- Coordinated cross-team delivery.\n"
                                "- Maintained project schedules across teams."
                            ),
                            "evidence": "Coordinated delivery across teams and maintained project schedules.",
                        },
                    }
                }
            ]
        },
        tokens_in=10,
        tokens_out=20,
    )


async def test_improve_experience_keeps_all_bullets(repo):
    # A multi-bullet experience must come back with every bullet, not collapsed to one.
    llm = FakeLlm(result=_improve_result_multi_bullet())
    service = CvGenerationService(llm)

    result = await service.improve(
        object(),
        "aisha",
        role_id="role-project",
        section="experience",
        experience_index=0,
        current_text=(
            "- Coordinated delivery across teams and maintained project schedules.\n"
            "- Maintained project schedules."
        ),
    )

    lines = result.suggestion.split("\n")
    assert len(lines) == 2
    assert all(line.startswith("- ") for line in lines)


async def test_missing_substantive_source_is_prerequisite_error(repo):
    plan = _plan()
    plan["cv"]["experiences"] = []
    plan["snapshot"]["professional_skills"] = []
    plan["snapshot"]["reframed_skills"] = []
    repo.plans["aisha"] = plan

    with pytest.raises(CvGenerationError) as exc_info:
        await CvGenerationService(FakeLlm(result=_function_result())).generate(object(), "aisha")

    assert exc_info.value.code == "journey_prerequisite_incomplete"
