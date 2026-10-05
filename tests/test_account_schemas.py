from app.schemas.account import (
    CreateAccountRequest,
    ProfileSkillMutationResponse,
    SignInResponse,
)


def test_create_request_defaults_plan_and_optional_display_name():
    req = CreateAccountRequest(username="aisha", password="password1")
    assert req.display_name is None
    assert req.plan == {}


def test_sign_in_response_plan_optional():
    resp = SignInResponse(username="aisha", display_name="Aisha")
    assert resp.plan is None


def test_profile_skill_mutation_response_has_optional_fragments():
    resp = ProfileSkillMutationResponse(
        status="added",
        skill_id="s1",
        skill="SQL",
        snapshot={"professional_skills": [{"skill_id": "s1", "skill": "SQL"}]},
    )

    assert resp.gap_result is None
    assert resp.learned_skills == []
