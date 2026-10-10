def test_request_carries_confirmed_skills_and_break_alias():
    from app.schemas.snapshot import SnapshotRequest

    req = SnapshotRequest(
        cv={"raw_text": "x", "experiences": [], "skill_mentions": []},
        **{"break": {"duration_years": 1, "activities": ["care_household.ran_household"]}},
        confirmed_skills=["s1", "s2"],
    )
    assert req.break_.activities == ["care_household.ran_household"]
    assert req.confirmed_skills == ["s1", "s2"]


def test_professional_skill_allows_role_confirmed_source():
    from app.schemas.snapshot import ProfessionalSkill

    ps = ProfessionalSkill(skill="SQL", skill_id="s1", source="role_confirmed")
    assert ps.source == "role_confirmed"


def test_skill_definition_is_optional_and_serializes_as_null():
    from app.schemas.snapshot import ProfessionalSkill, ReframedSkill

    professional = ProfessionalSkill(skill="SQL")
    reframed = ReframedSkill(skill="Coordination")

    assert professional.definition is None
    assert reframed.definition is None
    assert professional.model_dump()["definition"] is None
    assert reframed.model_dump()["definition"] is None
