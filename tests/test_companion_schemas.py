from app.schemas.companion import AskRequest, AskResponse, JourneyUpdate


def test_ask_request_minimal():
    req = AskRequest(question="hi", session_id="s1")
    assert req.current_page is None
    assert req.journey.cv is None


def test_ask_request_reads_journey_cv_and_break_alias():
    req = AskRequest(
        question="hi",
        session_id="s1",
        journey={
            "cv": {"raw_text": "x", "experiences": [], "skill_mentions": []},
            "break": {"duration_years": 3, "activities": ["caregiving"]},
        },
        current_page="cv-upload",
    )
    assert req.journey.cv.raw_text == "x"
    assert req.journey.break_.duration_years == 3


def test_journey_carries_role_skills_offered_flag():
    from app.schemas.companion import AskRequest

    req = AskRequest(
        question="hi",
        session_id="s1",
        journey={"roleSkillsOfferedForRoleId": "R1"},
    )
    assert req.journey.roleSkillsOfferedForRoleId == "R1"


def test_ask_response_carries_skill_choices_role_id():
    from app.schemas.companion import AskResponse, SkillChoice

    resp = AskResponse(
        answer="Here are skills for that role.",
        skill_choices=[SkillChoice(skill_id="s1", skill_name="SQL")],
        skill_choices_role_id="R1",
    )
    assert resp.model_dump()["skill_choices_role_id"] == "R1"
    assert AskResponse(answer="hi").skill_choices_role_id is None


def test_ask_response_carries_skill_choices():
    from app.schemas.companion import AskResponse, SkillChoice

    resp = AskResponse(
        answer="Here are skills common for a Marketing Manager.",
        skill_choices=[SkillChoice(skill_id="s1", skill_name="Campaign Management")],
    )
    d = resp.model_dump()
    assert d["skill_choices"][0]["skill_id"] == "s1"
    assert AskResponse(answer="hi").skill_choices is None


def test_journey_in_carries_confirmed_skills():
    from app.schemas.companion import AskRequest

    req = AskRequest(
        question="hi",
        session_id="s1",
        journey={"confirmedSkills": [{"skill_id": "s1", "skill_name": "SQL"}]},
    )
    assert req.journey.confirmedSkills[0].skill_id == "s1"


def test_ask_response_carries_optional_cta():
    from app.schemas.companion import AskResponse, CtaOut

    resp = AskResponse(
        answer="SQL is a priority gap because it is core to the role.",
        sources=["Your gap result"],
        cta=CtaOut(label="Open your learning plan", to="/plan/learning"),
    )
    d = resp.model_dump()
    assert d["cta"]["to"] == "/plan/learning"
    assert AskResponse(answer="hi").cta is None


def test_ask_response_round_trip_with_journey_update():
    resp = AskResponse(
        answer="done",
        sources=["Your CV"],
        journey_update=JourneyUpdate(
            cv={"raw_text": "x", "experiences": [], "skill_mentions": []},
            break_={"duration_years": 2, "activities": []},
        ),
    )
    d = resp.model_dump(by_alias=True)
    assert d["journey_update"]["break"]["duration_years"] == 2
    assert d["answer"] == "done"
