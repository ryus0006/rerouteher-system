from app.schemas.employers import (
    EmployerMatchOut,
    EmployerMatchRequest,
    EmployerMatchResponse,
    LogoOut,
    ReportOut,
)


def test_request_defaults():
    req = EmployerMatchRequest(priorities=["flexible_work"])
    assert req.priorities == ["flexible_work"]
    assert req.target_role_id is None


def test_employer_out_allows_missing_report_and_logo():
    emp = EmployerMatchOut(
        id="cimb",
        name="CIMB",
        discloses=["flexible_work"],
        met=["flexible_work"],
        unmet=[],
    )
    dumped = emp.model_dump()
    assert dumped["report"] is None
    assert dumped["logo"] is None
    assert dumped["industry"] is None


def test_response_round_trips():
    resp = EmployerMatchResponse(
        employers=[
            EmployerMatchOut(
                id="nestle",
                name="Nestle Malaysia",
                industry="Consumer Goods",
                location="Petaling Jaya",
                logo=LogoOut(text="Nestle", bg="#00539f", fg="#ffffff"),
                website="https://www.nestle.com.my/",
                summary="...",
                discloses=["flexible_work", "parental_support"],
                report=ReportOut(label="Sustainability Report 2024", url="https://x"),
                met=["flexible_work"],
                unmet=["inclusive_workplace"],
            )
        ]
    )
    d = resp.model_dump()
    assert d["employers"][0]["logo"]["text"] == "Nestle"
    assert d["employers"][0]["report"]["url"] == "https://x"
