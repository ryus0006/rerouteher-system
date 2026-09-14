from app.schemas.account import CreateAccountRequest, SignInResponse


def test_create_request_defaults_plan_and_optional_display_name():
    req = CreateAccountRequest(username="aisha", password="password1")
    assert req.display_name is None
    assert req.plan == {}


def test_sign_in_response_plan_optional():
    resp = SignInResponse(username="aisha", display_name="Aisha")
    assert resp.plan is None
