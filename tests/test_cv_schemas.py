from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas.cv import (
    CvDraft,
    CvGenerateRequest,
    CvGenerateResponse,
    CvImproveRequest,
    CvImproveResponse,
)


def test_generate_request_accepts_optional_role_alias_and_regenerate():
    default = CvGenerateRequest()
    request = CvGenerateRequest.model_validate({"roleId": "role-project", "regenerate": True})

    assert default.role_id is None
    assert default.regenerate is False
    assert request.role_id == "role-project"
    assert request.regenerate is True


def test_generate_request_rejects_blank_role_id():
    with pytest.raises(ValidationError):
        CvGenerateRequest.model_validate({"roleId": ""})


def test_improve_request_accepts_ui_aliases_and_bounds_index():
    request = CvImproveRequest.model_validate(
        {
            "roleId": "role-project",
            "section": "experience",
            "experienceIndex": 0,
            "currentText": "- Coordinated delivery.",
            "previousSuggestions": ["Prior wording."],
        }
    )

    assert request.role_id == "role-project"
    assert request.experience_index == 0
    assert request.current_text == "- Coordinated delivery."
    assert request.previous_suggestions == ["Prior wording."]

    with pytest.raises(ValidationError):
        CvImproveRequest.model_validate(
            {"roleId": "role-project", "section": "unknown"}
        )

    with pytest.raises(ValidationError):
        CvImproveRequest.model_validate(
            {"roleId": "role-project", "section": "experience", "experienceIndex": -1}
        )


def test_response_serialization_keeps_normalized_draft_shape():
    draft = CvDraft.model_validate(
        {
            "roleId": "role-project",
            "personal": {},
            "summary": "Operations professional.",
            "skills": ["Project coordination"],
            "experiences": [{"description": "- Coordinated delivery."}],
            "careerBreak": None,
        }
    )
    generated_at = datetime(2026, 10, 4, tzinfo=timezone.utc)
    response = CvGenerateResponse(
        role_id="role-project",
        generation_status="generated",
        generated_at=generated_at,
        draft=draft,
    )
    improvement = CvImproveResponse(
        section="summary",
        suggestion="Operations professional with project coordination experience.",
        evidence="Coordinated delivery.",
    )

    assert response.draft.role_id == "role-project"
    assert response.draft.personal.name == ""
    assert response.draft.experiences[0].description == "- Coordinated delivery."
    assert response.model_dump()["generation_status"] == "generated"
    assert response.model_dump()["draft"]["role_id"] == "role-project"
    assert improvement.model_dump()["section"] == "summary"
