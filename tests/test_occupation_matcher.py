"""Pure helpers of the occupation matcher: alias normalization, duration banding,
and the composed classifier text. No artifact or model needed."""
from app.services.occupation_matcher import compose_text, duration_band, normalize_text


def test_normalize_text_lowercases_and_applies_aliases():
    assert normalize_text("Software Engineer") == "software developer"
    assert normalize_text("HR Manager") == "human resources manager"
    assert normalize_text("Data Scientist") == "data analyst"
    assert normalize_text("  Business   Analyst ") == "business analyst"  # whitespace collapsed


def test_duration_band_boundaries():
    assert duration_band(None) == "work_length_unknown"
    assert duration_band(0.5) == "work_length_under_1_year"
    assert duration_band(2) == "work_length_1_to_3_years"
    assert duration_band(4) == "work_length_3_to_5_years"
    assert duration_band(7) == "work_length_5_to_10_years"
    assert duration_band(12) == "work_length_10_plus_years"


def test_compose_text_includes_title_skills_and_duration_band():
    txt = compose_text("Data Analyst", ["Excel", "SQL"], 4.0)
    assert "data analyst" in txt
    assert "excel" in txt and "sql" in txt
    assert "work_length_3_to_5_years" in txt


def test_compose_text_handles_no_skills():
    txt = compose_text("Nurse", [], None)
    assert "nurse" in txt
    assert "work_length_unknown" in txt
