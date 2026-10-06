"""Tests for the CV extractor. Builds real PDFs in-memory with PyMuPDF.

spaCy is optional here (nlp=None): text extraction, regex segmentation, skill matching,
and PII redaction all work without it. ORG detection is covered separately when spaCy is present.
"""
import pymupdf
import pytest

from app.services.cv_extractor import CVExtractor, UnreadableCVError

SKILLS = ["project management", "stakeholder management", "budgeting", "python", "javascript"]


def _pdf(text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=11)
    return doc.tobytes()


def _extractor() -> CVExtractor:
    return CVExtractor(nlp=None, skill_dictionary=SKILLS)


def test_extracts_text_and_experiences():
    cv_text = (
        "Jane Doe\n"
        "jane.doe@example.com | +60 12-345 6789\n"
        "\n"
        "Work Experience\n"
        "Project Coordinator\n"
        "Acme Sdn Bhd\n"
        "Jan 2018 - Mar 2020\n"
        "Led project management and budgeting for cross-functional teams.\n"
        "\n"
        "Education\n"
        "BSc Computer Science, 2014\n"
    )
    cv = _extractor().parse(_pdf(cv_text))

    assert "Work Experience" in cv.raw_text
    assert len(cv.experiences) == 1
    exp = cv.experiences[0]
    assert exp.start == "Jan 2018"
    assert exp.end.lower() == "mar 2020"
    assert exp.title == "Project Coordinator"


def test_matches_skills_including_fuzzy():
    cv_text = (
        "Work Experience\n"
        "Manager\n"
        "2019 - 2021\n"
        "Responsible for projct management and budgeting.\n"  # typo: projct
    )
    cv = _extractor().parse(_pdf(cv_text))
    assert "budgeting" in cv.skill_mentions          # exact
    assert "project management" in cv.skill_mentions  # fuzzy over the typo


def test_pii_is_redacted_from_raw_text():
    cv = _extractor().parse(_pdf("Contact: jane.doe@example.com +60 12-345 6789\nWork Experience\nRole\n2020 - 2021\nDid things.\n"))
    assert "jane.doe@example.com" not in cv.raw_text
    assert "[email]" in cv.raw_text
    assert "[phone]" in cv.raw_text


class _Ent:
    def __init__(self, text, label, start):
        self.text = text
        self.label_ = label
        self.start_char = start
        self.end_char = start + len(text)


class _Doc:
    def __init__(self, ents):
        self.ents = ents


def _fake_nlp(spec):
    """Minimal spaCy stand-in: tags the given (surface, label) pairs wherever they occur."""
    def nlp(text):
        ents = []
        for surface, label in spec:
            index = text.find(surface)
            if index >= 0:
                ents.append(_Ent(surface, label, index))
        return _Doc(ents)

    return nlp


def test_redacts_name_and_address_but_keeps_embedded_place_names():
    nlp = _fake_nlp(
        [
            ("Kelvin Ku Teck Foong", "PERSON"),
            ("Subang Jaya", "GPE"),
            ("Selangor", "GPE"),
            ("Malaysia", "GPE"),
        ]
    )
    extractor = CVExtractor(nlp=nlp, skill_dictionary=SKILLS)
    cv = extractor.parse(
        _pdf(
            "Kelvin Ku Teck Foong\n"
            "Subang Jaya, Selangor\n"
            "Work Experience\n"
            "Manager\n"
            "Grab Malaysia\n"
            "2019 - 2021\n"
            "Did things.\n"
        )
    )
    # Name and address line are masked (majority obscured), not left readable.
    assert "Kelvin Ku Teck Foong" not in cv.raw_text
    assert "K*****" in cv.raw_text  # "Kelvin" -> first letter kept, rest masked
    assert "Subang Jaya" not in cv.raw_text and "Selangor" not in cv.raw_text
    assert "*" in cv.raw_text
    # A place name inside an employer line is not an address line and stays.
    assert "Grab Malaysia" in cv.raw_text


def test_scanned_pdf_is_unreadable():
    # a PDF page with no text layer (image-only) -> no extractable text
    doc = pymupdf.open()
    doc.new_page()  # blank, no text
    with pytest.raises(UnreadableCVError):
        _extractor().parse(doc.tobytes())
