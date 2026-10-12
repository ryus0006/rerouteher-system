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


def test_masks_all_caps_header_name_and_keeps_employer_with_city():
    # No spaCy: the header is masked positionally, so an all-caps name (which NER misses)
    # is still redacted, while an employer line that contains a city is left intact.
    cv = _extractor().parse(
        _pdf(
            "SITI NUR AISYAH BINTI ABDULLAH\n"
            "Kuala Lumpur, Malaysia\n"
            "siti@example.com | +60 12-345 6789\n"
            "Work Experience\n"
            "Senior Software Developer\n"
            "Axiata Digital Labs, Kuala Lumpur\n"
            "Jan 2021 - Dec 2022\n"
            "Built backend services.\n"
        )
    )
    assert "SITI NUR AISYAH BINTI ABDULLAH" not in cv.raw_text  # all-caps name masked
    assert "S***" in cv.raw_text
    assert "Kuala Lumpur, Malaysia" not in cv.raw_text  # header address line masked
    assert "Axiata Digital Labs, Kuala Lumpur" in cv.raw_text  # employer line untouched


def test_masks_combined_header_line_keeping_contact_placeholders():
    # One header line holds address, contact and a profile URL. The contact placeholders
    # survive while the address and the name-bearing URL are masked.
    cv = _extractor().parse(
        _pdf(
            "NURUL AIN BINTI HASHIM\n"
            "Shah Alam, Selangor, Malaysia | nurul.ain@email.com | +60198765432 | "
            "linkedin.com/in/nurul-ain-hashim\n"
            "Professional Summary\n"
            "Mechanical Engineer with experience in HVAC.\n"
        )
    )
    rt = cv.raw_text
    assert "[email]" in rt and "[phone]" in rt  # contact placeholders preserved
    assert "NURUL AIN BINTI HASHIM" not in rt  # name masked
    assert "Shah Alam" not in rt  # address masked
    assert "nurul-ain-hashim" not in rt  # profile URL handle (embeds the name) masked


def test_scanned_pdf_is_unreadable():
    # a PDF page with no text layer (image-only) -> no extractable text
    doc = pymupdf.open()
    doc.new_page()  # blank, no text
    with pytest.raises(UnreadableCVError):
        _extractor().parse(doc.tobytes())
