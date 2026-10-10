"""Transcript redaction: the only boundary between raw speech and anything persisted
or sent to Gemini. spaCy is faked throughout (the real pipeline is loaded once at
startup and reused, same as CV parsing); these tests pin the redaction contract.
"""
from app.services.interview_privacy import TranscriptRedactor


class _Ent:
    def __init__(self, start_char, end_char, label_):
        self.start_char = start_char
        self.end_char = end_char
        self.label_ = label_


class _Doc:
    def __init__(self, ents):
        self.ents = ents


class FakeNlp:
    """Returns pre-wired entities for a given input text."""

    def __init__(self, ents_by_text: dict[str, list[_Ent]]):
        self._ents_by_text = ents_by_text
        self.calls: list[str] = []

    def __call__(self, text: str) -> _Doc:
        self.calls.append(text)
        return _Doc(self._ents_by_text.get(text, []))


def test_redacts_email_addresses():
    r = TranscriptRedactor(nlp=None)
    out = r.redact("You can reach me at aisha.hassan@example.com for references.")
    assert "aisha.hassan@example.com" not in out
    assert "[redacted]" in out


def test_redacts_malaysian_phone_numbers():
    r = TranscriptRedactor(nlp=None)
    out = r.redact("My number is 012-345 6789 if you need to follow up.")
    assert "012-345 6789" not in out
    assert "[redacted]" in out


def test_redacts_malaysian_nric():
    r = TranscriptRedactor(nlp=None)
    out = r.redact("My IC number is 901231-14-5678 for verification.")
    assert "901231-14-5678" not in out
    assert "[redacted]" in out


def test_redacts_address_like_text():
    r = TranscriptRedactor(nlp=None)
    out = r.redact("I used to live at No. 12, Jalan Bukit Bintang, before moving.")
    assert "Jalan Bukit Bintang" not in out


def test_without_spacy_available_entities_are_not_redacted():
    r = TranscriptRedactor(nlp=None)
    out = r.redact("My name is Aisha and I worked in Kuala Lumpur for five years.")
    assert "Aisha" in out
    assert "Kuala Lumpur" in out


def test_with_spacy_person_gpe_and_loc_entities_are_redacted():
    text = "My name is Aisha and I worked in Kuala Lumpur for five years."
    ents = [
        _Ent(text.index("Aisha"), text.index("Aisha") + len("Aisha"), "PERSON"),
        _Ent(text.index("Kuala Lumpur"), text.index("Kuala Lumpur") + len("Kuala Lumpur"), "GPE"),
    ]
    nlp = FakeNlp({text: ents})
    r = TranscriptRedactor(nlp=nlp)
    out = r.redact(text)
    assert "Aisha" not in out
    assert "Kuala Lumpur" not in out
    assert "five years" in out  # ordinary career evidence remains intact


def test_entity_replacement_applies_right_to_left_so_offsets_stay_valid():
    text = "Aisha managed a team in Penang and later in Johor Bahru."
    ents = [
        _Ent(text.index("Aisha"), text.index("Aisha") + len("Aisha"), "PERSON"),
        _Ent(text.index("Penang"), text.index("Penang") + len("Penang"), "GPE"),
        _Ent(text.index("Johor Bahru"), text.index("Johor Bahru") + len("Johor Bahru"), "GPE"),
    ]
    nlp = FakeNlp({text: ents})
    r = TranscriptRedactor(nlp=nlp)
    out = r.redact(text)
    assert "Aisha" not in out
    assert "Penang" not in out
    assert "Johor Bahru" not in out
    assert out.count("[redacted]") == 3
    assert "managed a team in" in out
    assert "and later in" in out


def test_non_person_gpe_loc_labels_are_left_alone():
    text = "I joined Shopee as an analyst."
    ents = [_Ent(text.index("Shopee"), text.index("Shopee") + len("Shopee"), "ORG")]
    nlp = FakeNlp({text: ents})
    r = TranscriptRedactor(nlp=nlp)
    out = r.redact(text)
    assert "Shopee" in out  # ORG is not in the redacted label set


def test_ordinary_career_evidence_remains_intact():
    r = TranscriptRedactor(nlp=None)
    text = (
        "I led a team of five, improved reporting turnaround by 30 percent, "
        "and managed a budget of RM50,000 over two years."
    )
    out = r.redact(text)
    assert out == text


def test_empty_transcript_returns_empty():
    r = TranscriptRedactor(nlp=None)
    assert r.redact("") == ""


def test_disabled_redactor_returns_text_unchanged():
    nlp = FakeNlp({"My name is Aisha and I live in Johor.": [_Ent(11, 16, "PERSON")]})
    redactor = TranscriptRedactor(nlp=nlp, enabled=False)
    text = "My name is Aisha and I live in Johor. Call 012-3456789."
    assert redactor.redact(text) == text
    assert nlp.calls == []
