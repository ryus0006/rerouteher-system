"""Transcript redaction (E7 AI Interview Coach).

This is the only boundary between raw spoken content and anything persisted or sent
to Gemini: only TranscriptRedactor.redact's return value may cross it. Deterministic
regex patterns catch structured PII (email, Malaysian phone numbers, Malaysian NRIC,
address-like text) even without spaCy; when an nlp pipeline is supplied (the same one
loaded once at startup for CV parsing), PERSON/GPE/LOC entities are redacted too.
"""
from __future__ import annotations

import re

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Malaysian mobile (01x) and landline (03-09) numbers only, to avoid catching ordinary
# figures like years of experience or amounts.
_PHONE_RE = re.compile(r"\b0(?:1[0-46-9]|[3-9])[\s-]?\d{3,4}[\s-]?\d{3,4}\b")
_NRIC_RE = re.compile(r"\b\d{6}-?\d{2}-?\d{4}\b")
_ADDRESS_RE = re.compile(
    r"(?:no\.?\s*\d+[a-z]?,?\s*)?\b(?:jalan|lorong|taman|persiaran|lot)\b[^.,;]*",
    re.IGNORECASE,
)

_REDACTED_ENTITY_LABELS = {"PERSON", "GPE", "LOC"}
_PLACEHOLDER = "[redacted]"


class TranscriptRedactor:
    def __init__(self, nlp=None, enabled: bool = True) -> None:
        self._nlp = nlp
        self._enabled = enabled

    def redact(self, text: str) -> str:
        if not text or not self._enabled:
            return text
        redacted = self._redact_entities(text) if self._nlp is not None else text
        redacted = _EMAIL_RE.sub(_PLACEHOLDER, redacted)
        redacted = _NRIC_RE.sub(_PLACEHOLDER, redacted)
        redacted = _PHONE_RE.sub(_PLACEHOLDER, redacted)
        redacted = _ADDRESS_RE.sub(_PLACEHOLDER, redacted)
        return redacted

    def _redact_entities(self, text: str) -> str:
        doc = self._nlp(text)
        spans = sorted(
            (
                (ent.start_char, ent.end_char)
                for ent in doc.ents
                if ent.label_ in _REDACTED_ENTITY_LABELS
            ),
            reverse=True,
        )
        for start, end in spans:
            text = text[:start] + _PLACEHOLDER + text[end:]
        return text
