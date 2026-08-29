"""NLP CV extractor.

Deterministic NLP, no model training. PDF only, no OCR, no PII. In-memory only.

Pipeline:
1. PyMuPDF text with column-aware, bounding-box reading order.
2. spaCy NER + regex rules segment the work-history section into experiences.
3. rapidfuzz matches skill mentions against the skill dictionary.
"""
from __future__ import annotations

import re

import pymupdf  # PyMuPDF (the modern import; `fitz` is deprecated)

from app.schemas.cv import CV, Experience


class UnreadableCVError(Exception):
    """Raised when the PDF has no extractable text layer (scanned/image PDF)."""


# --- PII (never emitted in structured fields; also redacted from raw_text) ---
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"(?<!\w)(\+?\d[\d\s().\-]{7,}\d)(?!\w)")

# --- Section headers ---
_EXPERIENCE_HEADER_RE = re.compile(
    r"^\s*(work\s+experience|professional\s+experience|employment(?:\s+history)?"
    r"|work\s+history|career\s+history|experience)\s*:?\s*$",
    re.IGNORECASE,
)
_OTHER_SECTION_RE = re.compile(
    r"^\s*(education|skills?|technical\s+skills?|certifications?|projects?|awards?"
    r"|references?|summary|profile|objective|interests?|languages?|volunteer"
    r"|publications?|achievements?|contact)\s*:?\s*$",
    re.IGNORECASE,
)

# --- Date ranges (anchor each experience entry) ---
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_DATE = rf"(?:{_MONTH}\s*)?(?:\d{{1,2}}[/-])?\d{{4}}"
_DATE_RANGE_RE = re.compile(
    rf"({_DATE})\s*(?:-|–|—|to|until|\bthrough\b)\s*({_DATE}|present|current|now|ongoing)",
    re.IGNORECASE,
)


class CVExtractor:
    def __init__(self, nlp, skill_dictionary: list[str]) -> None:
        # nlp: a loaded spaCy pipeline (may be None in degraded mode)
        # skill_dictionary: skill terms to match against (canonical names + aliases)
        self._nlp = nlp
        self._skill_dictionary = [t for t in skill_dictionary if t and len(t) >= 3]

    def parse(self, pdf_bytes: bytes) -> CV:
        raw_text = self._extract_text(pdf_bytes)
        if not raw_text.strip():
            # no text layer -> scanned/image PDF. We do not OCR.
            raise UnreadableCVError("unreadable, please upload a text-based PDF")
        raw_text = self._redact_pii(raw_text)
        experiences = self._segment_experiences(raw_text)
        skill_mentions = self._match_skills(raw_text)
        return CV(raw_text=raw_text, experiences=experiences, skill_mentions=skill_mentions)

    # ------------------------------------------------------------------ text
    def _extract_text(self, pdf_bytes: bytes) -> str:
        parts: list[str] = []
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                parts.append(self._page_text(page))
        return "\n".join(parts).strip()

    def _page_text(self, page) -> str:
        # blocks: (x0, y0, x1, y1, text, block_no, block_type); type 0 = text
        blocks = [
            b for b in page.get_text("blocks") if len(b) >= 7 and b[6] == 0 and b[4].strip()
        ]
        if not blocks:
            return ""
        ordered = self._reading_order(blocks, page.rect.width)
        return "\n".join(b[4].strip() for b in ordered)

    @staticmethod
    def _reading_order(blocks: list, page_width: float) -> list:
        """Column-aware sort so two-column CVs keep reading order.

        Heuristic: if content splits cleanly into a left and a right column with a
        near-empty gutter down the middle, read the whole left column then the whole
        right column. Otherwise fall back to a plain top-to-bottom, left-to-right sort.
        """
        mid = page_width / 2
        left = [b for b in blocks if (b[0] + b[2]) / 2 < mid]
        right = [b for b in blocks if (b[0] + b[2]) / 2 >= mid]
        crossing = [b for b in blocks if b[0] < mid < b[2]]  # blocks spanning the gutter

        two_column = (
            len(left) >= 2
            and len(right) >= max(2, 0.2 * len(blocks))
            and len(crossing) <= max(1, 0.1 * len(blocks))
        )
        if two_column:
            key = lambda b: (round(b[1], 1), b[0])  # noqa: E731
            return sorted(left, key=key) + sorted(right, key=key)
        return sorted(blocks, key=lambda b: (round(b[1], 1), b[0]))

    # ------------------------------------------------------------- experience
    def _segment_experiences(self, raw_text: str) -> list[Experience]:
        lines = [ln.rstrip() for ln in raw_text.splitlines()]
        section = self._experience_section(lines)
        entries = self._split_entries(section)
        experiences = [self._build_experience(entry) for entry in entries]
        return [e for e in experiences if e is not None]

    def _experience_section(self, lines: list[str]) -> list[str]:
        """Return the work-history lines: from the experience header to the next section.

        If no explicit header is present, fall back to every line, so the date-range
        splitter downstream can still recover entries.
        """
        start = None
        for i, line in enumerate(lines):
            if _EXPERIENCE_HEADER_RE.match(line):
                start = i + 1
                break
        if start is None:
            return lines
        end = len(lines)
        for j in range(start, len(lines)):
            if _OTHER_SECTION_RE.match(lines[j]):
                end = j
                break
        return lines[start:end]

    @staticmethod
    def _split_entries(section: list[str]) -> list[list[str]]:
        """Group lines into entries, starting a new entry at each dated line."""
        entries: list[list[str]] = []
        current: list[str] = []
        for line in section:
            has_date = bool(_DATE_RANGE_RE.search(line))
            already_dated = any(_DATE_RANGE_RE.search(ln) for ln in current)
            if has_date and already_dated:
                entries.append(current)
                current = [line]
            elif has_date and current and not already_dated:
                current.append(line)
            elif not line.strip():
                if already_dated:
                    entries.append(current)
                    current = []
            else:
                current.append(line)
        if current and any(_DATE_RANGE_RE.search(ln) for ln in current):
            entries.append(current)
        return entries

    def _build_experience(self, entry: list[str]) -> Experience | None:
        text = "\n".join(entry).strip()
        date_match = _DATE_RANGE_RE.search(text)
        if date_match is None:
            return None
        start, end = date_match.group(1).strip(), date_match.group(2).strip()

        organisation = self._first_org(text)

        # content lines with any inline date range stripped; drop lines that were only a date
        clean_lines: list[str] = []
        for raw in entry:
            stripped = _DATE_RANGE_RE.sub("", raw).strip(" -–—|,\t")
            if stripped:
                clean_lines.append(stripped)

        # title: first content line that is not the organisation
        title = None
        for line in clean_lines:
            if organisation and line.lower() == organisation.lower():
                continue
            title = line
            break

        # organisation fallback: the next content line after the title
        if organisation is None:
            for line in clean_lines:
                if line != title:
                    organisation = line
                    break

        described = [
            line
            for line in clean_lines
            if line != title and (organisation is None or line.lower() != organisation.lower())
        ]
        description = " ".join(described).strip() or None
        return Experience(
            title=title,
            organisation=organisation,
            start=start,
            end=end,
            description=description,
        )

    def _first_org(self, text: str) -> str | None:
        if self._nlp is None:
            return None
        doc = self._nlp(text)
        for ent in doc.ents:
            if ent.label_ == "ORG":
                return ent.text.strip()
        return None

    # ----------------------------------------------------------------- skills
    def _match_skills(self, raw_text: str) -> list[str]:
        if not self._skill_dictionary:
            return []
        from rapidfuzz import fuzz, process

        # match against word/phrase tokens (not raw substrings) so short terms like
        # "git" or "c" do not match inside larger words ("digital", "communication").
        phrases = self._candidate_phrases(raw_text.lower())
        matched: dict[str, str] = {}  # lowercased term -> surface term (dedupe case variants)
        for term in self._skill_dictionary:
            term_l = term.lower()
            if term_l in matched:
                continue
            if term_l in phrases:  # exact, word-boundary safe
                matched[term_l] = term
            elif process.extractOne(term_l, phrases, scorer=fuzz.ratio, score_cutoff=90) is not None:
                matched[term_l] = term
        return sorted(matched.values())

    @staticmethod
    def _candidate_phrases(text_lower: str) -> set[str]:
        words = re.findall(r"[a-z][a-z0-9+.#-]*", text_lower)
        phrases: set[str] = set()
        for n in (1, 2, 3, 4):
            for i in range(len(words) - n + 1):
                phrases.add(" ".join(words[i : i + n]))
        return phrases

    # ------------------------------------------------------------------- pii
    @staticmethod
    def _redact_pii(text: str) -> str:
        text = _EMAIL_RE.sub("[email]", text)
        text = _PHONE_RE.sub("[phone]", text)
        return text
