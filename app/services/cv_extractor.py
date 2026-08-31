"""NLP CV extractor.

Deterministic NLP, no model training. PDF only, no OCR, no PII. In-memory only.

Pipeline:
1. PyMuPDF text with column-aware, bounding-box reading order.
2. spaCy NER + regex rules segment the work-history section into experiences.
3. rapidfuzz matches skill mentions against the skill dictionary.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pymupdf  # PyMuPDF (the modern import; `fitz` is deprecated)

from app.schemas.cv import CV, Experience

logger = logging.getLogger("rerouteher")


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

# --- Semantic section detection (embedding fallback for headers regex misses) ---
# A short line whose embedding is this close to a header prototype is a section header.
_HEADER_SIM_THR = 0.62
_MAX_HEADER_WORDS = 4
_SECTION_PHRASES = {
    "EXPERIENCE": [
        "work experience",
        "professional experience",
        "employment history",
        "work history",
        "experience",
    ],
    "OTHER": [
        "education",
        "skills",
        "technical skills",
        "core skills",
        "software knowledge",
        "projects",
        "summary",
        "profile",
        "objective",
        "certifications",
        "languages",
        "awards",
        "references",
        "contact",
        "personal information",
        "hobbies",
        "interests",
    ],
}

# --- Date ranges (anchor each experience entry) ---
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
_DATE = rf"(?:{_MONTH}\s*)?(?:\d{{1,2}}[/-])?\d{{4}}"
_DATE_RANGE_RE = re.compile(
    rf"({_DATE})\s*(?:-|–|—|to|until|\bthrough\b)\s*({_DATE}|present|current|now|ongoing)",
    re.IGNORECASE,
)


class CVExtractor:
    def __init__(self, nlp, skill_dictionary: list[str], embedder=None) -> None:
        # nlp: a loaded spaCy pipeline (may be None in degraded mode)
        # skill_dictionary: skill terms to match against (canonical names + aliases)
        # embedder: sentence embedder for semantic section detection (may be None)
        self._nlp = nlp
        self._skill_dictionary = [t for t in skill_dictionary if t and len(t) >= 3]
        self._embedder = embedder
        self._proto_mat = None  # (N, dim) L2-normalized header prototypes
        self._proto_labels: list[str] = []
        if embedder is not None:
            phrases, labels = [], []
            for label, terms in _SECTION_PHRASES.items():
                for term in terms:
                    phrases.append(term)
                    labels.append(label)
            mat = np.asarray(embedder.encode(phrases), dtype="float32")
            mat /= np.linalg.norm(mat, axis=1, keepdims=True)
            self._proto_mat = mat
            self._proto_labels = labels

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
        experiences = self._experiences_from(section)
        # Layouts the sectioner cannot follow (e.g. a two-column reading order that
        # interleaves sections) can leave the experience section empty. Fall back to
        # the whole document so the date splitter still recovers entries.
        if not experiences and section is not lines:
            experiences = self._experiences_from(lines)
        return experiences

    def _experiences_from(self, section_lines: list[str]) -> list[Experience]:
        entries = self._split_entries(section_lines)
        experiences = [self._build_experience(entry) for entry in entries]
        return [e for e in experiences if e is not None]

    def _experience_section(self, lines: list[str]) -> list[str]:
        """Return the work-history lines by tracking the current section.

        Section headers are detected first by regex (certain, cheap) and, for
        headers the regex misses, by embedding similarity (when an embedder is
        available). Lines start in the PERSONAL/contact block, so the candidate's
        name and contact details are never read as an experience. If no experience
        header is found at all, fall back to every line so the date-range splitter
        can still recover entries.
        """
        exp: list[str] = []
        current = "PERSONAL"
        found_experience = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current == "EXPERIENCE":
                    exp.append(line)  # keep blank lines: the entry splitter uses them
                continue
            if _EXPERIENCE_HEADER_RE.match(stripped):
                current = "EXPERIENCE"
                found_experience = True
                continue
            if _OTHER_SECTION_RE.match(stripped):
                current = "OTHER"
                continue
            section = self._semantic_section(stripped)
            if section is not None:
                current = section
                found_experience = found_experience or section == "EXPERIENCE"
                continue
            if current == "EXPERIENCE":
                exp.append(line)
        if found_experience:
            return exp
        return lines

    def _semantic_section(self, line: str) -> str | None:
        """Embedding-based header detection: 'EXPERIENCE', 'OTHER', or None (not a header)."""
        if self._proto_mat is None:
            return None
        if not (1 <= len(line.split()) <= _MAX_HEADER_WORDS):
            return None
        if _DATE_RANGE_RE.search(line):  # a dated line is content, not a header
            return None
        vec = np.asarray(self._embedder.encode_one(line), dtype="float32")
        norm = np.linalg.norm(vec)
        if norm == 0:
            return None
        sims = self._proto_mat @ (vec / norm)
        best = int(np.argmax(sims))
        if sims[best] < _HEADER_SIM_THR:
            return None
        return self._proto_labels[best]

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
            stripped = _DATE_RANGE_RE.sub("", raw).strip(" -–—|,:\t")
            if stripped:
                clean_lines.append(stripped)

        # title: prefer the text on the dated line (many CVs write "DATE : Title" or
        # "Title  DATE"), which is the actual role; otherwise the first non-org line.
        title = None
        title_source = None
        for raw in entry:
            if _DATE_RANGE_RE.search(raw):
                residual = _DATE_RANGE_RE.sub("", raw).strip(" -–—|,:()[]{}\t")
                has_words = re.search(r"[A-Za-z]", residual) is not None
                if has_words and (organisation is None or residual.lower() != organisation.lower()):
                    title = residual
                    title_source = "dated_line"
                break
        if title is None:
            for line in clean_lines:
                if organisation and line.lower() == organisation.lower():
                    continue
                title = line
                title_source = "first_non_org_line"
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
        # Diagnostic: how each entry was segmented, so title mislabels (a company/location
        # line or a career-break bullet promoted to a title) and the branch that chose them
        # are visible. spacy_org=None means NER found no organisation for this entry.
        logger.info(
            "cv segment: title=%r (via %s) org=%r dates=%s->%s | entry_lines=%s",
            title, title_source, organisation, start, end,
            [ln.strip() for ln in entry][:6],
        )
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
