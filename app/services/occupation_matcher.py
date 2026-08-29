"""Occupation matcher: TF-IDF + Logistic Regression over ESCO occupations.

Loads a self-contained joblib artifact (LogReg pipeline + character-TF-IDF fallback
vectorizers/matrices + ESCO catalog) and returns the closest ESCO occupations for a CV.
Two-tier internally: LogisticRegression first, falling back to character-TF-IDF cosine
retrieval when the top probability is below the artifact's confidence threshold.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

_ALIASES = {
    r"\bhr\b": "human resources",
    r"\bhuman resource\b": "human resources",
    r"\badmin\b": "administrative",
    r"\bdev\b": "developer",
    r"\bsoftware engineer\b": "software developer",
    r"\bprogram+m?er\b": "software developer",
    r"\bcoder\b": "software developer",
    r"\bui[ /-]*ux\b": "user interface user experience",
    r"\bcsr\b": "customer service representative",
    r"\bsales rep\b": "sales representative",
    r"\bdata scientist\b": "data analyst",
    r"\baccounts assistant\b": "accounting assistant",
    r"\bwarehous(?:e)? operat(?:o)?r\b": "warehouse worker",
}


def normalize_text(value: str) -> str:
    text = value.lower().strip()
    for pattern, replacement in _ALIASES.items():
        text = re.sub(pattern, replacement, text)
    return re.sub(r"\s+", " ", text)


def duration_band(years: float | None) -> str:
    if years is None:
        return "work_length_unknown"
    if years < 1:
        return "work_length_under_1_year"
    if years < 3:
        return "work_length_1_to_3_years"
    if years < 5:
        return "work_length_3_to_5_years"
    if years < 10:
        return "work_length_5_to_10_years"
    return "work_length_10_plus_years"


def compose_text(job_title: str, skills: list[str], work_length_years: float | None) -> str:
    title = normalize_text(job_title)
    skill_text = ", ".join(normalize_text(skill) for skill in skills if skill.strip())
    return f"title {title}. job {title}. occupation {title}. skills {skill_text}. {duration_band(work_length_years)}"


@dataclass
class OccupationMatch:
    esco_code: str
    esco_title: str
    masco_code: str | None
    score: float
    method: str


class EscoTfidfMatcher:
    def __init__(self, artifact: dict[str, Any]) -> None:
        self.pipeline = artifact["pipeline"]
        self.title_vectorizer = artifact["title_vectorizer"]
        self.title_matrix = artifact["title_matrix"]
        self.skill_vectorizer = artifact["skill_vectorizer"]
        self.skill_matrix = artifact["skill_matrix"]
        self.catalog = artifact["catalog"]
        self.catalog_by_code = {row["esco_code"]: row for row in self.catalog}
        self.low_confidence_threshold = float(artifact.get("low_confidence_threshold", 0.45))

    @classmethod
    def load(cls, path: str) -> "EscoTfidfMatcher | None":
        p = Path(path)
        if not p.exists():
            return None
        # Load a trusted first-party artifact only; never point this at an externally supplied file.
        return cls(joblib.load(p))

    @staticmethod
    def _clean_masco(value: Any) -> str | None:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return str(value)

    def _match(self, profile: dict[str, Any], score: float, method: str) -> OccupationMatch:
        return OccupationMatch(
            esco_code=profile["esco_code"],
            esco_title=profile["esco_title"],
            masco_code=self._clean_masco(profile.get("masco_candidate_code")),
            score=round(float(score), 4),
            method=method,
        )

    def predict(
        self, job_title: str, skills: list[str], work_length_years: float | None = None, top_k: int = 3
    ) -> list[OccupationMatch]:
        text = compose_text(job_title, skills, work_length_years)
        probabilities = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.named_steps["classifier"].classes_
        order = np.argsort(probabilities)[::-1]

        if float(probabilities[order[0]]) >= self.low_confidence_threshold:
            matches: list[OccupationMatch] = []
            for idx in order[:top_k]:
                profile = self.catalog_by_code.get(str(classes[idx]))
                if profile:
                    matches.append(self._match(profile, probabilities[idx], "tfidf_logreg"))
            return matches

        # fallback: character-TF-IDF cosine retrieval over the title/skill matrices
        title_query = self.title_vectorizer.transform([normalize_text(job_title)])
        title_scores = cosine_similarity(title_query, self.title_matrix)[0]
        if skills:
            skill_query = self.skill_vectorizer.transform([" ".join(normalize_text(s) for s in skills)])
            skill_scores = cosine_similarity(skill_query, self.skill_matrix)[0]
        else:
            skill_scores = np.zeros_like(title_scores)
        scores = 0.82 * title_scores + 0.18 * skill_scores
        retrieval_order = np.argsort(scores)[::-1][:top_k]
        return [self._match(self.catalog[idx], scores[idx], "tfidf_retrieval") for idx in retrieval_order]
