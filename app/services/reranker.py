"""Local cross-encoder reranker. Scores (query, candidate) pairs jointly so
ranking is context-aware, unlike bi-encoder cosine. Loaded once at startup."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("rerouteher")


@dataclass
class RerankCandidate:
    role_id: str
    text: str


@dataclass
class RerankResult:
    role_id: str
    score: float


def _construct_cross_encoder(model_id: str, cache_dir: str | None):
    # imported lazily so the app boots for schema/route checks without torch,
    # and so a load failure is caught per-id by the caller.
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_id, cache_folder=cache_dir)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class CrossEncoderReranker:
    def __init__(self, model, model_id: str) -> None:
        self._model = model
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    @classmethod
    def load(cls, model_ids: list[str], cache_dir: str | None) -> "CrossEncoderReranker | None":
        """Try each id in order (best first); the first that constructs wins so the
        loader steps down to a lighter model when memory or download fails."""
        for model_id in model_ids:
            try:
                model = _construct_cross_encoder(model_id, cache_dir)
                logger.info("reranker loaded: %s", model_id)
                return cls(model, model_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("reranker %s failed to load (%s); trying next", model_id, exc, exc_info=True)
        logger.warning("no reranker model could be loaded; occupation reranking disabled")
        return None

    def rerank(self, query: str, candidates: list[RerankCandidate]) -> list[RerankResult]:
        if not candidates:
            return []
        pairs = [[query, c.text] for c in candidates]
        scores = _sigmoid(np.asarray(self._model.predict(pairs), dtype="float32"))
        return sorted(
            (RerankResult(c.role_id, float(s)) for c, s in zip(candidates, scores)),
            key=lambda r: r.score,
            reverse=True,
        )
