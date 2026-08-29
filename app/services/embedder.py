"""all-MiniLM-L6-v2 wrapper. Loaded once at startup and reused (CPU, 384-dim)."""
from __future__ import annotations

import numpy as np


class Embedder:
    def __init__(self, model_name: str) -> None:
        # imported lazily so the app can boot for schema/route checks without the model
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n, 384) float32 array."""
        return self._model.encode(texts, normalize_embeddings=True)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
