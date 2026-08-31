"""Environment-driven settings."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/rerouteher"
    cors_origins: str = "http://localhost:5173"

    # Local vendored model directory (checked into the repo). Loaded by path so there
    # is no Hugging Face Hub lookup at startup.
    embedding_model: str = "models/all-MiniLM-L6-v2"
    tfidf_model_path: str = "ml/tfidf_logreg.joblib"

    occupation_confidence_threshold: float = 0.65
    skill_cosine_threshold: float = 0.63
    # gap coverage: a role skill counts as covered when a user skill is at least this similar
    gap_cosine_threshold: float = 0.50

    # Occupation reranker (local cross-encoder). Models are downloaded from
    # HuggingFace at first boot and cached; nothing large is committed to Git.
    # Best-first ladder: the first id that loads wins (steps down if memory is tight).
    rerank_enabled: bool = True
    rerank_model_ids: str = (
        "cross-encoder/ms-marco-MiniLM-L6-v2,"
        "cross-encoder/ms-marco-MiniLM-L4-v2,"
        "cross-encoder/ms-marco-MiniLM-L2-v2"
    )
    rerank_cache_dir: str | None = None  # None -> default HF cache (HF_HOME)
    rerank_candidate_pool: int = 15
    rerank_top_k: int = 3

    ai_exposure_low: float = 0.1
    ai_exposure_medium: float = 0.15
    ai_exposure_high: float = 0.2

    max_cv_bytes: int = 10 * 1024 * 1024

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def rerank_model_id_list(self) -> list[str]:
        return [m.strip() for m in self.rerank_model_ids.split(",") if m.strip()]

    def ai_exposure_weight(self, level: str) -> float:
        return {
            "low": self.ai_exposure_low,
            "medium": self.ai_exposure_medium,
            "high": self.ai_exposure_high,
        }.get(level, self.ai_exposure_medium)


@lru_cache
def get_settings() -> Settings:
    return Settings()
