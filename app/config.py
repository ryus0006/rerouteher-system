"""Environment-driven settings."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/rerouteher"
    # Iteration 2 UI runs on 5174; 5173 kept so iteration 1 UI can also reach it.
    cors_origins: str = "http://localhost:5173,http://localhost:5174"

    # Local vendored model directory (checked into the repo). Loaded by path so there
    # is no Hugging Face Hub lookup at startup.
    embedding_model: str = "models/all-MiniLM-L6-v2"
    tfidf_model_path: str = "ml/tfidf_logreg.joblib"

    occupation_confidence_threshold: float = 0.65
    skill_cosine_threshold: float = 0.63
    # candidates retrieved per CV span before the cross-encoder re-ranks them by definition
    skill_candidate_k: int = 5
    # gap coverage: a role skill counts as covered when a user skill is at least this similar
    gap_cosine_threshold: float = 0.50

    # Occupation reranker (local cross-encoder). Vendored under models/ and loaded
    # from disk offline, like the embedder (the deploy image sets HF_HUB_OFFLINE=1).
    # Comma-separated paths tried in order; the first that loads wins.
    rerank_enabled: bool = True
    rerank_model_ids: str = "models/ms-marco-MiniLM-L6-v2"
    rerank_cache_dir: str | None = None  # unused for local paths; kept for override
    rerank_candidate_pool: int = 15
    rerank_top_k: int = 3

    ai_exposure_low: float = 0.1
    ai_exposure_medium: float = 0.15
    ai_exposure_high: float = 0.2

    max_cv_bytes: int = 10 * 1024 * 1024

    # Session cookie (Starlette SessionMiddleware). SESSION_SECRET signs the cookie;
    # set a strong random value in every deployed environment.
    session_secret: str = "dev-insecure-change-me"
    session_https_only: bool = False  # True behind HTTPS in production
    session_same_site: str = "lax"  # "none" (with https_only) for cross-site prod

    # E8 AI Companion (Gemini over direct HTTP, no SDK). GEMINI_API_KEY is read from
    # the environment / .env; without it the companion degrades to "not available".
    # gemini_model is verified against the list-models API, not assumed.
    gemini_api_key: str = ""
    # Fallback keys: when the active key is rate-limited (HTTP 429), the client
    # rotates to the next non-empty one. Set as many as you have quota for.
    gemini_api_key_2: str = ""
    gemini_api_key_3: str = ""
    gemini_model: str = "gemini-3.1-flash-lite"
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_timeout_s: float = 30.0

    # E6 on-demand learning fill (Tavily search + Gemini picks one free resource).
    learning_fill_enabled: bool = True
    learning_fill_url_timeout_s: float = 6.0
    # Web search for the fill (Tavily official API). Empty -> fill disabled.
    # Backup keys rotate on a rate-limit (HTTP 429), same as the Gemini keys.
    tavily_api_key: str = ""
    tavily_api_key_2: str = ""
    tavily_api_key_3: str = ""
    tavily_base_url: str = "https://api.tavily.com"
    tavily_timeout_s: float = 20.0
    learning_fill_candidates: int = 6

    @property
    def gemini_api_keys(self) -> list[str]:
        """Active key first, then fallbacks, skipping any that are unset."""
        return [
            k for k in (self.gemini_api_key, self.gemini_api_key_2, self.gemini_api_key_3) if k
        ]

    @property
    def llm_configured(self) -> bool:
        return bool(self.gemini_api_keys)

    @property
    def tavily_api_keys(self) -> list[str]:
        """Active key first, then backups, skipping any that are unset."""
        return [
            k for k in (self.tavily_api_key, self.tavily_api_key_2, self.tavily_api_key_3) if k
        ]

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
