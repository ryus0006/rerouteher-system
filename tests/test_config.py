"""Reranker settings defaults and model-id parsing."""
from app.config import Settings


def test_rerank_defaults_and_model_list():
    s = Settings()
    assert s.rerank_enabled is True
    assert s.rerank_top_k == 3
    assert s.rerank_candidate_pool == 15
    ids = s.rerank_model_id_list
    assert ids == ["models/ms-marco-MiniLM-L6-v2"]  # vendored local path, loaded offline
    assert all(i.strip() == i and i for i in ids)
