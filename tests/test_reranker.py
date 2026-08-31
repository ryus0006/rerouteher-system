"""Cross-encoder reranker: ordering, 0-1 normalisation, and the load ladder.

The model is injected as a plain object exposing .predict(pairs), so these run
without torch or any download.
"""
import numpy as np

from app.services.reranker import CrossEncoderReranker, RerankCandidate


class FakeModel:
    """Returns a fixed logit per candidate text, order-independent."""

    def __init__(self, scores_by_text):
        self.scores_by_text = scores_by_text
        self.calls = []

    def predict(self, pairs):
        self.calls.append(pairs)
        return np.array([self.scores_by_text[text] for _query, text in pairs], dtype="float32")


def test_rerank_orders_by_score_desc_and_normalises_to_0_1():
    model = FakeModel(
        {"data analyst role": 6.0, "manufacturing manager role": -4.0, "food analyst role": 0.0}
    )
    r = CrossEncoderReranker(model, "fake")
    out = r.rerank(
        "operations analyst. skills: excel, sql, power bi",
        [
            RerankCandidate("role_mfg", "manufacturing manager role"),
            RerankCandidate("role_data", "data analyst role"),
            RerankCandidate("role_food", "food analyst role"),
        ],
    )
    assert [x.role_id for x in out] == ["role_data", "role_food", "role_mfg"]
    assert all(0.0 <= x.score <= 1.0 for x in out)
    assert out[0].score > 0.9 and out[-1].score < 0.1
    # one batched predict call, query paired with each candidate text
    assert len(model.calls) == 1
    assert model.calls[0][0][0].startswith("operations analyst")


def test_rerank_empty_candidates_returns_empty():
    assert CrossEncoderReranker(FakeModel({}), "fake").rerank("q", []) == []


def test_load_returns_none_when_all_model_ids_fail(monkeypatch):
    import app.services.reranker as mod

    def boom(model_id, cache_dir):
        raise RuntimeError(f"cannot load {model_id}")

    monkeypatch.setattr(mod, "_construct_cross_encoder", boom)
    assert CrossEncoderReranker.load(["a", "b"], None) is None


def test_load_uses_first_model_that_constructs(monkeypatch):
    import app.services.reranker as mod

    built = []

    def maybe(model_id, cache_dir):
        built.append(model_id)
        if model_id == "good":
            return object()
        raise RuntimeError("nope")

    monkeypatch.setattr(mod, "_construct_cross_encoder", maybe)
    r = CrossEncoderReranker.load(["bad", "good", "unused"], None)
    assert r is not None and r.model_id == "good"
    assert built == ["bad", "good"]  # stops at first success
