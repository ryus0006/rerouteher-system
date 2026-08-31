"""The lifespan loads the reranker and injects it into SnapshotService.

Heavy model loads are stubbed so this runs without torch, the joblib artifact, or a DB.
"""
import pytest

import app.main as main_mod
from app.services.reranker import CrossEncoderReranker

pytestmark = pytest.mark.asyncio


class _SentinelReranker:
    model_id = "sentinel-model"


async def test_lifespan_wires_reranker(monkeypatch):
    sentinel = _SentinelReranker()
    captured: dict = {}

    def _no_embedder(model_name):
        raise RuntimeError("skip embedder in test")

    monkeypatch.setattr(main_mod, "Embedder", _no_embedder)
    monkeypatch.setattr(main_mod.EscoTfidfMatcher, "load", staticmethod(lambda path: None))
    monkeypatch.setattr(
        CrossEncoderReranker, "load", classmethod(lambda cls, ids, cache: sentinel)
    )

    real_init = main_mod.SnapshotService.__init__

    def spy(self, **kwargs):
        captured.update(kwargs)
        real_init(self, **kwargs)

    monkeypatch.setattr(main_mod.SnapshotService, "__init__", spy)

    app = main_mod.create_app()
    async with main_mod.lifespan(app):
        pass

    assert captured.get("reranker") is sentinel
    assert app.state.snapshot_service is not None
