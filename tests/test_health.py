"""Smoke test: the app builds and /health responds (no DB or models needed for this route)."""
from fastapi.testclient import TestClient

from app.main import create_app


def test_health():
    # build the app without triggering lifespan (no DB/model needed for /health)
    app = create_app()
    with TestClient(app, raise_server_exceptions=True) as client:
        # lifespan runs here; if models/DB are absent it logs warnings but still boots
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
