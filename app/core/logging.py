"""Request access logging for every API call (pure ASGI, no BaseHTTPMiddleware).

Logs method, path, status and duration only. Request/response bodies are never
logged: they carry CV-derived personal data. The model inputs actually used
(titles, skills, the embedding/reranker query) are logged in the service layer.
"""

from __future__ import annotations

import logging
import time

from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("rerouteher.request")


def configure_logging(level: int = logging.INFO) -> None:
    """Set a log format and level on the root logger so all app logs reach stdout."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=level
    )


class RequestLoggingMiddleware:
    """Log method, path, status and duration. Bodies are never logged (PII)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        status = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "%s %s failed after %.1f ms", scope["method"], scope["path"], elapsed_ms
            )
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("%s %s -> %d (%.1f ms)", scope["method"], scope["path"], status, elapsed_ms)
