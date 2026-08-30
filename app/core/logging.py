"""Request access logging for every API call (pure ASGI, no BaseHTTPMiddleware)."""

from __future__ import annotations

import logging
import time

from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("rerouteher.request")

# Bodies are truncated to this many bytes in the log; enough to capture the snapshot
# occupation/method and the gap readiness without flooding the log with every skill.
MAX_BODY_LOG = 4000


def configure_logging(level: int = logging.INFO) -> None:
    """Set a log format and level on the root logger so all app logs reach stdout."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=level
    )


def _preview(body: bytes) -> str:
    if not body:
        return "-"
    text = body[:MAX_BODY_LOG].decode("utf-8", "replace")
    return text + "...(truncated)" if len(body) > MAX_BODY_LOG else text


class RequestLoggingMiddleware:
    """Log method, path, status, duration, and request/response bodies."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()

        # Buffer the request body so it can be logged, then replayed to the app.
        request_body = b""
        while True:
            message = await receive()
            if message["type"] == "http.request":
                if len(request_body) < MAX_BODY_LOG:
                    request_body += message.get("body", b"")
                if not message.get("more_body", False):
                    break
            elif message["type"] == "http.disconnect":
                break

        replayed = False

        async def replay_receive() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": request_body, "more_body": False}
            return {"type": "http.disconnect"}

        status = 500
        response_body = b""

        async def send_wrapper(message: Message) -> None:
            nonlocal status, response_body
            if message["type"] == "http.response.start":
                status = message["status"]
            elif message["type"] == "http.response.body" and len(response_body) < MAX_BODY_LOG:
                response_body += message.get("body", b"")
            await send(message)

        try:
            await self.app(scope, replay_receive, send_wrapper)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "%s %s failed after %.1f ms | req=%s",
                scope["method"],
                scope["path"],
                elapsed_ms,
                _preview(request_body),
            )
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s -> %d (%.1f ms) | req=%s | resp=%s",
            scope["method"],
            scope["path"],
            status,
            elapsed_ms,
            _preview(request_body),
            _preview(response_body),
        )
