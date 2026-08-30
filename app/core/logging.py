"""Request access logging for every API call (pure ASGI, no BaseHTTPMiddleware).

Logs method, path, status, duration, and request/response bodies. Only text/JSON
bodies are logged (truncated); binary bodies (e.g. a PDF upload) are shown as their
content-type so the log stays readable.
"""

from __future__ import annotations

import logging
import time

from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger("rerouteher.request")

# Bodies are truncated to this many bytes in the log.
MAX_BODY_LOG = 4000


def configure_logging(level: int = logging.INFO) -> None:
    """Set a log format and level on the root logger so all app logs reach stdout."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=level
    )


def _loggable(content_type: str) -> bool:
    ct = (content_type or "").lower()
    return ct.startswith("application/json") or ct.startswith("text/")


def _content_type(headers: list[tuple[bytes, bytes]]) -> str:
    for key, value in headers or []:
        if key.lower() == b"content-type":
            return value.decode("latin-1")
    return ""


def _preview(body: bytes) -> str:
    if not body:
        return "-"
    text = body[:MAX_BODY_LOG].decode("utf-8", "replace")
    return text + "...(truncated)" if len(body) > MAX_BODY_LOG else text


class RequestLoggingMiddleware:
    """Log method, path, status, duration, and text/JSON bodies (binary bodies elided)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        req_ct = _content_type(scope.get("headers", []))
        log_req = _loggable(req_ct)

        # Only buffer+replay the request body when it is text/JSON; binary uploads
        # (e.g. a PDF) are passed straight through and shown as their content-type.
        request_body = b""
        receive_to_use = receive
        if log_req:
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

            receive_to_use = replay_receive

        status = 500
        resp_ct = ""
        response_body = b""

        async def send_wrapper(message: Message) -> None:
            nonlocal status, resp_ct, response_body
            if message["type"] == "http.response.start":
                status = message["status"]
                resp_ct = _content_type(message.get("headers", []))
            elif (
                message["type"] == "http.response.body"
                and _loggable(resp_ct)
                and len(response_body) < MAX_BODY_LOG
            ):
                response_body += message.get("body", b"")
            await send(message)

        try:
            await self.app(scope, receive_to_use, send_wrapper)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "%s %s failed after %.1f ms", scope["method"], scope["path"], elapsed_ms
            )
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000
        req_repr = _preview(request_body) if log_req else f"<{req_ct or 'binary'}>"
        resp_repr = _preview(response_body) if _loggable(resp_ct) else f"<{resp_ct or 'binary'}>"
        logger.info(
            "%s %s -> %d (%.1f ms) | req=%s | resp=%s",
            scope["method"],
            scope["path"],
            status,
            elapsed_ms,
            req_repr,
            resp_repr,
        )
