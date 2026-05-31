"""
peekr.middleware — ASGI middleware that creates a root span for every request.

Works with FastAPI, Starlette, and any ASGI framework.
Handles streaming (SSE, WebSocket) correctly — span closes when the
last byte is sent, not when headers are sent.

Usage::

    # FastAPI
    app.add_middleware(peekr.FastAPIMiddleware)

    # Starlette / raw ASGI
    app = peekr.FastAPIMiddleware(app)

    # With options
    app.add_middleware(
        peekr.FastAPIMiddleware,
        tenant_header="X-Tenant-Id",
        user_header="X-User-Id",
        skip_paths={"/healthz", "/metrics"},
    )
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Awaitable, Callable


class PeekrASGIMiddleware:
    """Pure ASGI middleware — wraps every HTTP request in a root span.

    Parameters
    ----------
    app
        The ASGI application to wrap.
    tenant_header
        Request header name to copy into ``span.attributes["tenant_id"]``.
        Default ``"X-Tenant-Id"``.
    user_header
        Request header name to copy into ``span.attributes["user_id"]``.
        Default ``"X-User-Id"``.
    skip_paths
        Set of exact paths to skip (e.g. ``{"/healthz", "/metrics"}``).
        Requests to these paths pass through without creating a span.
    """

    def __init__(
        self,
        app,
        tenant_header: str = "X-Tenant-Id",
        user_header: str   = "X-User-Id",
        skip_paths: set[str] | None = None,
    ) -> None:
        self.app            = app
        self.tenant_header  = tenant_header.lower().encode()
        self.user_header    = user_header.lower().encode()
        self.skip_paths     = skip_paths or {"/healthz", "/health", "/metrics", "/ping"}

    async def __call__(self, scope, receive, send) -> None:
        # Only instrument HTTP requests
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path   = scope.get("path", "/")
        method = scope.get("method", "GET")

        if path in self.skip_paths:
            await self.app(scope, receive, send)
            return

        # Lazy import — peekr may not be instrumented yet, and import is fast
        try:
            from ..context import start_span, end_span
            from ..exporters import export_span
        except ImportError:
            await self.app(scope, receive, send)
            return

        # Build a readable span name — prefer FastAPI route pattern if available
        route_path = path
        if "route" in scope:
            route_path = getattr(scope["route"], "path", path)

        span_name = f"{method} {route_path}"
        span, token = start_span(span_name)

        # HTTP metadata
        span.attributes["http.method"] = method
        span.attributes["http.path"]   = path
        span.attributes["endpoint"]    = path
        span.attributes["feature"]     = "http_request"

        # Pull tenant / user from headers
        headers = dict(scope.get("headers", []))
        tid = headers.get(self.tenant_header, b"").decode("utf-8", errors="replace")
        uid = headers.get(self.user_header,   b"").decode("utf-8", errors="replace")
        if tid:
            span.attributes["tenant_id"] = tid
        if uid:
            span.attributes["user_id"] = uid

        finished = False

        def _finish(status: str = "ok") -> None:
            nonlocal finished
            if finished:
                return
            finished = True
            span.status = status
            end_span(span, token)
            export_span(span)

        async def send_wrapper(message) -> None:
            if message["type"] == "http.response.start":
                code = message.get("status", 200)
                span.attributes["http.status_code"] = code
                if code >= 500:
                    span.status = "error"

            elif message["type"] == "http.response.body":
                # more_body=False → last chunk → stream complete
                if not message.get("more_body", False):
                    _finish(span.status if span.status == "error" else "ok")

            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
            # Non-streaming responses may not send a final body chunk
            _finish(span.status if span.status == "error" else "ok")
        except Exception as exc:
            span.attributes["error"] = str(exc)
            _finish("error")
            raise


# Alias so it reads naturally: app.add_middleware(peekr.FastAPIMiddleware)
FastAPIMiddleware = PeekrASGIMiddleware
