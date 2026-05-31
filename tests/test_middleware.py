"""Tests for PeekrASGIMiddleware."""
import asyncio
import pytest
from peekr.middleware import PeekrASGIMiddleware
from peekr.exporters import _exporters


def _captured_spans():
    spans = []

    class Cap:
        def export(self, s):
            spans.append(s)

    _exporters.clear()
    _exporters.append(Cap())
    return spans


# Minimal ASGI app helpers
async def _simple_app(scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok", "more_body": False})


async def _error_app(scope, receive, send):
    await send({"type": "http.response.start", "status": 500, "headers": []})
    await send({"type": "http.response.body", "body": b"err", "more_body": False})


async def _streaming_app(scope, receive, send):
    await send({"type": "http.response.start", "status": 200, "headers": []})
    for chunk in [b"chunk1", b"chunk2", b"chunk3"]:
        await send({"type": "http.response.body", "body": chunk, "more_body": True})
    await send({"type": "http.response.body", "body": b"", "more_body": False})


def _scope(path="/test", method="GET", headers=None):
    return {
        "type": "http",
        "path": path,
        "method": method,
        "headers": headers or [],
    }


async def _run(app_fn, scope):
    async def receive():
        return {}

    messages = []

    async def send(msg):
        messages.append(msg)

    await app_fn(scope, receive, send)
    return messages


class TestPeekrASGIMiddleware:

    def setup_method(self):
        _exporters.clear()

    def teardown_method(self):
        _exporters.clear()

    def test_creates_root_span(self):
        spans = _captured_spans()
        middleware = PeekrASGIMiddleware(_simple_app)
        asyncio.run(_run(middleware, _scope("/v1/answer", "POST")))
        assert len(spans) == 1
        s = spans[0]
        assert s.name == "POST /v1/answer"
        assert s.attributes["http.method"] == "POST"
        assert s.attributes["http.path"] == "/v1/answer"
        assert s.status == "ok"

    def test_records_status_code(self):
        spans = _captured_spans()
        middleware = PeekrASGIMiddleware(_simple_app)
        asyncio.run(_run(middleware, _scope()))
        assert spans[0].attributes["http.status_code"] == 200

    def test_error_status(self):
        spans = _captured_spans()
        middleware = PeekrASGIMiddleware(_error_app)
        asyncio.run(_run(middleware, _scope()))
        assert spans[0].status == "error"
        assert spans[0].attributes["http.status_code"] == 500

    def test_extracts_tenant_header(self):
        spans = _captured_spans()
        middleware = PeekrASGIMiddleware(_simple_app)
        headers = [(b"x-tenant-id", b"acme-corp")]
        asyncio.run(_run(middleware, _scope(headers=headers)))
        assert spans[0].attributes["tenant_id"] == "acme-corp"

    def test_extracts_user_header(self):
        spans = _captured_spans()
        middleware = PeekrASGIMiddleware(_simple_app)
        headers = [(b"x-user-id", b"user_42")]
        asyncio.run(_run(middleware, _scope(headers=headers)))
        assert spans[0].attributes["user_id"] == "user_42"

    def test_skip_healthz(self):
        spans = _captured_spans()
        middleware = PeekrASGIMiddleware(_simple_app)
        asyncio.run(_run(middleware, _scope("/healthz")))
        assert len(spans) == 0

    def test_custom_skip_paths(self):
        spans = _captured_spans()
        middleware = PeekrASGIMiddleware(_simple_app, skip_paths={"/custom-skip"})
        asyncio.run(_run(middleware, _scope("/custom-skip")))
        assert len(spans) == 0

    def test_streaming_span_closes_at_end(self):
        spans = _captured_spans()
        middleware = PeekrASGIMiddleware(_streaming_app)
        asyncio.run(_run(middleware, _scope("/v1/answer", "POST")))
        # Span must be exported (once, after final chunk)
        assert len(spans) == 1
        assert spans[0].status == "ok"

    def test_non_http_passthrough(self):
        spans = _captured_spans()
        middleware = PeekrASGIMiddleware(_simple_app)
        # WebSocket scope — should pass through without creating a span
        ws_scope = {"type": "websocket", "path": "/ws"}
        # Won't raise even without a proper websocket app
        try:
            asyncio.run(_run(middleware, ws_scope))
        except Exception:
            pass
        assert len(spans) == 0

    def test_fastapi_middleware_alias(self):
        from peekr import FastAPIMiddleware
        from peekr.middleware import PeekrASGIMiddleware
        assert FastAPIMiddleware is PeekrASGIMiddleware
