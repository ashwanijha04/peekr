"""Behavioural tests for HTTPExporter (Peekr Cloud client).

These mock urllib so no real network is hit. Constructor/signature tests live
in test_tenant_schema.py — this file covers batching, retries, and the
on-wire payload shape.
"""

import io
import json
import time
from unittest.mock import patch, MagicMock

import urllib.error

from peekr.exporters import HTTPExporter
from peekr.span import Span


def _make_response(status: int = 200, body: bytes = b'{"accepted":1}') -> MagicMock:
    """Build a MagicMock that satisfies the urlopen context-manager protocol."""
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = body
    resp.__enter__ = lambda self: self
    resp.__exit__ = lambda self, *a: None
    return resp


def _finished_span(name: str = "op") -> Span:
    s = Span(name=name, trace_id="t-1")
    s.finish()
    return s


class TestHTTPExporter:
    def test_export_buffers_until_batch_size(self):
        """Spans below batch_size shouldn't trigger a network call."""
        exp = HTTPExporter(
            endpoint="https://x", api_key="k", batch_size=10, flush_interval_seconds=60
        )
        with patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value = _make_response()
            for _ in range(3):
                exp.export(_finished_span())
            assert urlopen.call_count == 0
        exp.shutdown()

    def test_export_flushes_at_batch_size(self):
        """Hitting batch_size triggers an immediate POST."""
        exp = HTTPExporter(
            endpoint="https://x", api_key="k", batch_size=3, flush_interval_seconds=60
        )
        with patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value = _make_response()
            for _ in range(3):
                exp.export(_finished_span())
            assert urlopen.call_count == 1
        exp.shutdown()

    def test_shutdown_flushes_remaining_spans(self):
        """Sub-batch buffer must drain at shutdown so we don't lose spans on exit."""
        exp = HTTPExporter(
            endpoint="https://x", api_key="k", batch_size=100, flush_interval_seconds=60
        )
        with patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value = _make_response()
            exp.export(_finished_span())
            exp.export(_finished_span())
            exp.shutdown()
            assert urlopen.call_count == 1

    def test_post_url_headers_and_body(self):
        """Wire format: POST {endpoint}/v1/spans, Bearer auth, JSON body."""
        exp = HTTPExporter(
            endpoint="https://ingest.peekr.cloud/", api_key="pk_live_x", batch_size=1
        )
        with patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value = _make_response()
            exp.export(_finished_span("hello"))

        req = urlopen.call_args.args[0]
        assert req.full_url == "https://ingest.peekr.cloud/v1/spans"
        assert req.get_method() == "POST"
        assert req.headers["Authorization"] == "Bearer pk_live_x"
        assert req.headers["Content-type"] == "application/json"

        payload = json.loads(req.data.decode("utf-8"))
        assert "spans" in payload and len(payload["spans"]) == 1
        assert payload["spans"][0]["name"] == "hello"
        assert payload["spans"][0]["trace_id"] == "t-1"
        # duration_ms is computed from the Span dataclass — must be on the wire.
        assert "duration_ms" in payload["spans"][0]
        exp.shutdown()

    def test_retries_once_on_5xx_then_drops(self):
        """One retry after a 1s sleep, then log and drop."""
        exp = HTTPExporter(endpoint="https://x", api_key="k", batch_size=1)
        err = urllib.error.HTTPError(
            url="https://x/v1/spans",
            code=503,
            msg="bad",
            hdrs=None,
            fp=io.BytesIO(b""),
        )
        with (
            patch("urllib.request.urlopen", side_effect=err) as urlopen,
            patch("time.sleep") as sleep,
        ):
            exp.export(_finished_span())
            assert urlopen.call_count == 2  # initial + 1 retry
            sleep.assert_called_once_with(1.0)
        exp.shutdown()

    def test_does_not_retry_on_401(self):
        """4xx (auth/validation) shouldn't be retried — fail fast."""
        exp = HTTPExporter(endpoint="https://x", api_key="bad", batch_size=1)
        err = urllib.error.HTTPError(
            url="https://x/v1/spans",
            code=401,
            msg="invalid",
            hdrs=None,
            fp=io.BytesIO(b""),
        )
        with (
            patch("urllib.request.urlopen", side_effect=err) as urlopen,
            patch("time.sleep") as sleep,
        ):
            exp.export(_finished_span())
            assert urlopen.call_count == 1
            sleep.assert_not_called()
        exp.shutdown()

    def test_network_error_retries_once(self):
        with (
            patch(
                "urllib.request.urlopen", side_effect=urllib.error.URLError("boom")
            ) as urlopen,
            patch("time.sleep") as sleep,
        ):
            exp = HTTPExporter(endpoint="https://x", api_key="k", batch_size=1)
            exp.export(_finished_span())
            assert urlopen.call_count == 2
            sleep.assert_called_once_with(1.0)
        exp.shutdown()

    def test_background_flush_on_interval(self):
        """Buffer below batch_size still flushes on the interval timer."""
        exp = HTTPExporter(
            endpoint="https://x",
            api_key="k",
            batch_size=100,
            flush_interval_seconds=0.05,
        )
        with patch("urllib.request.urlopen") as urlopen:
            urlopen.return_value = _make_response()
            exp.export(_finished_span())
            # Wait long enough for at least one tick of the flusher thread.
            time.sleep(0.2)
            assert urlopen.call_count >= 1
        exp.shutdown()
