"""Tests for SlackSink and WebhookSink + the sink-routing machinery.

Sinks live on the trace write path, so two invariants matter most:
  1. They must reach the destination with the right payload shape.
  2. They must NEVER raise out of `.notify()` — a flaky webhook should not
     break the application's tracing.
"""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from peekr.alerts import (
    ErrorRate,
    CostSpike,
    SlackSink,
    WebhookSink,
    StderrSink,
    _BaseSink,
    alert,
)


# ---------------------------------------------------------------------------
# Tiny capturing HTTP server for end-to-end tests
# ---------------------------------------------------------------------------


class _CapturingHandler(BaseHTTPRequestHandler):
    received: list[tuple[str, dict, str]] = []  # (path, headers, body)

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        type(self).received.append((self.path, dict(self.headers), body))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *_):  # silence default logging
        pass


@pytest.fixture
def http_capture():
    """Spin up a one-shot HTTP server on a random localhost port."""
    _CapturingHandler.received = []
    server = HTTPServer(("127.0.0.1", 0), _CapturingHandler)
    host, port = server.server_address
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://{host}:{port}", _CapturingHandler.received
    server.shutdown()


# ---------------------------------------------------------------------------
# 1. Constructor validation
# ---------------------------------------------------------------------------


class TestConstructors:
    def test_slack_requires_url(self):
        with pytest.raises(ValueError):
            SlackSink(webhook_url="")

    def test_webhook_requires_url(self):
        with pytest.raises(ValueError):
            WebhookSink(url="")

    def test_webhook_sets_default_content_type(self):
        s = WebhookSink(url="https://x", headers={"Authorization": "Bearer t"})
        assert s.headers["Content-Type"] == "application/json"
        assert s.headers["Authorization"] == "Bearer t"

    def test_namespace_exposes_sinks(self):
        assert alert.SlackSink is SlackSink
        assert alert.WebhookSink is WebhookSink
        assert alert.StderrSink is StderrSink


# ---------------------------------------------------------------------------
# 2. Payload shape — end-to-end against a real local HTTP server
# ---------------------------------------------------------------------------


class TestSlackPayload:
    def test_slack_posts_text_payload(self, http_capture):
        url, received = http_capture
        SlackSink(webhook_url=url).notify("ErrorRate", "5.2% errored")

        assert len(received) == 1
        path, headers, body = received[0]
        assert headers.get("Content-Type") == "application/json"
        payload = json.loads(body)
        assert "ErrorRate" in payload["text"]
        assert "5.2% errored" in payload["text"]


class TestWebhookPayload:
    def test_webhook_default_payload(self, http_capture):
        url, received = http_capture
        WebhookSink(url=url).notify("CostSpike", "trace used 5000 tokens")

        assert len(received) == 1
        _, _, body = received[0]
        payload = json.loads(body)
        assert payload == {"alert": "CostSpike", "message": "trace used 5000 tokens"}

    def test_webhook_custom_payload_builder(self, http_capture):
        url, received = http_capture

        def pagerduty_shape(name, msg):
            return {
                "routing_key": "test-key",
                "event_action": "trigger",
                "payload": {
                    "summary": msg,
                    "source": "peekr",
                    "severity": "warning",
                    "custom_details": {"alert": name},
                },
            }

        WebhookSink(url=url, payload_builder=pagerduty_shape).notify(
            "LatencyP95", "p95 9876ms"
        )

        _, _, body = received[0]
        payload = json.loads(body)
        assert payload["routing_key"] == "test-key"
        assert payload["payload"]["summary"] == "p95 9876ms"
        assert payload["payload"]["custom_details"]["alert"] == "LatencyP95"

    def test_webhook_sends_custom_headers(self, http_capture):
        url, received = http_capture
        WebhookSink(
            url=url,
            headers={"X-API-Key": "secret-token", "X-Source": "peekr"},
        ).notify("ErrorRate", "msg")

        _, headers, _ = received[0]
        assert headers.get("X-Api-Key") == "secret-token"  # http.server lowercases
        assert headers.get("X-Source") == "peekr"


# ---------------------------------------------------------------------------
# 3. Resilience — flaky sinks must never raise out of notify()
# ---------------------------------------------------------------------------


class TestResilience:
    def test_slack_silently_swallows_network_error(self):
        """SlackSink against an unreachable URL must not raise."""
        # Reserve a port (don't listen) to guarantee connection refused
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        _, port = s.getsockname()
        s.close()  # released; nothing listening now
        sink = SlackSink(webhook_url=f"http://127.0.0.1:{port}", timeout_seconds=1.0)
        # If this raises, the test fails
        sink.notify("ErrorRate", "should not crash")

    def test_webhook_silently_swallows_timeout(self):
        """WebhookSink with an unresolvable host must not raise."""
        sink = WebhookSink(
            url="http://this-domain-does-not-resolve.invalid:80/x",
            timeout_seconds=1.0,
        )
        sink.notify("CostSpike", "should not crash")

    def test_one_bad_sink_does_not_break_other_sinks(self):
        """If one sink raises (it shouldn't, but if it does), the others still fire."""

        class _Boom(_BaseSink):
            def notify(self, alert_name, message):
                raise RuntimeError("network exploded")

        seen: list[tuple[str, str]] = []

        class _Capture(_BaseSink):
            def notify(self, alert_name, message):
                seen.append((alert_name, message))

        a = ErrorRate(threshold=0.0, window=1, sinks=[_Boom(), _Capture()])
        # Force the alert to trigger
        a.check([{"status": "error"}])
        assert seen == [("ErrorRate", a._history and seen[0][1])] or len(seen) == 1
        assert seen[0][0] == "ErrorRate"
        assert "ErrorRate" in seen[0][1]


# ---------------------------------------------------------------------------
# 4. Alert-to-sink wiring
# ---------------------------------------------------------------------------


class TestAlertWiring:
    def test_alert_with_no_sinks_writes_to_stderr(self, capsys):
        a = ErrorRate(threshold=0.0, window=1)
        a.check([{"status": "error"}])
        out = capsys.readouterr()
        assert "[peekr alert]" in out.err
        assert "ErrorRate" in out.err

    def test_sinks_via_kwarg(self):
        seen: list[str] = []

        class _Capture(_BaseSink):
            def notify(self, alert_name, message):
                seen.append(f"{alert_name}: {message}")

        a = ErrorRate(threshold=0.0, window=1, sinks=[_Capture()])
        a.check([{"status": "error"}])
        assert len(seen) == 1
        assert seen[0].startswith("ErrorRate:")

    def test_with_sinks_chainable(self):
        seen: list[str] = []

        class _Capture(_BaseSink):
            def notify(self, alert_name, message):
                seen.append(alert_name)

        a = CostSpike(multiplier=1.5, window=2).with_sinks(_Capture(), _Capture())
        # Seed history then trigger
        a.check([{"attributes": {"tokens_total": 100}}])
        a.check([{"attributes": {"tokens_total": 100}}])
        a.check([{"attributes": {"tokens_total": 500}}])  # 5× rolling avg → fires
        assert seen.count("CostSpike") == 2

    def test_chain_appends_not_overwrites(self):
        """with_sinks() must extend, not replace, an already-set sinks list."""

        class _A(_BaseSink):
            def notify(self, *_):
                pass

        class _B(_BaseSink):
            def notify(self, *_):
                pass

        s = _A()
        t = _B()
        a = ErrorRate(sinks=[s]).with_sinks(t)
        assert a.sinks == [s, t]
