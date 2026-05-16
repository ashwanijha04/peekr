from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from collections import defaultdict, deque
from typing import Any, Callable, Optional

from .span import Span


# ---------------------------------------------------------------------------
# Sinks
# ---------------------------------------------------------------------------

class _BaseSink:
    """Where an alert is delivered when it fires.

    Subclass and override ``notify(alert_name, message)`` to ship the alert
    somewhere. Must never raise out of ``notify`` — sinks live on the trace
    write path and a flaky sink should not break the application.
    """

    def notify(self, alert_name: str, message: str) -> None:
        raise NotImplementedError


class StderrSink(_BaseSink):
    """Default sink — writes ``[peekr alert] <message>`` to stderr."""

    def notify(self, alert_name: str, message: str) -> None:
        print(f"[peekr alert] {message}", file=sys.stderr)


class SlackSink(_BaseSink):
    """Post the alert message to a Slack incoming-webhook URL.

    Get a webhook URL: https://api.slack.com/messaging/webhooks

    Parameters
    ----------
    webhook_url:
        Full incoming-webhook URL (https://hooks.slack.com/services/…).
    timeout_seconds:
        HTTP timeout. Default 5s. Failures are swallowed silently.
    """

    def __init__(self, webhook_url: str, *, timeout_seconds: float = 5.0) -> None:
        if not webhook_url:
            raise ValueError("SlackSink: webhook_url is required")
        self.webhook_url = webhook_url
        self.timeout_seconds = timeout_seconds

    def notify(self, alert_name: str, message: str) -> None:
        body = json.dumps({"text": f":rotating_light: *{alert_name}* — {message}"}).encode("utf-8")
        req = urllib.request.Request(
            self.webhook_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=self.timeout_seconds).close()
        except (urllib.error.URLError, OSError, TimeoutError):
            pass  # sinks must never break the alert path


class WebhookSink(_BaseSink):
    """POST a JSON payload to an arbitrary HTTPS endpoint.

    Body shape::

        {"alert": "<class name>", "message": "<human-readable>"}

    Pass ``payload_builder`` to override the body shape (e.g. for PagerDuty
    Events API v2, Opsgenie, custom incident systems).

    Parameters
    ----------
    url:
        Endpoint to POST to.
    headers:
        Optional headers (e.g. auth tokens).
    payload_builder:
        Optional callable ``(alert_name, message) -> dict``. Defaults to
        ``{"alert": alert_name, "message": message}``.
    timeout_seconds:
        HTTP timeout. Default 5s. Failures are swallowed silently.
    """

    def __init__(
        self,
        url: str,
        *,
        headers: Optional[dict] = None,
        payload_builder: Optional[Callable[[str, str], dict]] = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        if not url:
            raise ValueError("WebhookSink: url is required")
        self.url = url
        self.headers = dict(headers or {})
        self.headers.setdefault("Content-Type", "application/json")
        self.payload_builder = payload_builder
        self.timeout_seconds = timeout_seconds

    def notify(self, alert_name: str, message: str) -> None:
        if self.payload_builder is not None:
            payload = self.payload_builder(alert_name, message)
        else:
            payload = {"alert": alert_name, "message": message}
        body = json.dumps(payload, default=str).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=body,
            headers=self.headers,
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=self.timeout_seconds).close()
        except (urllib.error.URLError, OSError, TimeoutError):
            pass


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

class _BaseAlert:
    """Base class for all alert types.

    Override ``on_trigger`` for one-off custom behaviour, or pass ``sinks=[…]``
    (or call ``.with_sinks(…)``) to ship the alert through one or more sinks.
    Default behaviour with no sinks is to write to stderr.
    """

    def __init__(self, *, sinks: Optional[list[_BaseSink]] = None) -> None:
        self.sinks: list[_BaseSink] = list(sinks) if sinks else []

    def with_sinks(self, *sinks: _BaseSink) -> "_BaseAlert":
        """Attach one or more sinks; returns self so it chains in a list."""
        self.sinks.extend(sinks)
        return self

    def on_trigger(self, message: str) -> None:
        if self.sinks:
            for sink in self.sinks:
                try:
                    sink.notify(self.__class__.__name__, message)
                except Exception:
                    pass  # one bad sink must not break others
        else:
            print(f"[peekr alert] {message}", file=sys.stderr)

    def check(self, trace_spans: list[dict[str, Any]]) -> None:
        raise NotImplementedError


class ErrorRate(_BaseAlert):
    """Fire when the error-rate in the last *window* traces exceeds *threshold*.

    Parameters
    ----------
    threshold:
        Fraction (0–1) of traces that must be errors before firing.
        Default 0.05 (5 %).
    window:
        Number of recent traces kept in the rolling window.
        Default 100.
    """

    def __init__(
        self,
        threshold: float = 0.05,
        window: int = 100,
        *,
        sinks: Optional[list[_BaseSink]] = None,
    ) -> None:
        super().__init__(sinks=sinks)
        self.threshold = threshold
        self.window = window
        self._history: deque[bool] = deque(maxlen=window)

    def check(self, trace_spans: list[dict[str, Any]]) -> None:
        has_error = any(s.get("status") == "error" for s in trace_spans)
        self._history.append(has_error)
        if not self._history:
            return
        rate = sum(self._history) / len(self._history)
        if rate > self.threshold:
            pct = rate * 100
            self.on_trigger(
                f"ErrorRate: {pct:.1f}% of last {len(self._history)} traces errored "
                f"(threshold {self.threshold * 100:.1f}%)"
            )


class CostSpike(_BaseAlert):
    """Fire when total tokens in the current trace exceed *multiplier* × rolling average.

    Parameters
    ----------
    multiplier:
        How many times larger than the rolling average the current trace must
        be to trigger. Default 2.0.
    window:
        Number of recent token counts kept in the rolling window.
        Default 100.
    """

    def __init__(
        self,
        multiplier: float = 2.0,
        window: int = 100,
        *,
        sinks: Optional[list[_BaseSink]] = None,
    ) -> None:
        super().__init__(sinks=sinks)
        self.multiplier = multiplier
        self.window = window
        self._history: deque[float] = deque(maxlen=window)

    def check(self, trace_spans: list[dict[str, Any]]) -> None:
        tokens = sum(
            s.get("attributes", {}).get("tokens_total", 0) or 0
            for s in trace_spans
        )
        if self._history:
            avg = sum(self._history) / len(self._history)
            if avg > 0 and tokens > self.multiplier * avg:
                self.on_trigger(
                    f"CostSpike: current trace used {tokens:.0f} tokens "
                    f"({tokens / avg:.1f}× rolling avg of {avg:.0f}), "
                    f"threshold {self.multiplier}×"
                )
        self._history.append(tokens)


class LatencyP95(_BaseAlert):
    """Fire when the P95 span duration inside the current trace exceeds *ms* milliseconds.

    Parameters
    ----------
    ms:
        Latency threshold in milliseconds. Default 5000.
    """

    def __init__(
        self,
        ms: float = 5000.0,
        *,
        sinks: Optional[list[_BaseSink]] = None,
    ) -> None:
        super().__init__(sinks=sinks)
        self.ms = ms

    def check(self, trace_spans: list[dict[str, Any]]) -> None:
        durations = sorted(
            s["duration_ms"]
            for s in trace_spans
            if s.get("duration_ms") is not None
        )
        if not durations:
            return
        # P95 index (nearest-rank method: ceil(n * 0.95) - 1, clamped)
        import math
        idx = min(len(durations) - 1, math.ceil(len(durations) * 0.95) - 1)
        idx = max(0, idx)
        p95 = durations[idx]
        if p95 > self.ms:
            self.on_trigger(
                f"LatencyP95: p95 span duration is {p95:.1f}ms "
                f"(threshold {self.ms:.0f}ms)"
            )


class TokenGrowth(_BaseAlert):
    """Fire when total token usage has grown monotonically for *runs* consecutive traces.

    Parameters
    ----------
    runs:
        Number of consecutive traces with growing token usage needed to fire.
        Default 5.
    """

    def __init__(
        self,
        runs: int = 5,
        *,
        sinks: Optional[list[_BaseSink]] = None,
    ) -> None:
        super().__init__(sinks=sinks)
        self.runs = runs
        self._history: deque[float] = deque(maxlen=runs + 1)

    def check(self, trace_spans: list[dict[str, Any]]) -> None:
        tokens = sum(
            s.get("attributes", {}).get("tokens_total", 0) or 0
            for s in trace_spans
        )
        self._history.append(tokens)
        if len(self._history) < self.runs + 1:
            return
        values = list(self._history)
        if all(values[i] < values[i + 1] for i in range(len(values) - 1)):
            self.on_trigger(
                f"TokenGrowth: token usage has grown for {self.runs} consecutive "
                f"traces (latest {tokens:.0f} tokens)"
            )


class AlertExporter:
    """Exporter that accumulates spans per trace and runs alerts when a root span arrives.

    Pass an instance to ``peekr.instrument(exporter=AlertExporter([...]))``, or
    register it alongside other exporters via ``peekr.exporters.add_exporter``.

    Parameters
    ----------
    alerts:
        List of alert instances (``ErrorRate``, ``CostSpike``, etc.).
    """

    def __init__(self, alerts: list[_BaseAlert]) -> None:
        self.alerts = alerts
        # trace_id → list of span dicts accumulated so far
        self._pending: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def export(self, span: Span) -> None:
        d = span.to_dict()
        self._pending[span.trace_id].append(d)

        # Root span signals the end of a complete trace
        if span.parent_id is None:
            trace_spans = self._pending.pop(span.trace_id, [])
            for alert in self.alerts:
                alert.check(trace_spans)


# Convenience namespace so users can write ``peekr.alert.ErrorRate(...)``
# and ``peekr.alert.SlackSink(...)``.
class alert:
    ErrorRate = ErrorRate
    CostSpike = CostSpike
    LatencyP95 = LatencyP95
    TokenGrowth = TokenGrowth
    SlackSink = SlackSink
    WebhookSink = WebhookSink
    StderrSink = StderrSink
