from __future__ import annotations

import sys
from collections import defaultdict, deque
from typing import Any, Optional

from .span import Span


class _BaseAlert:
    """Base class for all alert types.

    Subclass and override ``on_trigger`` to customise what happens when the
    alert fires (e.g. send a Slack message, raise an exception, …).
    """

    def on_trigger(self, message: str) -> None:
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

    def __init__(self, threshold: float = 0.05, window: int = 100) -> None:
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

    def __init__(self, multiplier: float = 2.0, window: int = 100) -> None:
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

    def __init__(self, ms: float = 5000.0) -> None:
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

    def __init__(self, runs: int = 5) -> None:
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
class alert:
    ErrorRate = ErrorRate
    CostSpike = CostSpike
    LatencyP95 = LatencyP95
    TokenGrowth = TokenGrowth
