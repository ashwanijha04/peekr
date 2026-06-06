"""budget.py — Rolling cost tracking with configurable warning and hard-limit callbacks.

BudgetAlert is a first-class exporter: pass it directly to add_exporter() or
include it in the alerts=[] list passed to AlertExporter / peekr.instrument().

When used via alerts=[], AlertExporter calls check(trace_spans: list[dict]).
When used via add_exporter() or instrument(exporter=...), export(span) is called
directly with a Span object.  Both paths are supported.

Usage::

    import peekr
    from peekr.budget import BudgetAlert, BudgetExceededError

    peekr.instrument(
        exporter=BudgetAlert(
            limit_usd=5.0,
            warning_pct=0.8,
            window="day",
        )
    )
    # or as an alert alongside others:
    from peekr.alerts import AlertExporter, ErrorRate
    peekr.exporters.add_exporter(
        AlertExporter([
            ErrorRate(threshold=0.05),
            BudgetAlert(limit_usd=5.0),
        ])
    )
"""

from __future__ import annotations

import threading
import time
import warnings
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BudgetExceededError(Exception):
    """Raised (and optionally a callback is fired) when rolling spend >= limit_usd."""


# ---------------------------------------------------------------------------
# Pricing table  (input $/1M tokens, output $/1M tokens)
# ---------------------------------------------------------------------------

_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-4": (30.0, 60.0),
    "gpt-3.5": (0.50, 1.50),
    "claude-3-5-sonnet": (3.0, 15.0),
    "claude-3-opus": (15.0, 75.0),
    "claude-3-haiku": (0.25, 1.25),
    "claude-sonnet-4": (3.0, 15.0),
    "gemini-1.5-pro": (1.25, 5.0),
    "gemini-1.5-flash": (0.075, 0.30),
}

# Window durations in seconds
_WINDOW_SECONDS: dict[str, float] = {
    "hour": 3_600.0,
    "day": 86_400.0,
    "month": 2_592_000.0,  # 30 days
}


# ---------------------------------------------------------------------------
# BudgetAlert
# ---------------------------------------------------------------------------


class BudgetAlert:
    """Rolling-cost budget guard that fires callbacks at warning and hard-limit.

    Implements both the *exporter* protocol (``export(span)``) and the
    *alert* protocol (``check(trace_spans)``), so it can be used in two ways:

    **As a direct exporter** (recommended — every span is checked individually)::

        peekr.add_exporter(BudgetAlert(limit_usd=5.0))
        # or
        peekr.instrument(exporter=BudgetAlert(limit_usd=5.0))

    **Inside an AlertExporter alerts list** (checked per-trace on root span)::

        from peekr.alerts import AlertExporter
        peekr.add_exporter(AlertExporter([BudgetAlert(limit_usd=5.0)]))

    Parameters
    ----------
    limit_usd:
        Hard spending cap in USD for the current window.
    on_warning:
        Optional callable ``(spent: float, limit: float) -> None`` fired once
        when spending crosses ``warning_pct * limit_usd``.  If omitted a
        ``warnings.warn`` is issued instead.
    on_exceeded:
        Optional callable ``(spent: float, limit: float) -> None`` fired when
        spending reaches or exceeds ``limit_usd``.  After the callback,
        ``BudgetExceededError`` is raised regardless.
    warning_pct:
        Fraction (0–1) of the limit at which the warning fires.  Default 0.8.
    window:
        Rolling time window: ``"hour"``, ``"day"`` (default), or ``"month"``.
    """

    def __init__(
        self,
        limit_usd: float,
        on_warning: Optional[Callable[[float, float], None]] = None,
        on_exceeded: Optional[Callable[[float, float], None]] = None,
        warning_pct: float = 0.8,
        window: str = "day",
    ) -> None:
        if limit_usd <= 0:
            raise ValueError("BudgetAlert: limit_usd must be > 0")
        if window not in _WINDOW_SECONDS:
            raise ValueError(
                f"BudgetAlert: window must be one of {list(_WINDOW_SECONDS)}"
            )
        if not (0.0 < warning_pct < 1.0):
            raise ValueError(
                "BudgetAlert: warning_pct must be between 0 and 1 (exclusive)"
            )

        self.limit_usd = limit_usd
        self.on_warning = on_warning
        self.on_exceeded = on_exceeded
        self.warning_pct = warning_pct
        self.window = window

        self._spent: float = 0.0
        self._warned: bool = False
        self._lock = threading.Lock()
        self._window_start: float = time.time()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cost_from_attributes(
        model: str,
        tok_in: int,
        tok_out: int,
    ) -> float:
        """Return the estimated USD cost for the given model and token counts."""
        ip, op = 1.0, 1.0  # default fallback: $1/1M each
        model_lower = model.lower()
        for prefix, pricing in _PRICING.items():
            if prefix in model_lower:
                ip, op = pricing
                break
        return (tok_in * ip + tok_out * op) / 1_000_000

    def _cost_from_span(self, span: Any) -> float:
        """Extract cost from a Span object (has .attributes dict)."""
        attrs = getattr(span, "attributes", {}) or {}
        model = attrs.get("model", "") or ""
        tok_in = int(attrs.get("tokens_input", 0) or 0)
        tok_out = int(attrs.get("tokens_output", 0) or 0)
        return self._cost_from_attributes(model, tok_in, tok_out)

    def _cost_from_span_dict(self, span_dict: dict[str, Any]) -> float:
        """Extract cost from a serialised span dict (used by check())."""
        attrs = span_dict.get("attributes", {}) or {}
        model = attrs.get("model", "") or ""
        tok_in = int(attrs.get("tokens_input", 0) or 0)
        tok_out = int(attrs.get("tokens_output", 0) or 0)
        return self._cost_from_attributes(model, tok_in, tok_out)

    def _reset_if_new_window(self) -> None:
        """Reset counters if the rolling window has elapsed."""
        now = time.time()
        window_seconds = _WINDOW_SECONDS[self.window]
        if now - self._window_start >= window_seconds:
            self._spent = 0.0
            self._warned = False
            self._window_start = now

    def _record_cost(self, cost: float) -> None:
        """Thread-safe accumulation; fires warning/exceeded callbacks as needed.

        Must be called while NOT holding ``self._lock``, because callbacks may
        themselves call into peekr.  The lock is acquired internally only for
        the mutation of shared state.
        """
        if cost <= 0:
            return

        fire_warning = False
        fire_exceeded = False
        spent_snapshot = 0.0

        with self._lock:
            self._reset_if_new_window()
            self._spent += cost
            spent_snapshot = self._spent
            pct = self._spent / self.limit_usd

            if not self._warned and pct >= self.warning_pct:
                self._warned = True
                fire_warning = True

            if pct >= 1.0:
                fire_exceeded = True

        # Fire callbacks outside the lock to avoid potential deadlocks.
        if fire_warning:
            pct_display = (spent_snapshot / self.limit_usd) * 100
            msg = (
                f"[peekr] Budget warning: ${spent_snapshot:.4f} / ${self.limit_usd} "
                f"({pct_display:.0f}%) used this {self.window}"
            )
            if self.on_warning:
                try:
                    self.on_warning(spent_snapshot, self.limit_usd)
                except Exception:
                    pass
            else:
                warnings.warn(msg, stacklevel=4)

        if fire_exceeded:
            msg = (
                f"[peekr] Budget exceeded: ${spent_snapshot:.4f} / ${self.limit_usd} "
                f"this {self.window}"
            )
            if self.on_exceeded:
                try:
                    self.on_exceeded(spent_snapshot, self.limit_usd)
                except Exception:
                    pass
            raise BudgetExceededError(msg)

    # ------------------------------------------------------------------
    # Exporter protocol  (called by export_span / add_exporter)
    # ------------------------------------------------------------------

    def export(self, span: Any) -> None:
        """Called by the exporter pipeline with a Span object."""
        cost = self._cost_from_span(span)
        self._record_cost(cost)

    # ------------------------------------------------------------------
    # Alert protocol  (called by AlertExporter with serialised span dicts)
    # ------------------------------------------------------------------

    def check(self, trace_spans: list[dict[str, Any]]) -> None:
        """Called by AlertExporter with all spans from a completed trace."""
        total_cost = sum(self._cost_from_span_dict(s) for s in trace_spans)
        self._record_cost(total_cost)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def spent(self) -> float:
        """Current rolling spend in USD (snapshot; may change between reads)."""
        with self._lock:
            self._reset_if_new_window()
            return self._spent

    @property
    def remaining(self) -> float:
        """Remaining budget in USD for the current window."""
        return max(0.0, self.limit_usd - self.spent)

    def reset(self) -> None:
        """Manually reset the rolling window (useful in tests)."""
        with self._lock:
            self._spent = 0.0
            self._warned = False
            self._window_start = time.time()

    def __repr__(self) -> str:
        return (
            f"BudgetAlert(limit_usd={self.limit_usd}, window={self.window!r}, "
            f"spent={self._spent:.4f}, warning_pct={self.warning_pct})"
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def budget_alert(
    limit_usd: float,
    on_warning: Optional[Callable[[float, float], None]] = None,
    on_exceeded: Optional[Callable[[float, float], None]] = None,
    warning_pct: float = 0.8,
    window: str = "day",
) -> BudgetAlert:
    """Convenience wrapper — returns ``BudgetAlert(limit_usd, ...)``.

    Example::

        from peekr.budget import budget_alert
        peekr.add_exporter(budget_alert(5.0, window="day"))
    """
    return BudgetAlert(
        limit_usd=limit_usd,
        on_warning=on_warning,
        on_exceeded=on_exceeded,
        warning_pct=warning_pct,
        window=window,
    )
