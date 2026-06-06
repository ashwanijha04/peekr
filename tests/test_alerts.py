from __future__ import annotations

import time
from typing import Any


from peekr.alerts import (
    AlertExporter,
    CostSpike,
    ErrorRate,
    LatencyP95,
    TokenGrowth,
    alert,
    _BaseAlert,
)
from peekr.span import Span


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_span(
    name: str = "test",
    parent_id: str | None = None,
    status: str = "ok",
    tokens: int = 0,
    duration_ms: float = 100.0,
    trace_id: str = "trace-1",
) -> dict[str, Any]:
    """Return a span-dict that looks like Span.to_dict()."""
    return {
        "name": name,
        "trace_id": trace_id,
        "span_id": f"span-{name}",
        "parent_id": parent_id,
        "start_time": time.time(),
        "end_time": time.time() + duration_ms / 1000,
        "duration_ms": duration_ms,
        "status": status,
        "attributes": {"tokens_total": tokens} if tokens else {},
    }


def _root_span(**kwargs) -> Span:
    """Create a real Span object with parent_id=None (root)."""
    s = Span(
        name=kwargs.get("name", "root"), trace_id=kwargs.get("trace_id", "trace-1")
    )
    s.attributes.update(kwargs.get("attributes", {}))
    s.status = kwargs.get("status", "ok")
    s.finish()
    return s


def _child_span(parent: Span, name: str = "child", **kwargs) -> Span:
    s = Span(name=name, trace_id=parent.trace_id, parent_id=parent.span_id)
    s.attributes.update(kwargs.get("attributes", {}))
    s.status = kwargs.get("status", "ok")
    s.finish()
    return s


# ---------------------------------------------------------------------------
# AlertExporter accumulation & dispatch
# ---------------------------------------------------------------------------


class _RecordingAlert(_BaseAlert):
    def __init__(self):
        self.calls: list[list[dict]] = []

    def check(self, trace_spans):
        self.calls.append(trace_spans)


def test_exporter_accumulates_child_spans_before_root():
    recorder = _RecordingAlert()
    exp = AlertExporter([recorder])

    root = _root_span(trace_id="t1")
    child = _child_span(root, name="child1")

    # export child first, then root
    exp.export(child)
    assert recorder.calls == [], "should not fire until root arrives"

    exp.export(root)
    assert len(recorder.calls) == 1
    names = {s["name"] for s in recorder.calls[0]}
    assert "child1" in names
    assert "root" in names


def test_exporter_clears_accumulator_after_root():
    recorder = _RecordingAlert()
    exp = AlertExporter([recorder])

    root = _root_span(trace_id="t2")
    exp.export(root)

    # second root with same trace_id should produce an independent call
    root2 = _root_span(trace_id="t2")
    exp.export(root2)

    assert len(recorder.calls) == 2
    # each call should contain exactly one span
    assert len(recorder.calls[0]) == 1
    assert len(recorder.calls[1]) == 1


def test_exporter_calls_all_alerts():
    r1, r2 = _RecordingAlert(), _RecordingAlert()
    exp = AlertExporter([r1, r2])

    root = _root_span(trace_id="t3")
    exp.export(root)

    assert len(r1.calls) == 1
    assert len(r2.calls) == 1


def test_exporter_multiple_traces_independent():
    recorder = _RecordingAlert()
    exp = AlertExporter([recorder])

    root_a = _root_span(trace_id="trace-a", name="root-a")
    root_b = _root_span(trace_id="trace-b", name="root-b")

    exp.export(root_a)
    exp.export(root_b)

    assert len(recorder.calls) == 2


# ---------------------------------------------------------------------------
# ErrorRate
# ---------------------------------------------------------------------------


class _TriggerCapture(_BaseAlert):
    """Mixin that records trigger messages instead of printing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.triggered: list[str] = []

    def on_trigger(self, message: str) -> None:
        self.triggered.append(message)


class _ErrorRate(_TriggerCapture, ErrorRate):
    pass


class _CostSpike(_TriggerCapture, CostSpike):
    pass


class _LatencyP95(_TriggerCapture, LatencyP95):
    pass


class _TokenGrowth(_TriggerCapture, TokenGrowth):
    pass


def test_error_rate_no_trigger_below_threshold():
    er = _ErrorRate(threshold=0.5, window=10)
    spans = [_make_span(status="ok")]
    er.check(spans)
    assert er.triggered == []


def test_error_rate_triggers_when_exceeded():
    er = _ErrorRate(threshold=0.5, window=4)
    # 3 errors out of 4 → 75 % > 50 %
    for _ in range(3):
        er.check([_make_span(status="error")])
    er.check([_make_span(status="error")])
    assert er.triggered, "should have triggered"
    assert "ErrorRate" in er.triggered[-1]


def test_error_rate_rolling_window_evicts_old_data():
    er = _ErrorRate(threshold=0.5, window=4)
    # fill window with errors; this will trigger
    for _ in range(4):
        er.check([_make_span(status="error")])
    er.triggered.clear()

    # slide in 4 ok traces; after all 4 the window is 100% ok — no new trigger
    # (intermediate traces still see a mixed window and may trigger, so we
    # only assert the very last state: once the window is all-ok, no trigger)
    for _ in range(4):
        er.check([_make_span(status="ok")])
        er.triggered.clear()  # reset after each step; only care about final

    # final window is [ok, ok, ok, ok] → 0% error rate → no trigger
    er.check([_make_span(status="ok")])
    assert er.triggered == []


def test_error_rate_trace_with_mixed_spans():
    er = _ErrorRate(threshold=0.0, window=10)
    # even one error span should mark the trace as errored
    spans = [
        _make_span(status="ok"),
        _make_span(status="error"),
    ]
    er.check(spans)
    assert er.triggered


# ---------------------------------------------------------------------------
# CostSpike
# ---------------------------------------------------------------------------


def test_cost_spike_no_history_no_trigger():
    cs = _CostSpike(multiplier=2.0, window=10)
    cs.check([_make_span(tokens=1000)])
    assert cs.triggered == []


def test_cost_spike_triggers_on_spike():
    cs = _CostSpike(multiplier=2.0, window=10)
    # establish baseline
    for _ in range(5):
        cs.check([_make_span(tokens=100)])
    cs.triggered.clear()

    # send a trace with 3× the average → should trigger
    cs.check([_make_span(tokens=300)])
    assert cs.triggered
    assert "CostSpike" in cs.triggered[-1]


def test_cost_spike_no_trigger_within_threshold():
    cs = _CostSpike(multiplier=3.0, window=10)
    for _ in range(5):
        cs.check([_make_span(tokens=100)])
    cs.triggered.clear()

    # 2× should NOT trigger when multiplier is 3×
    cs.check([_make_span(tokens=200)])
    assert cs.triggered == []


def test_cost_spike_uses_sum_of_tokens_across_spans():
    cs = _CostSpike(multiplier=2.0, window=10)
    for _ in range(5):
        cs.check([_make_span(tokens=50), _make_span(tokens=50)])  # 100 each
    cs.triggered.clear()

    # sum 300 > 2×100 avg
    cs.check([_make_span(tokens=150), _make_span(tokens=150)])
    assert cs.triggered


# ---------------------------------------------------------------------------
# LatencyP95
# ---------------------------------------------------------------------------


def test_latency_p95_no_trigger_below_threshold():
    lp = _LatencyP95(ms=5000)
    spans = [_make_span(duration_ms=100) for _ in range(20)]
    lp.check(spans)
    assert lp.triggered == []


def test_latency_p95_triggers_when_exceeded():
    lp = _LatencyP95(ms=500)
    # With 10 spans the P95 index is ceil(10*0.95)-1 = 9 (the last element).
    # Make the last (slowest) span 6000ms so it lands at P95.
    spans = [_make_span(duration_ms=100) for _ in range(9)]
    spans.append(_make_span(duration_ms=6000))
    lp.check(spans)
    assert lp.triggered
    assert "LatencyP95" in lp.triggered[-1]


def test_latency_p95_empty_spans_no_trigger():
    lp = _LatencyP95(ms=100)
    lp.check([])
    assert lp.triggered == []


def test_latency_p95_single_fast_span_no_trigger():
    lp = _LatencyP95(ms=1000)
    lp.check([_make_span(duration_ms=50)])
    assert lp.triggered == []


def test_latency_p95_single_slow_span_triggers():
    lp = _LatencyP95(ms=1000)
    lp.check([_make_span(duration_ms=2000)])
    assert lp.triggered


def test_latency_p95_ignores_none_duration():
    lp = _LatencyP95(ms=100)
    spans = [_make_span(duration_ms=50)]
    spans[0]["duration_ms"] = None  # simulate an unfinished span
    lp.check(spans)
    assert lp.triggered == []


# ---------------------------------------------------------------------------
# TokenGrowth
# ---------------------------------------------------------------------------


def test_token_growth_no_trigger_too_few_runs():
    tg = _TokenGrowth(runs=5)
    for i in range(4):
        tg.check([_make_span(tokens=i * 10)])
    assert tg.triggered == []


def test_token_growth_triggers_on_consecutive_growth():
    tg = _TokenGrowth(runs=3)
    tg.check([_make_span(tokens=10)])
    tg.check([_make_span(tokens=20)])
    tg.check([_make_span(tokens=30)])
    tg.check([_make_span(tokens=40)])
    assert tg.triggered
    assert "TokenGrowth" in tg.triggered[-1]


def test_token_growth_no_trigger_on_plateau():
    tg = _TokenGrowth(runs=3)
    tg.check([_make_span(tokens=100)])
    tg.check([_make_span(tokens=100)])
    tg.check([_make_span(tokens=100)])
    tg.check([_make_span(tokens=100)])
    assert tg.triggered == []


def test_token_growth_no_trigger_on_decrease():
    tg = _TokenGrowth(runs=3)
    tg.check([_make_span(tokens=100)])
    tg.check([_make_span(tokens=90)])
    tg.check([_make_span(tokens=80)])
    tg.check([_make_span(tokens=70)])
    assert tg.triggered == []


def test_token_growth_resets_after_drop():
    tg = _TokenGrowth(runs=3)
    for i in range(1, 5):
        tg.check([_make_span(tokens=i * 10)])
    assert tg.triggered
    tg.triggered.clear()

    # inject a drop — rolling window now has a decrease, no new trigger
    tg.check([_make_span(tokens=5)])
    assert tg.triggered == []


def test_token_growth_sums_multiple_spans():
    tg = _TokenGrowth(runs=3)
    # each trace has two spans; growth is still detected
    tg.check([_make_span(tokens=10), _make_span(tokens=10)])  # 20
    tg.check([_make_span(tokens=20), _make_span(tokens=20)])  # 40
    tg.check([_make_span(tokens=30), _make_span(tokens=30)])  # 60
    tg.check([_make_span(tokens=40), _make_span(tokens=40)])  # 80
    assert tg.triggered


# ---------------------------------------------------------------------------
# alert namespace alias
# ---------------------------------------------------------------------------


def test_alert_namespace_has_all_types():
    assert alert.ErrorRate is ErrorRate
    assert alert.CostSpike is CostSpike
    assert alert.LatencyP95 is LatencyP95
    assert alert.TokenGrowth is TokenGrowth


def test_alert_namespace_instantiable():
    er = alert.ErrorRate(threshold=0.1, window=50)
    assert isinstance(er, ErrorRate)


# ---------------------------------------------------------------------------
# on_trigger default writes to stderr (smoke test)
# ---------------------------------------------------------------------------


def test_default_on_trigger_writes_to_stderr(capsys):
    base = _BaseAlert()
    base.on_trigger("hello alert")
    captured = capsys.readouterr()
    assert "hello alert" in captured.err


# ---------------------------------------------------------------------------
# Integration: AlertExporter with real Span objects
# ---------------------------------------------------------------------------


def test_alert_exporter_full_integration():
    triggered: list[str] = []

    class CapturingErrorRate(ErrorRate):
        def on_trigger(self, message):
            triggered.append(message)

    exp = AlertExporter([CapturingErrorRate(threshold=0.0, window=10)])

    root = _root_span(trace_id="full-t1", status="error")
    exp.export(root)

    assert triggered, "alert should have fired"
