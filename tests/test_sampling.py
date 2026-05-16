"""Tests for sampling controls.

Sampling is a *trace-level* decision made at root-span creation and
inherited by all children via ContextVar. Storage exporters skip dropped
spans; mutating exporters (Eval, Alerts) always see every span so their
metrics stay accurate.

These tests verify:
  - 100% sample rate keeps everything (default behaviour unchanged)
  - 0% sample rate drops everything (except errors when keep_errors=True)
  - 0% + keep_errors=False drops absolutely everything
  - All children of a kept trace are kept; all children of a dropped trace
    are dropped — coherent traces, no orphan spans
  - Eval/Alert exporters still see every span regardless of sampling
  - Statistical sanity check at ~50%
  - validation of sample_rate bounds
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import pytest

from peekr import context as ctx
from peekr.context import set_process_defaults, start_span, end_span
from peekr.exporters import _exporters, add_exporter, export_span
from peekr.span import Span


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

@contextmanager
def _isolated_sampling(sample_rate: Optional[float] = None,
                       keep_errors: Optional[bool] = None):
    """Reset sampling defaults around a test, plus clear exporters + contextvar."""
    saved = (ctx._sample_rate, ctx._keep_errors)
    saved_exporters = list(_exporters)
    _exporters.clear()
    # Force a fresh sampling decision per test
    token = ctx._trace_sample_keep.set(None)
    if sample_rate is not None or keep_errors is not None:
        set_process_defaults(sample_rate=sample_rate, keep_errors=keep_errors)
    try:
        yield
    finally:
        ctx._sample_rate, ctx._keep_errors = saved
        ctx._trace_sample_keep.reset(token)
        _exporters.clear()
        _exporters.extend(saved_exporters)


class _StorageExporter:
    """Marker-bearing capture exporter for tests."""
    _is_storage = True

    def __init__(self):
        self.spans: list[Span] = []

    def export(self, span):
        self.spans.append(span)


class _MutatingExporter:
    """No _is_storage marker — should see every span."""

    def __init__(self):
        self.spans: list[Span] = []

    def export(self, span):
        self.spans.append(span)


def _drive_trace(*, n_children: int = 2, child_status: str = "ok") -> None:
    """Create a root span + N children and export each."""
    root, root_tok = start_span("root")
    root.finish()
    for i in range(n_children):
        child, ctok = start_span(f"child{i}")
        child.status = child_status
        child.finish()
        end_span(child, ctok)
        export_span(child)
    end_span(root, root_tok)
    export_span(root)


# ---------------------------------------------------------------------------
# 1. Default behaviour — keep everything
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_full_sample_rate_keeps_everything(self):
        with _isolated_sampling(sample_rate=1.0):
            cap = _StorageExporter()
            add_exporter(cap)
            _drive_trace(n_children=3)
            assert len(cap.spans) == 4  # root + 3 children

    def test_no_sampling_setting_keeps_everything(self):
        """Out of the box (no instrument(sample_rate=...) call), nothing drops."""
        with _isolated_sampling():
            cap = _StorageExporter()
            add_exporter(cap)
            _drive_trace(n_children=2)
            assert len(cap.spans) == 3


# ---------------------------------------------------------------------------
# 2. Sampling drops storage spans; mutators still see them
# ---------------------------------------------------------------------------

class TestSampling:
    def test_zero_rate_drops_all_non_error_spans(self):
        with _isolated_sampling(sample_rate=0.0, keep_errors=True):
            cap = _StorageExporter()
            add_exporter(cap)
            _drive_trace(n_children=3, child_status="ok")
            assert cap.spans == []

    def test_zero_rate_with_keep_errors_false_drops_everything(self):
        with _isolated_sampling(sample_rate=0.0, keep_errors=False):
            cap = _StorageExporter()
            add_exporter(cap)
            _drive_trace(n_children=2, child_status="error")
            assert cap.spans == []

    def test_errored_span_kept_even_when_trace_dropped(self):
        with _isolated_sampling(sample_rate=0.0, keep_errors=True):
            cap = _StorageExporter()
            add_exporter(cap)
            _drive_trace(n_children=2, child_status="error")
            # All 2 child spans errored → kept. Root succeeded → dropped.
            assert len(cap.spans) == 2
            assert all(s.status == "error" for s in cap.spans)

    def test_mutators_see_every_span_regardless_of_sampling(self):
        with _isolated_sampling(sample_rate=0.0, keep_errors=False):
            storage = _StorageExporter()
            mutator = _MutatingExporter()
            add_exporter(mutator)
            add_exporter(storage)
            _drive_trace(n_children=3)
            assert len(storage.spans) == 0
            assert len(mutator.spans) == 4  # mutator always sees everything


# ---------------------------------------------------------------------------
# 3. Trace coherence — children inherit the root decision
# ---------------------------------------------------------------------------

class TestTraceCoherence:
    def test_all_children_of_kept_trace_are_kept(self):
        with _isolated_sampling(sample_rate=1.0):
            cap = _StorageExporter()
            add_exporter(cap)
            _drive_trace(n_children=5)
            assert len(cap.spans) == 6
            assert {s.trace_id for s in cap.spans} == {cap.spans[0].trace_id}

    def test_all_children_of_dropped_trace_are_dropped(self):
        with _isolated_sampling(sample_rate=0.0, keep_errors=False):
            cap = _StorageExporter()
            add_exporter(cap)
            _drive_trace(n_children=5)
            assert cap.spans == []  # no orphan children


# ---------------------------------------------------------------------------
# 4. Statistical sanity at 50%
# ---------------------------------------------------------------------------

class TestStatistical:
    def test_fifty_percent_rate_is_roughly_half(self):
        with _isolated_sampling(sample_rate=0.5):
            cap = _StorageExporter()
            add_exporter(cap)
            n = 400  # enough to bound flakiness; tolerance is wide
            kept_traces = 0
            for _ in range(n):
                # Each iteration is a brand-new "root" — reset contextvar
                token = ctx._trace_sample_keep.set(None)
                root, tok = start_span("root")
                root.finish()
                end_span(root, tok)
                export_span(root)
                ctx._trace_sample_keep.reset(token)
                if cap.spans and cap.spans[-1].name == "root" and len(cap.spans) > kept_traces:
                    kept_traces = len(cap.spans)
            # 99.999% CI for binomial(400, 0.5) is roughly 150–250
            assert 150 <= kept_traces <= 250, f"got {kept_traces} kept of {n}"


# ---------------------------------------------------------------------------
# 5. Validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_rejects_negative_rate(self):
        with pytest.raises(ValueError):
            set_process_defaults(sample_rate=-0.1)

    def test_rejects_rate_above_one(self):
        with pytest.raises(ValueError):
            set_process_defaults(sample_rate=1.5)

    def test_accepts_boundary_values(self):
        set_process_defaults(sample_rate=0.0)
        set_process_defaults(sample_rate=1.0)
        # restore default so we don't pollute other tests
        set_process_defaults(sample_rate=1.0)
