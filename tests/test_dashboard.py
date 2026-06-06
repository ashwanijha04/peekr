from __future__ import annotations

import json
import os
import time

import pytest

from peekr.dashboard import (
    _channel_heatmap,
    _channel_values,
    _distribution,
    _drift,
    _narrative,
    _rolling,
    _rows,
    _series,
    _summary,
    _verdict_totals,
    _worst_offenders,
    generate_dashboard,
)


def _llm_span(
    trace_id: str,
    start_time: float,
    score: float | None = None,
    rubric: float | None = None,
    tokens: int = 100,
    status: str = "ok",
    output: str = "Some answer.",
    details: dict | None = None,
    tenant: str = "acme",
    endpoint: str = "/api/qa",
    model: str = "gpt-4o-mini",
) -> dict:
    attrs: dict = {
        "model": model,
        "input": "Question?",
        "output": output,
        "tokens_total": tokens,
        "user_id": tenant,
        "endpoint": endpoint,
    }
    if score is not None or rubric is not None:
        attrs["eval_scores"] = {}
        if score is not None:
            attrs["eval_scores"]["Hallucination"] = score
        if rubric is not None:
            attrs["eval_scores"]["Rubric"] = rubric
    if details is not None:
        attrs["hallucination_details"] = details
    return {
        "name": "openai.chat.completions",
        "trace_id": trace_id,
        "span_id": f"span-{trace_id}",
        "parent_id": None,
        "start_time": start_time,
        "end_time": start_time + 0.5,
        "duration_ms": 500.0,
        "status": status,
        "attributes": attrs,
    }


def _make_corpus():
    t0 = time.time()
    # 10 spans, hallucination scores trending downward (drift!)
    spans = []
    for i in range(10):
        score = 0.9 - (i * 0.07)  # 0.9 → 0.27
        spans.append(_llm_span(f"t{i:02d}", t0 + i, score=score, rubric=0.8))
    # one detailed span with verdicts
    spans.append(
        _llm_span(
            "tdet",
            t0 + 11,
            score=0.5,
            details={
                "claims": [
                    {"text": "A is true", "verdict": "supported"},
                    {"text": "B is invented", "verdict": "contradicted"},
                ],
                "supported": 1,
                "contradicted": 1,
                "unsupported": 0,
                "total": 2,
                "score": 0.5,
            },
        )
    )
    return spans


# ---------------------------------------------------------------------------
# Unit-level data prep
# ---------------------------------------------------------------------------


class TestDataPrep:
    def test_series_flattens_eval_scores(self):
        spans = _make_corpus()
        series = _series(spans)
        assert len(series) == 11
        assert series[0]["Hallucination"] == pytest.approx(0.9)
        assert series[0]["Rubric"] == pytest.approx(0.8)

    def test_rolling_returns_running_mean(self):
        spans = _make_corpus()
        out = _rolling(spans, window=3)
        assert len(out["Hallucination"]) == 11
        # First point's rolling mean equals first value
        assert out["Hallucination"][0] == pytest.approx(0.9)
        # Last 3-window mean of Hallucination should be lower than first
        assert out["Hallucination"][-1] < out["Hallucination"][0]

    def test_distribution_is_10_buckets(self):
        spans = _make_corpus()
        dist = _distribution(spans)
        assert len(dist["Hallucination"]) == 10
        assert sum(dist["Hallucination"]) == 11  # every span had a score

    def test_drift_detects_regression(self):
        spans = _make_corpus()
        d = _drift(spans)["Hallucination"]
        assert d is not None
        assert d["current"] < d["baseline"]  # downward drift
        assert d["delta"] < 0

    def test_drift_returns_none_below_threshold(self):
        spans = _make_corpus()[:3]
        assert _drift(spans)["Hallucination"] is None

    def test_worst_offenders_sorted_ascending(self):
        spans = _make_corpus()
        worst = _worst_offenders(spans, k=5)
        assert len(worst) == 5
        scores = [w["score"] for w in worst]
        assert scores == sorted(scores)
        assert scores[0] < scores[-1]

    def test_worst_includes_claim_details_when_present(self):
        spans = _make_corpus()
        worst = _worst_offenders(spans, k=20)
        with_details = [w for w in worst if w["details"]]
        assert len(with_details) == 1
        assert with_details[0]["details"]["contradicted"] == 1

    def test_verdict_totals_sum_across_spans(self):
        spans = _make_corpus()
        totals = _verdict_totals(spans)
        assert totals == {"supported": 1, "contradicted": 1, "unsupported": 0}

    def test_summary_counts(self):
        spans = _make_corpus()
        s = _summary(spans, spans)
        assert s["total_spans"] == 11
        assert s["llm_spans"] == 11
        assert s["scored_spans"] == 11
        assert s["detailed_spans"] == 1
        assert s["error_count"] == 0


# ---------------------------------------------------------------------------
# End-to-end: produce HTML file from a JSONL trace
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_generates_self_contained_html(self, tmp_path):
        traces_path = tmp_path / "traces.jsonl"
        with open(traces_path, "w") as f:
            for s in _make_corpus():
                f.write(json.dumps(s) + "\n")

        out_path = tmp_path / "dashboard.html"
        result = generate_dashboard(str(traces_path), output=str(out_path))

        assert result == str(out_path)
        assert os.path.exists(out_path)
        html = out_path.read_text()
        # Embedded payload
        assert '"summary"' in html
        assert '"drift"' in html
        assert '"verdict_totals"' in html
        # Chart.js CDN link present
        assert "chart.js" in html.lower()
        # Source label is rendered
        assert str(traces_path) in html

    def test_handles_empty_traces(self, tmp_path):
        traces_path = tmp_path / "empty.jsonl"
        traces_path.write_text("")
        out_path = tmp_path / "dashboard.html"
        generate_dashboard(str(traces_path), output=str(out_path))
        # Should still produce a file with the skeleton
        html = out_path.read_text()
        assert "peekr · observability" in html

    def test_rich_payload_keys_present(self, tmp_path):
        """The redesigned UI consumes a richer payload — make sure every key is wired."""
        traces_path = tmp_path / "traces.jsonl"
        with open(traces_path, "w") as f:
            for s in _make_corpus():
                f.write(json.dumps(s) + "\n")
        out_path = tmp_path / "dashboard.html"
        generate_dashboard(str(traces_path), output=str(out_path))
        html = out_path.read_text()
        for key in (
            '"rows"',
            '"channels"',
            '"narrative"',
            '"channel_heatmap"',
            '"thresholds"',
        ):
            assert key in html, f"missing payload key: {key}"
        # Filter chip / hero / offender mount points
        for marker in (
            "filter-bar",
            "hero",
            "narrative-list",
            "offender-list",
            "heatmaps",
        ):
            assert marker in html, f"missing UI mount: {marker}"


# ---------------------------------------------------------------------------
# Rich payload helpers
# ---------------------------------------------------------------------------


class TestRichPayload:
    def test_rows_includes_per_span_record(self):
        spans = _make_corpus()
        rows = _rows(spans)
        assert len(rows) == len(spans)
        first = rows[0]
        for k in (
            "trace_id",
            "span_id",
            "ts",
            "model",
            "tenant",
            "endpoint",
            "Hallucination",
            "Rubric",
            "input",
            "output",
        ):
            assert k in first
        # Tenant comes from attributes.user_id (peekr's session machinery)
        assert first["tenant"] == "acme"

    def test_channel_values_lists_sorted_unique(self):
        spans = _make_corpus()
        # Force a second tenant / endpoint for variety
        spans[0]["attributes"]["user_id"] = "globex"
        spans[0]["attributes"]["endpoint"] = "/api/agent"
        cv = _channel_values(spans)
        assert "model" in cv and "tenant" in cv and "endpoint" in cv
        assert cv["tenant"] == sorted(cv["tenant"])
        assert "acme" in cv["tenant"] and "globex" in cv["tenant"]
        assert "/api/qa" in cv["endpoint"] and "/api/agent" in cv["endpoint"]

    def test_channel_heatmap_buckets_and_grids(self):
        spans = _make_corpus()
        hm = _channel_heatmap(spans, n_buckets=4)
        assert len(hm["buckets"]) == 4
        assert (
            "model" in hm["grids"]
            and "tenant" in hm["grids"]
            and "endpoint" in hm["grids"]
        )
        for grid in hm["grids"].values():
            for row in grid:
                assert len(row["cells"]) == 4
                assert row["n_total"] > 0
                for cell in row["cells"]:
                    if cell["mean"] is not None:
                        assert 0.0 <= cell["mean"] <= 1.0
                        assert cell["n"] > 0

    def test_channel_heatmap_handles_no_data(self):
        hm = _channel_heatmap([], n_buckets=4)
        assert hm["buckets"] == [] and hm["grids"] == {}

    def test_narrative_marks_regression(self):
        spans = _make_corpus()
        n = _narrative(spans)
        assert n["health"] is not None
        assert n["health"]["tier"] in ("good", "ok", "warning", "critical")
        # Corpus trends down, so the narrative should mention the drop
        joined = " ".join(n["insights"])
        assert (
            "regress" in joined.lower()
            or "drop" in joined.lower()
            or "dropped" in joined.lower()
        )

    def test_narrative_handles_no_scores(self):
        # Strip eval_scores from every span
        spans = _make_corpus()
        for s in spans:
            (s["attributes"]).pop("eval_scores", None)
        n = _narrative(spans)
        assert n["health"] is None
        assert any(
            "no Hallucination scores" in i.lower() or "no hallucination" in i.lower()
            for i in n["insights"]
        )
