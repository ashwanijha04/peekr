"""Static HTML dashboard for hallucination, eval scores, and drift.

`peekr dashboard <path> [-o report.html]` reads traces from JSONL or SQLite
and emits a single self-contained HTML file. Charts are rendered client-side
with Chart.js (loaded from a CDN). No server, no build step, no backend —
consistent with the rest of peekr.
"""

from __future__ import annotations

import html
import json
from typing import Any

_LLM_PREFIXES = ("openai.", "anthropic.", "bedrock.")
_METRICS = ("Hallucination", "Rubric", "CitationAccuracy", "NotEmpty", "NoError")
# Attribute keys used to bucket spans for the "drift by channel" panel.
# (Field name, display label.)
_CHANNEL_FIELDS = (("model", "model"), ("user_id", "tenant"), ("endpoint", "endpoint"))
_BUCKET_COUNT = 10
_WORST_LIMIT = 20
_ROLLING_WINDOW = 20


def generate_dashboard(path: str, output: str = "dashboard.html") -> str:
    from .cli import _read_jsonl, _read_sqlite  # noqa: PLC0415 — reuse loaders

    if path.endswith(".db"):
        spans = _read_sqlite(path)
    else:
        spans = _read_jsonl(path)

    llm_spans = [
        s
        for s in spans
        if any(s["name"].startswith(p) for p in _LLM_PREFIXES)
        # Hide evaluator-judge calls — they're already in the JSONL for auditing
        # token costs, but they'd otherwise appear as duplicate worst-offender
        # cards and skew distributions.
        and not (s.get("attributes") or {}).get("peekr.internal")
    ]
    llm_spans.sort(key=lambda s: s.get("start_time") or 0)

    data = {
        "summary": _summary(spans, llm_spans),
        "series": _series(llm_spans),
        "rolling": _rolling(llm_spans, window=_ROLLING_WINDOW),
        "distribution": _distribution(llm_spans),
        "drift": _drift(llm_spans),
        "worst": _worst_offenders(llm_spans, k=_WORST_LIMIT),
        "verdict_totals": _verdict_totals(llm_spans),
        "channel_drift": _channel_drift(llm_spans),
        "citation_totals": _citation_totals(llm_spans),
        # New, richer payload for the redesigned UI
        "rows": _rows(llm_spans),
        "channels": _channel_values(llm_spans),
        "narrative": _narrative(llm_spans),
        "channel_heatmap": _channel_heatmap(llm_spans, n_buckets=6),
        "thresholds": {"warning": 0.7, "critical": 0.5},
    }

    rendered = _render_html(data, source=path)
    with open(output, "w") as f:
        f.write(rendered)
    return output


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------


def _scores(span: dict) -> dict[str, float]:
    return (span.get("attributes") or {}).get("eval_scores") or {}


def _summary(all_spans: list[dict], llm_spans: list[dict]) -> dict[str, Any]:
    eval_spans = [s for s in llm_spans if _scores(s)]
    detailed = [
        s
        for s in eval_spans
        if (s.get("attributes") or {}).get("hallucination_details")
    ]
    times = [s.get("start_time") for s in llm_spans if s.get("start_time")]
    return {
        "total_spans": len(all_spans),
        "llm_spans": len(llm_spans),
        "scored_spans": len(eval_spans),
        "detailed_spans": len(detailed),
        "error_count": sum(1 for s in llm_spans if s.get("status") == "error"),
        "first_ts": min(times) if times else None,
        "last_ts": max(times) if times else None,
    }


def _series(llm_spans: list[dict]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for s in llm_spans:
        attrs = s.get("attributes") or {}
        scores = _scores(s)
        points.append(
            {
                "trace_id": (s.get("trace_id") or "")[:8],
                "ts": s.get("start_time") or 0,
                "Hallucination": scores.get("Hallucination"),
                "Rubric": scores.get("Rubric"),
                "CitationAccuracy": scores.get("CitationAccuracy"),
                "NotEmpty": scores.get("NotEmpty"),
                "NoError": scores.get("NoError"),
                "error": 1 if s.get("status") == "error" else 0,
                "tokens": attrs.get("tokens_total") or 0,
                "duration_ms": s.get("duration_ms") or 0,
                "tenant": attrs.get("user_id"),
                "endpoint": attrs.get("endpoint"),
                "model": attrs.get("model"),
            }
        )
    return points


def _rolling(llm_spans: list[dict], window: int) -> dict[str, list[float | None]]:
    """N-trace rolling mean for each eval metric (None where the window has no data)."""
    out: dict[str, list[float | None]] = {m: [] for m in _METRICS}
    buf: dict[str, list[float]] = {m: [] for m in _METRICS}
    for s in llm_spans:
        scores = _scores(s)
        for m in _METRICS:
            v = scores.get(m)
            if v is not None:
                buf[m].append(v)
                if len(buf[m]) > window:
                    buf[m].pop(0)
            out[m].append(sum(buf[m]) / len(buf[m]) if buf[m] else None)
    return out


def _distribution(llm_spans: list[dict]) -> dict[str, list[int]]:
    """Score histogram per metric — 10 equal-width buckets over [0, 1]."""
    dist: dict[str, list[int]] = {m: [0] * _BUCKET_COUNT for m in _METRICS}
    for s in llm_spans:
        scores = _scores(s)
        for m in _METRICS:
            v = scores.get(m)
            if v is None:
                continue
            idx = min(int(float(v) * _BUCKET_COUNT), _BUCKET_COUNT - 1)
            dist[m][idx] += 1
    return dist


def _drift(llm_spans: list[dict]) -> dict[str, dict[str, Any] | None]:
    """For each metric, mean over the oldest 30% vs the newest 30%, plus delta."""
    out: dict[str, dict[str, Any] | None] = {}
    for m in _METRICS:
        values = [
            s for s in (_scores(span).get(m) for span in llm_spans) if s is not None
        ]
        n = len(values)
        if n < 6:
            out[m] = None
            continue
        k = max(1, n // 3)
        baseline = sum(values[:k]) / k
        current = sum(values[-k:]) / k
        out[m] = {
            "baseline": baseline,
            "current": current,
            "delta": current - baseline,
            "n_baseline": k,
            "n_current": k,
            "n_total": n,
        }
    return out


def _channel_drift(llm_spans: list[dict]) -> dict[str, list[dict]]:
    """For each channel field (model / tenant / endpoint), compute per-segment drift
    of the Hallucination score: baseline (oldest third) → current (newest third)."""
    out: dict[str, list[dict]] = {}
    for field, label in _CHANNEL_FIELDS:
        buckets: dict[str, list[dict]] = {}
        for s in llm_spans:
            val = (s.get("attributes") or {}).get(field)
            if not val:
                continue
            buckets.setdefault(val, []).append(s)

        rows: list[dict] = []
        for segment, items in buckets.items():
            # Sort segment's spans by time so baseline/current windows are real
            items_sorted = sorted(items, key=lambda x: x.get("start_time") or 0)
            scores = [_scores(x).get("Hallucination") for x in items_sorted]
            scores = [v for v in scores if v is not None]
            n = len(scores)
            if n < 4:
                # Not enough data — still include the segment but mark NA
                rows.append(
                    {
                        "segment": segment,
                        "n": n,
                        "current": (sum(scores) / n) if n else None,
                        "baseline": None,
                        "delta": None,
                        "n_total": n,
                    }
                )
                continue
            k = max(1, n // 3)
            baseline = sum(scores[:k]) / k
            current = sum(scores[-k:]) / k
            rows.append(
                {
                    "segment": segment,
                    "n": n,
                    "baseline": baseline,
                    "current": current,
                    "delta": current - baseline,
                    "n_baseline": k,
                    "n_current": k,
                    "n_total": n,
                }
            )

        # Most-degraded segments first (most negative delta or lowest current)
        def _sort_key(r: dict) -> tuple[float, float]:
            delta = r.get("delta")
            current = r.get("current")
            return (
                delta if delta is not None else 0.0,
                current if current is not None else 1.0,
            )

        rows.sort(key=_sort_key)
        out[label] = rows
    return out


def _citation_totals(llm_spans: list[dict]) -> dict[str, int]:
    totals = {"total": 0, "grounded": 0, "invented": 0}
    for s in llm_spans:
        details = (s.get("attributes") or {}).get("citation_details")
        if not details:
            continue
        for k in totals:
            totals[k] += int(details.get(k, 0) or 0)
    return totals


def _verdict_totals(llm_spans: list[dict]) -> dict[str, int]:
    totals = {"supported": 0, "contradicted": 0, "unsupported": 0}
    for s in llm_spans:
        details = (s.get("attributes") or {}).get("hallucination_details")
        if not details:
            continue
        for k in totals:
            totals[k] += int(details.get(k, 0) or 0)
    return totals


def _worst_offenders(llm_spans: list[dict], k: int) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for s in llm_spans:
        attrs = s.get("attributes") or {}
        h = _scores(s).get("Hallucination")
        if h is None:
            continue
        scored.append(
            {
                "trace_id": s.get("trace_id"),
                "span_id": s.get("span_id"),
                "ts": s.get("start_time") or 0,
                "model": attrs.get("model", ""),
                "score": h,
                "output": (attrs.get("output") or "")[:300],
                "input": (attrs.get("input") or "")[:300],
                "details": attrs.get("hallucination_details"),
            }
        )
    scored.sort(key=lambda x: x["score"])
    return scored[:k]


# ---------------------------------------------------------------------------
# Rich payload for the redesigned UI
# ---------------------------------------------------------------------------


def _rows(llm_spans: list[dict]) -> list[dict[str, Any]]:
    """One record per LLM span with everything the redesigned UI needs.

    Pre-computed so client-side filtering is cheap — the user can toggle
    chips and every panel refilters/re-renders from this single array.
    """
    out: list[dict[str, Any]] = []
    for s in llm_spans:
        attrs = s.get("attributes") or {}
        scores = _scores(s)
        out.append(
            {
                "trace_id": s.get("trace_id"),
                "span_id": s.get("span_id"),
                "ts": s.get("start_time") or 0,
                "model": attrs.get("model"),
                "tenant": attrs.get("user_id"),
                "endpoint": attrs.get("endpoint"),
                "status": s.get("status", "ok"),
                "tokens": attrs.get("tokens_total") or 0,
                "duration_ms": s.get("duration_ms") or 0,
                "Hallucination": scores.get("Hallucination"),
                "Rubric": scores.get("Rubric"),
                "CitationAccuracy": scores.get("CitationAccuracy"),
                "NotEmpty": scores.get("NotEmpty"),
                "NoError": scores.get("NoError"),
                "input": (attrs.get("input") or "")[:600],
                "output": (attrs.get("output") or "")[:600],
                # Fallback for Anthropic spans captured before the patch merged
                # `system` into messages — the dashboard's parseInput uses this.
                "system": (attrs.get("system") or "")[:600]
                if attrs.get("system")
                else None,
                "details": attrs.get("hallucination_details"),
                "citation_details": attrs.get("citation_details"),
                "error": attrs.get("error"),
                # Evaluator failures (judge unreachable, parse error, etc.). Empty
                # dict when no failures — distinct from a real 0.0 score, which
                # lives in `eval_scores`.
                "eval_errors": dict(attrs.get("eval_errors") or {}),
            }
        )
    return out


def _channel_values(llm_spans: list[dict]) -> dict[str, list[str]]:
    """Sorted unique values for each channel field — drives the filter chips."""
    out: dict[str, set[str]] = {label: set() for _, label in _CHANNEL_FIELDS}
    field_by_label = {label: field for field, label in _CHANNEL_FIELDS}
    for s in llm_spans:
        attrs = s.get("attributes") or {}
        for label, vals in out.items():
            v = attrs.get(field_by_label[label])
            if v:
                vals.add(str(v))
    return {k: sorted(v) for k, v in out.items()}


def _channel_heatmap(llm_spans: list[dict], n_buckets: int = 6) -> dict[str, Any]:
    """Mean Hallucination score per (channel-segment, time-bucket) grid.

    Produces a structure like:
        {
          "field_labels": ["model", "tenant", "endpoint"],
          "buckets": [{"start": ts0, "end": ts1, "label": "07:30"}, ...],
          "grids": {
              "model": [
                  {"segment": "gpt-4o-mini", "n_total": 32,
                   "cells": [{"mean": 0.82, "n": 5}, {"mean": 0.41, "n": 6}, ...]},
                  ...
              ],
              "tenant": [...],
              "endpoint": [...],
          }
        }
    """
    times = [s.get("start_time") for s in llm_spans if s.get("start_time")]
    if not times:
        return {"buckets": [], "grids": {}, "field_labels": []}

    t_min, t_max = min(times), max(times)
    span = t_max - t_min if t_max > t_min else 1.0
    width = span / n_buckets

    buckets: list[dict[str, Any]] = []
    for i in range(n_buckets):
        b_start = t_min + i * width
        b_end = t_min + (i + 1) * width
        buckets.append(
            {
                "start": b_start,
                "end": b_end,
                "label": f"{i + 1}/{n_buckets}",  # short label; tooltip gives the time range
            }
        )

    grids: dict[str, list[dict[str, Any]]] = {}
    for field, label in _CHANNEL_FIELDS:
        by_segment: dict[str, list[list[float]]] = {}
        for s in llm_spans:
            attrs = s.get("attributes") or {}
            seg = attrs.get(field)
            score = _scores(s).get("Hallucination")
            ts = s.get("start_time")
            if not seg or score is None or ts is None:
                continue
            idx = min(int((ts - t_min) / width), n_buckets - 1) if width > 0 else 0
            if seg not in by_segment:
                by_segment[seg] = [[] for _ in range(n_buckets)]
            by_segment[seg][idx].append(score)

        rows: list[dict[str, Any]] = []
        for seg, cell_lists in by_segment.items():
            cells = [
                {"mean": sum(c) / len(c), "n": len(c)} if c else {"mean": None, "n": 0}
                for c in cell_lists
            ]
            n_total = sum(c["n"] for c in cells)
            rows.append(
                {
                    "segment": str(seg),
                    "n_total": n_total,
                    "cells": cells,
                }
            )

        # Sort: most degraded segments (lowest current-bucket mean) first.
        def _sort_key(r: dict[str, Any]) -> float:
            tail = [c["mean"] for c in r["cells"][-2:] if c["mean"] is not None]
            return min(tail) if tail else 1.0

        rows.sort(key=_sort_key)
        grids[label] = rows

    return {
        "field_labels": [label for _, label in _CHANNEL_FIELDS],
        "buckets": buckets,
        "grids": grids,
    }


def _narrative(llm_spans: list[dict]) -> dict[str, Any]:
    """Plain-English insights computed from the trace data.

    Returns:
        {
          "health": {"score": 0.65, "tier": "warning", "tier_label": "needs attention",
                     "flagged_pct": 0.17, "flagged_count": 21, "scored_count": 120,
                     "baseline": 0.92, "current": 0.65, "delta": -0.27},
          "insights": [str, ...],
        }
    """
    scored = [
        (s.get("start_time") or 0, _scores(s)["Hallucination"], s)
        for s in llm_spans
        if _scores(s).get("Hallucination") is not None
    ]
    insights: list[str] = []

    if not scored:
        return {
            "health": None,
            "insights": [
                "No Hallucination scores recorded — add peekr.eval.Hallucination() to your evaluators."
            ],
        }

    scored.sort(key=lambda x: x[0])
    score_values = [v for _, v, _ in scored]
    n = len(score_values)
    current = sum(score_values[-max(1, n // 3) :]) / max(1, n // 3)
    baseline = (
        sum(score_values[: max(1, n // 3)]) / max(1, n // 3) if n >= 6 else current
    )

    delta = current - baseline
    flagged = [s for _, v, s in scored if v < 0.5]
    flagged_pct = len(flagged) / n

    if current >= 0.85 and flagged_pct < 0.1:
        tier, tier_label = "good", "healthy"
    elif current >= 0.7 and flagged_pct < 0.2:
        tier, tier_label = "ok", "watch"
    elif current >= 0.5:
        tier, tier_label = "warning", "needs attention"
    else:
        tier, tier_label = "critical", "regressing"

    if n >= 6:
        if delta < -0.1:
            insights.append(
                f"Hallucination score regressed by {delta:+.2f} from baseline ({baseline:.2f} → {current:.2f})."
            )
        elif delta > 0.1:
            insights.append(
                f"Hallucination score improved by {delta:+.2f} from baseline ({baseline:.2f} → {current:.2f})."
            )

    # Worst (model, tenant, endpoint) combo by % flagged
    combos: dict[tuple[str, str, str], list[float]] = {}
    for _, v, s in scored:
        a = s.get("attributes") or {}
        key = (a.get("model") or "?", a.get("user_id") or "?", a.get("endpoint") or "?")
        combos.setdefault(key, []).append(v)
    if combos:
        worst = min(
            (((k, vs) for k, vs in combos.items() if len(vs) >= 3)),
            default=None,
            key=lambda kv: sum(kv[1]) / len(kv[1]),
        )
        if worst:
            (model, tenant, endpoint), vs = worst
            mean = sum(vs) / len(vs)
            insights.append(
                f"Worst channel: {model} · {tenant} · {endpoint} — mean score {mean:.2f} across {len(vs)} calls."
            )

    # Citation invention rate
    cit_total = 0
    cit_invented = 0
    for s in llm_spans:
        d = (s.get("attributes") or {}).get("citation_details")
        if not d:
            continue
        cit_total += int(d.get("total", 0))
        cit_invented += int(d.get("invented", 0))
    if cit_total:
        rate = cit_invented / cit_total
        insights.append(
            f"Citations: {cit_invented} of {cit_total} were invented ({rate:.0%}). "
            "Invented references are common in RAG when retrieval misses."
        )

    # Most common verdict type when detailed mode is on
    verdict_totals = {"supported": 0, "contradicted": 0, "unsupported": 0}
    for s in llm_spans:
        d = (s.get("attributes") or {}).get("hallucination_details")
        if not d:
            continue
        for k in verdict_totals:
            verdict_totals[k] += int(d.get(k, 0) or 0)
    total_claims = sum(verdict_totals.values())
    if total_claims:
        bad = verdict_totals["contradicted"] + verdict_totals["unsupported"]
        if bad:
            insights.append(
                f"Of {total_claims} atomic claims judged, {verdict_totals['contradicted']} were contradicted "
                f"and {verdict_totals['unsupported']} were unsupported."
            )

    if not insights:
        insights.append("All scored signals are within expected ranges.")

    return {
        "health": {
            "score": current,
            "tier": tier,
            "tier_label": tier_label,
            "flagged_pct": flagged_pct,
            "flagged_count": len(flagged),
            "scored_count": n,
            "baseline": baseline,
            "current": current,
            "delta": delta,
        },
        "insights": insights,
    }


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def _render_html(data: dict[str, Any], source: str) -> str:
    payload = json.dumps(data, default=str)
    src = html.escape(source)
    return _TEMPLATE.replace("__DATA_JSON__", payload).replace("__SOURCE__", src)


# ---------------------------------------------------------------------------
# HTML / CSS / JS template (single string so the dashboard is one file).
# Tabbed structure: Overview · Traces · Quality · Diagnose · Help.
# A persistent filter bar sits above the tab panes. Hash routing keeps the
# active tab in the URL so links are shareable. Side panel slides in from
# the right when a trace row is clicked.
# ---------------------------------------------------------------------------
_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>peekr · observability</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #0d1117; --bg2: #161b22; --bg3: #21262d; --bg4: #2d333b;
    --border: #30363d; --border2: #444c56;
    --text: #e6edf3; --text2: #c9d1d9; --muted: #8b949e;
    --accent: #58a6ff; --accent-soft: rgba(88,166,255,0.12);
    --green: #3fb950; --red: #f85149; --orange: #ffa657;
    --yellow: #e3b341; --purple: #bc8cff;
    --shadow-lg: 0 10px 30px rgba(0,0,0,0.45);
    --shadow-md: 0 4px 14px rgba(0,0,0,0.3);
  }
  body { background: var(--bg); color: var(--text); font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
  canvas { display: block; }
  code { font-family: "SFMono-Regular", Consolas, monospace; }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }

  /* ───────── App shell ───────── */
  .app { min-height: 100vh; }
  .topbar { position: sticky; top: 0; z-index: 30; background: rgba(13,17,23,0.92); backdrop-filter: blur(10px); border-bottom: 1px solid var(--border); }
  .topbar-row { max-width: 1320px; margin: 0 auto; padding: 0.85rem 1.5rem; display: flex; align-items: center; gap: 1rem; }
  .brand { font-weight: 800; letter-spacing: -0.02em; font-size: 1.05rem; }
  .brand span { color: var(--accent); }
  .source-tag { color: var(--muted); font-size: 0.78rem; padding: 0.2rem 0.55rem; background: var(--bg2); border: 1px solid var(--border); border-radius: 6px; font-family: "SFMono-Regular", Consolas, monospace; max-width: 320px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .topbar-spacer { flex: 1; }
  .topbar-meta { font-size: 0.78rem; color: var(--muted); }
  .topbar-meta strong { color: var(--text); font-weight: 600; }

  /* ───────── Tab nav ───────── */
  .tab-nav { max-width: 1320px; margin: 0 auto; padding: 0 1.5rem; display: flex; gap: 0.15rem; align-items: stretch; }
  .tab-btn { padding: 0.85rem 1.1rem; background: transparent; border: none; cursor: pointer; color: var(--muted); font-size: 0.92rem; font-weight: 500; position: relative; font-family: inherit; display: flex; align-items: center; gap: 0.45rem; transition: color 0.12s; }
  .tab-btn:hover { color: var(--text); }
  .tab-btn .tb-icon { font-size: 0.95rem; opacity: 0.85; }
  .tab-btn.active { color: var(--text); font-weight: 600; }
  .tab-btn.active::after { content: ""; position: absolute; left: 0; right: 0; bottom: -1px; height: 2px; background: var(--accent); border-radius: 2px; }
  .tab-btn .tb-count { font-size: 0.7rem; padding: 0.05rem 0.4rem; background: var(--bg3); color: var(--muted); border-radius: 999px; }

  /* ───────── Filter bar ───────── */
  .filter-bar { max-width: 1320px; margin: 0 auto; padding: 0.6rem 1.5rem; display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; border-bottom: 1px solid var(--border); background: var(--bg); }
  .filter-group { display: inline-flex; flex-wrap: wrap; gap: 0.3rem; align-items: center; }
  .filter-label { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin-right: 0.3rem; padding-right: 0.4rem; border-right: 1px solid var(--border); }
  .filter-group:first-child .filter-label { border-right: none; padding-right: 0; }
  .chip { font-size: 0.78rem; padding: 0.25rem 0.7rem; border-radius: 999px; background: var(--bg2); border: 1px solid var(--border); color: var(--muted); cursor: pointer; transition: all 0.12s; user-select: none; }
  .chip:hover { color: var(--text); border-color: var(--border2); }
  .chip.active { background: var(--accent); color: #0d1117; border-color: var(--accent); font-weight: 600; }
  .chip-reset { color: var(--accent); background: transparent; border: 1px dashed var(--accent); margin-left: auto; }
  .chip-reset:hover { background: var(--accent-soft); }
  .filter-count { font-size: 0.78rem; color: var(--muted); margin-left: 0.5rem; white-space: nowrap; }
  .custom-range-row { display: inline-flex; gap: 0.4rem; align-items: center; margin-left: 0.3rem; }
  .custom-range-row input { background: var(--bg3); color: var(--text); border: 1px solid var(--border); border-radius: 6px; padding: 0.2rem 0.45rem; font-size: 0.75rem; color-scheme: dark; font-family: inherit; }

  /* ───────── Main area ───────── */
  main { max-width: 1320px; margin: 0 auto; padding: 1.25rem 1.5rem 5rem; }
  .tab-pane { display: none; animation: fadeIn 0.18s ease; }
  .tab-pane.active { display: block; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: none; } }

  h1, h2, h3, h4 { letter-spacing: -0.01em; }
  h2.section { font-size: 1.05rem; font-weight: 700; margin-bottom: 0.2rem; display: flex; align-items: center; gap: 0.5rem; }
  h2.section .help { margin-left: 0; }
  p.hint { color: var(--muted); font-size: 0.84rem; margin-bottom: 1rem; line-height: 1.6; }
  p.hint code { background: var(--bg3); padding: 0.1em 0.35em; border-radius: 3px; font-size: 0.85em; }

  /* Help / info badge */
  .help { display: inline-block; margin-left: 0.4em; width: 15px; height: 15px; border-radius: 50%;
          background: var(--bg3); color: var(--muted); font-size: 0.62rem; line-height: 15px;
          text-align: center; cursor: help; position: relative; font-weight: 700; vertical-align: 0.15em; }
  .help:hover { background: var(--accent); color: var(--bg); }
  .help[data-tip]:hover::after { content: attr(data-tip); position: absolute; bottom: 150%; left: 50%;
          transform: translateX(-50%); background: var(--bg4); color: var(--text); padding: 0.6rem 0.85rem;
          border-radius: 6px; font-size: 0.78rem; white-space: pre-wrap; width: 290px;
          box-shadow: var(--shadow-md); z-index: 100; font-weight: 400; text-align: left;
          line-height: 1.55; border: 1px solid var(--border2); pointer-events: none; }
  .help[data-tip]:hover::before { content: ""; position: absolute; bottom: 145%; left: 50%;
          transform: translateX(-50%); border: 6px solid transparent; border-top-color: var(--bg4); z-index: 100; }

  /* ───────── Hero / health ───────── */
  .hero { display: grid; grid-template-columns: minmax(340px, 1.1fr) 1.6fr; gap: 1.1rem; background: var(--bg2); border: 1px solid var(--border); border-radius: 14px; padding: 1.65rem; margin-bottom: 1.1rem; box-shadow: var(--shadow-md); }
  .hero-left { display: flex; gap: 1.35rem; align-items: center; }
  .health-dot { width: 70px; height: 70px; border-radius: 50%; flex-shrink: 0; position: relative; box-shadow: 0 0 0 8px rgba(255,255,255,0.03); }
  .health-dot.good     { background: radial-gradient(circle at 30% 30%, #6fdc8c, var(--green)); }
  .health-dot.ok       { background: radial-gradient(circle at 30% 30%, #f0d674, var(--yellow)); }
  .health-dot.warning  { background: radial-gradient(circle at 30% 30%, #ffbe7a, var(--orange)); }
  .health-dot.critical { background: radial-gradient(circle at 30% 30%, #ff7c75, var(--red)); animation: pulse 1.8s infinite; }
  @keyframes pulse { 0%,100% { box-shadow: 0 0 0 8px rgba(248,81,73,0.15); } 50% { box-shadow: 0 0 0 14px rgba(248,81,73,0.05); } }
  .hero-text .hero-label { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.25rem; }
  .hero-text .hero-value { font-size: 2.5rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1; margin-bottom: 0.35rem; }
  .hero-text .hero-value small { font-size: 1.1rem; font-weight: 600; color: var(--muted); }
  .hero-text .hero-tier { font-size: 0.88rem; font-weight: 600; }
  .hero-text .hero-tier.good     { color: var(--green); }
  .hero-text .hero-tier.ok       { color: var(--yellow); }
  .hero-text .hero-tier.warning  { color: var(--orange); }
  .hero-text .hero-tier.critical { color: var(--red); }
  .hero-text .hero-sub { color: var(--muted); font-size: 0.82rem; margin-top: 0.55rem; line-height: 1.6; }
  .hero-sub .down { color: var(--red); } .hero-sub .up { color: var(--green); }
  .hero-right { display: flex; flex-direction: column; }
  .hero-spark-label { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.4rem; }
  .hero-spark-meta { color: var(--text); font-weight: 600; font-size: 0.78rem; letter-spacing: 0; text-transform: none; }
  .hero-spark { position: relative; height: 110px; width: 100%; }

  /* ───────── Narrative ───────── */
  .narrative { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 1.15rem 1.5rem; margin-bottom: 1.25rem; }
  .narrative h2 { font-size: 0.74rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.55rem; }
  .narrative ul { list-style: none; padding: 0; margin: 0; }
  .narrative li { padding: 0.32rem 0; padding-left: 1.4rem; position: relative; font-size: 0.92rem; line-height: 1.55; }
  .narrative li::before { content: "›"; position: absolute; left: 0.3rem; color: var(--accent); font-weight: 700; font-size: 1.1em; line-height: 1; top: 0.5rem; }
  .narrative li strong { color: var(--text); font-weight: 600; }

  /* ───────── Metric strip ───────── */
  .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 0.85rem; margin-bottom: 1.5rem; }
  .metric { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 1.05rem 1.15rem; transition: border-color 0.12s; }
  .metric:hover { border-color: var(--border2); }
  .metric-label { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.4rem; }
  .metric-row { display: flex; align-items: baseline; gap: 0.55rem; }
  .metric-value { font-size: 1.7rem; font-weight: 700; letter-spacing: -0.01em; }
  .metric-value.good { color: var(--green); } .metric-value.ok { color: var(--text); }
  .metric-value.warning { color: var(--orange); } .metric-value.critical { color: var(--red); }
  .metric-delta { font-size: 0.78rem; font-weight: 600; }
  .metric-delta.up { color: var(--green); } .metric-delta.down { color: var(--red); } .metric-delta.flat { color: var(--muted); }
  .metric-spark-wrap { position: relative; height: 42px; width: 100%; margin-top: 0.45rem; }
  .metric-foot { font-size: 0.72rem; color: var(--muted); margin-top: 0.2rem; }
  .metric-action { font-size: 0.74rem; color: var(--accent); margin-top: 0.35rem; line-height: 1.45; font-weight: 500; }
  .metric-action.warn { color: var(--orange); }
  .metric-action.bad  { color: var(--red); }
  .metric-action.ok   { color: var(--green); }
  .metric-action.flat { color: var(--muted); font-weight: 400; }

  /* ───────── Section card (generic) ───────── */
  section.panel { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 1.25rem 1.5rem; margin-bottom: 1.25rem; }
  .panel-head { display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap; margin-bottom: 0.85rem; }
  .panel-head .panel-actions { display: inline-flex; gap: 0.4rem; align-items: center; font-size: 0.78rem; color: var(--muted); }

  /* Chart sizing — fixed parent heights so Chart.js measures stably. */
  .chart-wrap { position: relative; height: 320px; overflow: hidden; }
  .chart-wrap.tall { height: 380px; }
  .chart-wrap.short { height: 240px; }

  /* ───────── Two-column layouts on quality tab ───────── */
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; margin-bottom: 1.25rem; }
  @media (max-width: 980px) { .two-col { grid-template-columns: 1fr; } }

  /* ───────── Heatmap ───────── */
  .heatmap-group { margin-bottom: 1.5rem; }
  .heatmap-group:last-child { margin-bottom: 0; }
  .heatmap-title { font-size: 0.74rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.55rem; }
  .heatmap { display: grid; gap: 2px; align-items: stretch; }
  .heatmap-row-label { font-size: 0.78rem; color: var(--text); padding: 0.45rem 0.6rem 0.45rem 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 240px; }
  .heatmap-row-label code { background: var(--bg3); padding: 0.1em 0.4em; border-radius: 4px; font-family: "SFMono-Regular", Consolas, monospace; font-size: 0.82em; }
  .heatmap-row-n { color: var(--muted); font-size: 0.68rem; margin-left: 0.35rem; }
  .heatmap-bucket-label { font-size: 0.68rem; color: var(--muted); text-align: center; padding-bottom: 0.45rem; }
  .heatmap-cell { padding: 0.55rem 0; border-radius: 4px; text-align: center; font-size: 0.78rem; font-weight: 600; cursor: pointer; transition: transform 0.12s; position: relative; }
  .heatmap-cell:hover { transform: scale(1.05); z-index: 1; box-shadow: 0 0 0 2px var(--text); }
  .heatmap-cell.empty { background: var(--bg3); color: var(--muted); }
  .heatmap-legend { display: flex; align-items: center; gap: 0.4rem; margin-top: 0.7rem; font-size: 0.72rem; color: var(--muted); }
  .heatmap-legend .swatch { display: inline-block; width: 14px; height: 14px; border-radius: 3px; }

  /* ───────── Recommendations ───────── */
  .recs-list { display: flex; flex-direction: column; gap: 0.85rem; }
  .rec { background: var(--bg3); border: 1px solid var(--border); border-left: 4px solid var(--muted); border-radius: 8px; padding: 0.95rem 1.1rem; transition: border-color 0.15s; }
  .rec:hover { border-color: var(--border2); }
  .rec.high { border-left-color: var(--red); }
  .rec.medium { border-left-color: var(--orange); }
  .rec.low { border-left-color: var(--yellow); }
  .rec.info { border-left-color: var(--accent); }
  .rec.good { border-left-color: var(--green); }
  .rec-head { display: flex; align-items: center; gap: 0.55rem; margin-bottom: 0.45rem; }
  .rec-sev { font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 700; padding: 0.12em 0.5em; border-radius: 4px; }
  .rec-sev.high   { background: rgba(248,81,73,0.18);  color: var(--red); }
  .rec-sev.medium { background: rgba(255,166,87,0.18); color: var(--orange); }
  .rec-sev.low    { background: rgba(227,179,65,0.18); color: var(--yellow); }
  .rec-sev.info   { background: rgba(88,166,255,0.18); color: var(--accent); }
  .rec-sev.good   { background: rgba(63,185,80,0.18);  color: var(--green); }
  .rec-title { font-weight: 700; font-size: 0.95rem; }
  .rec-cause { font-size: 0.86rem; color: var(--text2); line-height: 1.6; margin-bottom: 0.6rem; }
  .rec-cause strong { color: var(--text); }
  .rec-checks { font-size: 0.84rem; line-height: 1.65; }
  .rec-checks .checks-label { font-weight: 600; color: var(--text); margin-bottom: 0.25rem; }
  .rec-checks ol { margin: 0; padding-left: 1.4rem; color: var(--text2); }
  .rec-checks li { margin-bottom: 0.3rem; }
  .rec-checks code { background: var(--bg2); padding: 0.1em 0.35em; border-radius: 3px; font-family: "SFMono-Regular", Consolas, monospace; font-size: 0.88em; }
  .rec-evidence { font-size: 0.78rem; color: var(--muted); margin-top: 0.55rem; padding-top: 0.55rem; border-top: 1px dashed var(--border); }

  /* ───────── Action items (compact on Overview) ───────── */
  .action-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 0.85rem; margin-bottom: 1.25rem; }
  .action-card { background: var(--bg2); border: 1px solid var(--border); border-left: 3px solid var(--accent); border-radius: 10px; padding: 1rem 1.15rem; cursor: pointer; transition: border-color 0.12s; }
  .action-card:hover { border-color: var(--accent); }
  .action-card.bad  { border-left-color: var(--red); }
  .action-card.warn { border-left-color: var(--orange); }
  .action-card.ok   { border-left-color: var(--green); }
  .action-card .ac-title { font-weight: 700; font-size: 0.92rem; margin-bottom: 0.35rem; }
  .action-card .ac-body { font-size: 0.83rem; color: var(--text2); line-height: 1.55; }
  .action-card .ac-cta { font-size: 0.76rem; color: var(--accent); margin-top: 0.5rem; font-weight: 500; }

  /* ───────── Traces table ───────── */
  .traces-controls { display: flex; gap: 0.6rem; align-items: center; margin-bottom: 1rem; flex-wrap: wrap; }
  .search-box { flex: 1; min-width: 240px; position: relative; }
  .search-box input { width: 100%; background: var(--bg2); color: var(--text); border: 1px solid var(--border); border-radius: 8px; padding: 0.55rem 0.9rem 0.55rem 2.2rem; font-size: 0.92rem; font-family: inherit; }
  .search-box input:focus { outline: none; border-color: var(--accent); background: var(--bg3); }
  .search-box::before { content: "🔍"; position: absolute; left: 0.7rem; top: 50%; transform: translateY(-50%); font-size: 0.85rem; opacity: 0.6; }
  .traces-meta { font-size: 0.78rem; color: var(--muted); white-space: nowrap; }
  .table-wrap { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
  table.traces { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  table.traces thead th { background: var(--bg3); color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: 0.7rem; letter-spacing: 0.06em; padding: 0.65rem 0.9rem; text-align: left; border-bottom: 1px solid var(--border); cursor: pointer; user-select: none; white-space: nowrap; }
  table.traces thead th:hover { color: var(--text); background: var(--bg4); }
  table.traces thead th .sort { opacity: 0.4; margin-left: 0.3em; font-size: 0.85em; }
  table.traces thead th.sorted .sort { opacity: 1; color: var(--accent); }
  table.traces tbody tr { border-bottom: 1px solid var(--border); cursor: pointer; transition: background 0.1s; }
  table.traces tbody tr:hover { background: var(--bg3); }
  table.traces tbody tr.active { background: rgba(88,166,255,0.08); }
  table.traces tbody td { padding: 0.55rem 0.9rem; vertical-align: middle; }
  table.traces tbody td code { font-size: 0.82em; color: var(--accent); }
  .score-pill { display: inline-flex; align-items: center; gap: 0.35em; padding: 0.1em 0.55em; border-radius: 999px; font-weight: 700; font-size: 0.82em; }
  .score-pill.good     { background: rgba(63,185,80,0.18); color: var(--green); }
  .score-pill.ok       { background: rgba(227,179,65,0.18); color: var(--yellow); }
  .score-pill.warning  { background: rgba(255,166,87,0.18); color: var(--orange); }
  .score-pill.critical { background: rgba(248,81,73,0.18); color: var(--red); }
  .score-pill.na       { background: var(--bg3); color: var(--muted); }
  .tag { display: inline-block; background: var(--bg3); border: 1px solid var(--border); padding: 0.08em 0.55em; border-radius: 4px; font-family: "SFMono-Regular", Consolas, monospace; font-size: 0.78em; color: var(--text2); }
  .tag.err { color: var(--red); border-color: rgba(248,81,73,0.5); }
  .empty-state { padding: 3rem 1.5rem; text-align: center; color: var(--muted); }
  .empty-state .es-title { font-weight: 600; color: var(--text); font-size: 1rem; margin-bottom: 0.4rem; }
  .empty-state .es-body  { font-size: 0.88rem; max-width: 520px; margin: 0 auto; line-height: 1.6; }
  .empty-state code { background: var(--bg3); padding: 0.1em 0.4em; border-radius: 4px; font-size: 0.88em; }

  /* ───────── Side panel (trace detail) ───────── */
  .side-panel { position: fixed; top: 0; right: 0; height: 100vh; width: min(640px, 100vw); background: var(--bg2); border-left: 1px solid var(--border); box-shadow: var(--shadow-lg); transform: translateX(100%); transition: transform 0.22s ease; z-index: 50; overflow-y: auto; }
  .side-panel.open { transform: translateX(0); }
  .side-panel-head { position: sticky; top: 0; background: var(--bg2); padding: 1rem 1.25rem; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; z-index: 1; }
  .side-panel-title { font-weight: 700; font-size: 1rem; }
  .side-panel-close { background: none; border: 1px solid var(--border); color: var(--muted); border-radius: 6px; padding: 0.3rem 0.6rem; cursor: pointer; font-size: 0.85rem; font-family: inherit; }
  .side-panel-close:hover { color: var(--text); border-color: var(--border2); }
  .side-panel-body { padding: 1.25rem; }
  .side-panel-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.4); opacity: 0; pointer-events: none; transition: opacity 0.22s; z-index: 49; }
  .side-panel-overlay.open { opacity: 1; pointer-events: auto; }

  /* ───────── Detail content (used by side panel + Diagnose offenders) ───────── */
  .detail-meta { display: flex; gap: 0.4rem; flex-wrap: wrap; margin-bottom: 0.75rem; }
  .detail-q { color: var(--muted); font-size: 0.85rem; margin-bottom: 0.85rem; padding: 0.7rem 0.85rem; background: var(--bg3); border-radius: 8px; }
  .detail-q strong { color: var(--text); font-weight: 600; margin-right: 0.4rem; }
  .text-panel { background: var(--bg3); border: 1px solid var(--border); border-radius: 8px; padding: 0.7rem 0.85rem; margin-bottom: 0.75rem; }
  .text-panel-label { font-size: 0.66rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.4rem; display: flex; justify-content: space-between; }
  .text-panel-content { font-size: 0.83rem; line-height: 1.6; word-break: break-word; }
  .text-panel-content .mark-bad   { background: rgba(248,81,73,0.25); color: #ffd0cc; padding: 0 2px; border-radius: 2px; }
  .text-panel-content .mark-warn  { background: rgba(255,166,87,0.22); color: #ffe4c8; padding: 0 2px; border-radius: 2px; }
  .text-panel-content .mark-good  { background: rgba(63,185,80,0.22); color: #c5f0c9; padding: 0 2px; border-radius: 2px; }
  .claims-list, .citations-list { display: flex; flex-direction: column; gap: 0.4rem; margin-top: 0.55rem; }
  .claim { display: flex; align-items: flex-start; gap: 0.55rem; padding: 0.5rem 0.7rem; background: var(--bg3); border-radius: 6px; border-left: 3px solid var(--border); font-size: 0.85rem; }
  .claim.supported    { border-left-color: var(--green); }
  .claim.contradicted { border-left-color: var(--red); }
  .claim.unsupported  { border-left-color: var(--orange); }
  .claim .v { display: inline-block; width: 96px; flex-shrink: 0; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 700; }
  .claim.supported    .v { color: var(--green); }
  .claim.contradicted .v { color: var(--red); }
  .claim.unsupported  .v { color: var(--orange); }
  .citation { display: flex; gap: 0.6rem; padding: 0.4rem 0.65rem; border-radius: 4px; font-size: 0.82rem; align-items: center; }
  .citation.invented { background: rgba(248,81,73,0.1); }
  .citation.grounded { background: rgba(63,185,80,0.08); }
  .citation .kind { font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); width: 86px; }
  .citation code { font-size: 0.85em; }
  .citation .badge { font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 700; margin-left: auto; }
  .citation.invented .badge { color: var(--red); }
  .citation.grounded .badge { color: var(--green); }
  .judge-error { margin: 0.75rem 0; padding: 0.75rem 0.9rem; background: rgba(248,81,73,0.08); border: 1px solid rgba(248,81,73,0.35); border-left: 3px solid var(--red); border-radius: 8px; font-size: 0.84rem; line-height: 1.55; color: #ffd0cc; }
  .judge-error-label { font-size: 0.66rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--red); margin-bottom: 0.4rem; }
  .judge-error-row { font-family: "SFMono-Regular", Consolas, monospace; font-size: 0.78rem; padding: 0.15rem 0; }
  .judge-error-row code { background: rgba(248,81,73,0.18); padding: 0.05em 0.3em; border-radius: 3px; }
  .judge-error-hint { margin-top: 0.5rem; color: var(--muted); font-size: 0.78rem; font-family: inherit; }
  .judge-error-hint code { background: var(--bg2); padding: 0.1em 0.35em; border-radius: 3px; font-family: "SFMono-Regular", Consolas, monospace; font-size: 0.85em; }
  .span-actions { margin: 0.85rem 0 0.3rem; padding: 0.75rem 0.9rem; background: rgba(88,166,255,0.06); border: 1px solid rgba(88,166,255,0.25); border-left: 3px solid var(--accent); border-radius: 8px; }
  .span-actions-label { font-size: 0.66rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--accent); margin-bottom: 0.45rem; }
  .span-actions ul { list-style: none; padding: 0; margin: 0; }
  .span-actions li { padding: 0.3rem 0; font-size: 0.83rem; line-height: 1.5; color: var(--text2); }
  .span-actions li strong { color: var(--text); }
  .span-actions li code { background: var(--bg2); padding: 0.1em 0.35em; border-radius: 3px; font-family: "SFMono-Regular", Consolas, monospace; font-size: 0.88em; }
  .span-actions .why { color: var(--muted); font-size: 0.76rem; margin-top: 0.1rem; display: block; }

  /* ───────── Diagnose offender cards ───────── */
  .offender-list { display: flex; flex-direction: column; gap: 1rem; }
  .offender { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; transition: border-color 0.15s; }
  .offender:hover { border-color: var(--border2); }
  .offender-head { display: flex; justify-content: space-between; align-items: center; padding: 0.75rem 1.1rem; background: var(--bg3); border-bottom: 1px solid var(--border); flex-wrap: wrap; gap: 0.5rem; }
  .offender-rank { display: inline-block; background: var(--bg2); border-radius: 50%; width: 26px; height: 26px; text-align: center; font-size: 0.74rem; font-weight: 700; line-height: 26px; color: var(--muted); }
  .offender-body { padding: 1.1rem; }
  .offender-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-bottom: 0.85rem; }
  @media (max-width: 760px) { .offender-grid { grid-template-columns: 1fr; } }

  /* ───────── Help tab ───────── */
  .help-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; margin-bottom: 1.5rem; }
  @media (max-width: 980px) { .help-grid { grid-template-columns: 1fr; } }
  .help-card { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 1.25rem 1.5rem; }
  .help-card h3 { font-size: 1rem; font-weight: 700; margin-bottom: 0.7rem; display: flex; align-items: center; gap: 0.45rem; }
  .help-card h3 .emo { font-size: 1.15em; }
  .help-card p, .help-card li { font-size: 0.9rem; line-height: 1.6; color: var(--text2); margin-bottom: 0.55rem; }
  .help-card ol, .help-card ul { padding-left: 1.3rem; margin-bottom: 0.5rem; }
  .help-card code { background: var(--bg3); padding: 0.1em 0.4em; border-radius: 4px; font-size: 0.88em; }
  .help-card pre { background: var(--bg3); border: 1px solid var(--border); border-radius: 8px; padding: 0.85rem; overflow-x: auto; font-size: 0.82rem; line-height: 1.55; margin: 0.6rem 0 0.85rem; }
  .help-card pre code { background: none; padding: 0; }
  .checklist li { list-style: none; padding-left: 1.6rem; position: relative; }
  .checklist li::before { content: "☐"; position: absolute; left: 0; color: var(--muted); font-size: 1em; }
  .checklist li.done::before { content: "✓"; color: var(--green); }
  .checklist li.done { color: var(--muted); text-decoration: line-through; }
  dl.glossary { display: grid; grid-template-columns: max-content 1fr; gap: 0.45rem 1rem; font-size: 0.88rem; }
  dl.glossary dt { font-weight: 600; color: var(--text); }
  dl.glossary dd { color: var(--text2); line-height: 1.5; }

  /* ───────── Generic empties + responsive ───────── */
  .empty { color: var(--muted); padding: 2rem 1rem; text-align: center; font-style: italic; font-size: 0.92rem; }
  .empty code { background: var(--bg3); padding: 0.1em 0.4em; border-radius: 4px; font-style: normal; font-family: "SFMono-Regular", Consolas, monospace; }
  @media (max-width: 900px) {
    .topbar-row { padding: 0.7rem 1rem; }
    .tab-nav, .filter-bar, main { padding-left: 1rem; padding-right: 1rem; }
    .hero { grid-template-columns: 1fr; gap: 1.25rem; }
    .source-tag { max-width: 180px; }
    .topbar-meta { display: none; }
    .tab-btn { padding: 0.7rem 0.75rem; font-size: 0.85rem; }
    .tab-btn .tb-icon { display: none; }
  }
</style>
</head>
<body>

<div class="app">

  <!-- Topbar -->
  <header class="topbar">
    <div class="topbar-row">
      <div class="brand">peek<span>r</span></div>
      <div class="source-tag" title="__SOURCE__">__SOURCE__</div>
      <div class="topbar-spacer"></div>
      <div class="topbar-meta" id="topbar-meta"></div>
    </div>
    <nav class="tab-nav" id="tab-nav">
      <button class="tab-btn active" data-tab="overview"><span class="tb-icon">●</span>Overview</button>
      <button class="tab-btn" data-tab="traces"><span class="tb-icon">🔍</span>Traces<span class="tb-count" id="tc-traces"></span></button>
      <button class="tab-btn" data-tab="quality"><span class="tb-icon">📈</span>Quality</button>
      <button class="tab-btn" data-tab="diagnose"><span class="tb-icon">🩺</span>Diagnose<span class="tb-count" id="tc-recs"></span></button>
      <button class="tab-btn" data-tab="help"><span class="tb-icon">?</span>Help</button>
    </nav>
  </header>

  <!-- Persistent filter bar -->
  <div class="filter-bar" id="filter-bar"></div>

  <main>

    <!-- ─────────────────────────── OVERVIEW ─────────────────────────── -->
    <section class="tab-pane active" data-tab="overview">
      <div class="hero" id="hero"></div>
      <div class="narrative">
        <h2>What's happening</h2>
        <ul id="narrative-list"></ul>
      </div>
      <div class="metrics" id="metrics-row"></div>
      <h2 class="section">Top action items<span class="help" data-tip="The three most impactful suggestions for the current filter, drawn from the diagnostic engine. Click one to jump to the full Diagnose tab.">?</span></h2>
      <p class="hint">If you do one thing today based on this view, do something on this list.</p>
      <div class="action-cards" id="action-cards"></div>
    </section>

    <!-- ─────────────────────────── TRACES ─────────────────────────── -->
    <section class="tab-pane" data-tab="traces">
      <div class="traces-controls">
        <div class="search-box"><input type="text" id="trace-search" placeholder="Search by trace ID, model, content, error…" autocomplete="off"></div>
        <span class="traces-meta" id="traces-meta"></span>
      </div>
      <div class="table-wrap">
        <table class="traces" id="traces-table">
          <thead>
            <tr>
              <th data-sort="ts">Time<span class="sort">▾</span></th>
              <th data-sort="trace_id">Trace<span class="sort"></span></th>
              <th data-sort="model">Model<span class="sort"></span></th>
              <th data-sort="tenant">Tenant<span class="sort"></span></th>
              <th data-sort="endpoint">Endpoint<span class="sort"></span></th>
              <th data-sort="Hallucination" style="text-align:center">Halluc.<span class="sort"></span></th>
              <th data-sort="tokens" style="text-align:right">Tokens<span class="sort"></span></th>
              <th data-sort="duration_ms" style="text-align:right">ms<span class="sort"></span></th>
              <th data-sort="status">Status<span class="sort"></span></th>
            </tr>
          </thead>
          <tbody id="traces-tbody"></tbody>
        </table>
        <div id="traces-empty"></div>
      </div>
    </section>

    <!-- ─────────────────────────── QUALITY ─────────────────────────── -->
    <section class="tab-pane" data-tab="quality">
      <section class="panel">
        <div class="panel-head"><h2 class="section">Score over time<span class="help" data-tip="Rolling 20-call mean for each evaluator. Dashed orange line = warning (0.7). Dashed red = critical (0.5). Hover a point to see the trace; click to drill in.">?</span></h2></div>
        <p class="hint">Hallucination tends to be the bellwether — when it drops below the orange line, the other signals usually follow.</p>
        <div class="chart-wrap tall"><canvas id="rolling-chart"></canvas></div>
      </section>

      <section class="panel">
        <div class="panel-head"><h2 class="section">Failure breakdown by channel &amp; time<span class="help" data-tip="Rows are your models, tenants, and endpoints. Columns are time buckets, oldest on the left. Cell colour = mean Hallucination. Red = hallucinating, green = grounded. Click a cell to filter.">?</span></h2></div>
        <p class="hint">A red <em>row</em> tells you <strong>which</strong> channel is failing. A row that goes green → red tells you <strong>when</strong> it started — usually right after a deploy, an index update, or a model swap.</p>
        <div id="heatmaps"></div>
        <div class="heatmap-legend">
          <span>0.0</span>
          <span class="swatch" style="background:#f85149"></span>
          <span class="swatch" style="background:#ffa657"></span>
          <span class="swatch" style="background:#e3b341"></span>
          <span class="swatch" style="background:#6fdc8c"></span>
          <span class="swatch" style="background:#3fb950"></span>
          <span>1.0</span>
          <span style="margin-left:1rem">Grey = no data in that bucket</span>
        </div>
      </section>

      <div class="two-col">
        <section class="panel">
          <h2 class="section">Score distribution<span class="help" data-tip="Histogram of Hallucination scores across 10 equal-width buckets. A right-leaning distribution (mass near 1.0) is healthy; mass piled at 0.0 means a lot of fully-ungrounded answers.">?</span></h2>
          <div class="chart-wrap short"><canvas id="dist-chart"></canvas></div>
        </section>
        <section class="panel">
          <h2 class="section">Claim verdict totals<span class="help" data-tip="Across all spans where you used detailed (RAGAS-style) mode, the breakdown of supported / contradicted / unsupported claims.">?</span></h2>
          <div class="chart-wrap short"><canvas id="verdict-chart"></canvas></div>
        </section>
      </div>

      <section class="panel">
        <div class="panel-head"><h2 class="section">Citation accuracy<span class="help" data-tip="Pure-regex evaluator that checks if URLs, arXiv IDs, paper titles, and statute numbers in outputs actually appear in the source context. Zero LLM calls — a free, complementary signal to Hallucination.">?</span></h2></div>
        <div id="citation-card"></div>
      </section>
    </section>

    <!-- ─────────────────────────── DIAGNOSE ─────────────────────────── -->
    <section class="tab-pane" data-tab="diagnose">
      <section class="panel">
        <div class="panel-head"><h2 class="section">Likely causes &amp; next steps<span class="help" data-tip="The diagnostic engine inspects your filtered traces and emits suggestions for common RAG / memory failures. Each card has a severity badge and a 'what to try' list with concrete commands and prompt changes.">?</span></h2></div>
        <p class="hint">These are hypotheses, not certainties. Pair them with the worst-offender cards below to confirm. The list updates as you change filters, so toggle a tenant or model chip to compare healthy vs degraded views.</p>
        <div class="recs-list" id="recommendations"></div>
      </section>

      <section class="panel">
        <div class="panel-head"><h2 class="section">Worst offenders — what hallucinated<span class="help" data-tip="The 12 lowest-scoring calls in your current filter. Each card shows the question, the source context, the model's answer with claim-level highlights, the per-claim verdicts (detailed mode), and any invented citations.">?</span></h2></div>
        <p class="hint">Red highlights = claims contradicted by context. Orange = unsupported. Green = supported. Each card ends with "What to try for this call" — fixes specific to that span's failure pattern.</p>
        <div class="offender-list" id="offender-list"></div>
      </section>
    </section>

    <!-- ─────────────────────────── HELP ─────────────────────────── -->
    <section class="tab-pane" data-tab="help">
      <h2 class="section">Get the most out of this dashboard</h2>
      <p class="hint">Five panels you can mostly read in any order — start with <strong>Setup checklist</strong> if it's your first time.</p>
      <div class="help-grid">
        <div class="help-card">
          <h3><span class="emo">🚀</span>Setup checklist</h3>
          <p class="hint" style="margin-bottom:0.6rem">peekr captures spans automatically once instrumented. To get the most out of <em>this</em> dashboard, make sure the following are in place:</p>
          <ul class="checklist" id="setup-checklist"></ul>
        </div>

        <div class="help-card">
          <h3><span class="emo">📚</span>Glossary</h3>
          <dl class="glossary">
            <dt>Hallucination score</dt><dd>0 = fully ungrounded, 1 = every factual claim is supported by the context. LLM-as-judge.</dd>
            <dt>Detailed mode</dt><dd>RAGAS-style: judge decomposes the output into atomic claims and verdicts each as <code>supported</code> / <code>contradicted</code> / <code>unsupported</code>.</dd>
            <dt>Rubric</dt><dd>Custom LLM-as-judge with a criterion you define ("be concise", "cite sources", etc.). Score 0-1.</dd>
            <dt>Citation accuracy</dt><dd>Pure-regex evaluator. Extracts URLs/arXiv/DOI/<em>Author et al. YEAR</em>/<em>§ N</em> from the output and checks if each appears in the context.</dd>
            <dt>Baseline vs current</dt><dd>peekr splits the spans in your filter into thirds by time. Baseline = oldest third, current = newest. A negative delta = score is worse than it used to be.</dd>
            <dt>n / scored</dt><dd>Number of calls with a score for that metric. Less than the total span count because some spans (empty context, tool-call outputs, judge crashes) are skipped.</dd>
            <dt>peekr.internal</dt><dd>Spans created by the eval judge itself. Hidden from the dashboard but kept in the JSONL so you can audit judge token cost.</dd>
            <dt>eval_errors</dt><dd>When a judge call fails (no API key, rate limit), peekr records the error here <strong>instead</strong> of writing a misleading 0.0 score. Surfaced as a red banner on offender cards.</dd>
          </dl>
        </div>

        <div class="help-card">
          <h3><span class="emo">🎯</span>Configuring evaluators</h3>
          <p>The minimum useful configuration is a single line:</p>
          <pre><code>import peekr
peekr.instrument(evaluators=[
    peekr.eval.Hallucination(),
])</code></pre>
          <p>For RAG, attach the retrieved docs so the judge has the source of truth:</p>
          <pre><code>peekr.eval.Hallucination(
    context_extractor=lambda span:
        span.attributes.get("retrieved_docs", "")
)</code></pre>
          <p>Detailed (per-claim) mode is one flag away:</p>
          <pre><code>peekr.eval.Hallucination(detailed=True)</code></pre>
          <p>If both <code>openai</code> and <code>anthropic</code> are installed but only one has credentials, force the working one:</p>
          <pre><code>peekr.eval.Hallucination(judge_provider="anthropic")</code></pre>
        </div>

        <div class="help-card">
          <h3><span class="emo">🛠</span>Troubleshooting</h3>
          <p><strong>"Every span shows 0.0"</strong> — check the Diagnose tab. If you see a "Judge unavailable" card, set <code>OPENAI_API_KEY</code> or <code>ANTHROPIC_API_KEY</code>, or pass <code>judge_provider=</code> explicitly. peekr 0.3.1+ shows judge crashes as <em>eval_errors</em>, not 0.0 — older traces may need to be regenerated.</p>
          <p><strong>"Heatmap has no endpoint row"</strong> — your spans need <code>attributes.endpoint</code> set. Tag your code with <code>get_current_span().attributes["endpoint"] = request.path</code> or do it in middleware.</p>
          <p><strong>"Citations flag real product names as invented"</strong> — fixed in 0.3.1. quoted-title pattern now requires a citation preamble (<em>in/from/titled/cited 'X'</em>). Tool-use outputs (<code>ToolUseBlock</code>) skip citation eval entirely.</p>
          <p><strong>"Dashboard is empty"</strong> — likely no spans match the current filter. Click <em>Clear filters</em> at the top. If your JSONL has rows but no LLM spans, verify <code>peekr.instrument()</code> ran before your LLM calls.</p>
        </div>

        <div class="help-card">
          <h3><span class="emo">🔗</span>Where to learn more</h3>
          <ul>
            <li><a href="https://ashwanijha04.github.io/peekr/docs.html" target="_blank" rel="noopener">Full peekr documentation</a></li>
            <li><a href="https://github.com/ashwanijha04/peekr" target="_blank" rel="noopener">peekr on GitHub</a></li>
            <li><a href="https://pypi.org/project/peekr/" target="_blank" rel="noopener">peekr on PyPI</a></li>
          </ul>
          <p style="margin-top:0.8rem">Found a bug or have a feature request? <a href="https://github.com/ashwanijha04/peekr/issues" target="_blank" rel="noopener">Open an issue</a>.</p>
        </div>

        <div class="help-card">
          <h3><span class="emo">⌨️</span>Keyboard shortcuts</h3>
          <dl class="glossary">
            <dt><code>1</code> – <code>5</code></dt><dd>Switch tabs</dd>
            <dt><code>/</code></dt><dd>Focus the trace search box</dd>
            <dt><code>Esc</code></dt><dd>Close the trace detail panel</dd>
            <dt><code>R</code></dt><dd>Clear all filters</dd>
          </dl>
        </div>
      </div>
    </section>

  </main>
</div>

<!-- Side panel (trace detail) -->
<div class="side-panel-overlay" id="side-overlay"></div>
<aside class="side-panel" id="side-panel" role="dialog" aria-label="Trace detail">
  <div class="side-panel-head">
    <div class="side-panel-title" id="side-title">Trace</div>
    <button class="side-panel-close" id="side-close" title="Close (Esc)">Close</button>
  </div>
  <div class="side-panel-body" id="side-body"></div>
</aside>

<script>
const DATA = __DATA_JSON__;

// ═══════════════════════════════════════════════════════════════════
//  Filter state, time ranges, applyFilter
// ═══════════════════════════════════════════════════════════════════
const FILTER = {
  tenant: null, model: null, endpoint: null,
  time: "all", customFrom: null, customTo: null,
};
const TIME_RANGES = [
  { key: "all", label: "All time", seconds: null },
  { key: "5m",  label: "5 min",    seconds: 5 * 60 },
  { key: "15m", label: "15 min",   seconds: 15 * 60 },
  { key: "30m", label: "30 min",   seconds: 30 * 60 },
  { key: "1h",  label: "1h",       seconds: 60 * 60 },
  { key: "24h", label: "24h",      seconds: 24 * 60 * 60 },
  { key: "7d",  label: "7d",       seconds: 7 * 24 * 60 * 60 },
  { key: "30d", label: "30d",      seconds: 30 * 24 * 60 * 60 },
];
function timeCutoff() {
  if (FILTER.time === "custom") return 0;
  const r = TIME_RANGES.find(x => x.key === FILTER.time);
  if (!r || !r.seconds) return 0;
  const maxTs = DATA.rows.reduce((m, r) => Math.max(m, r.ts || 0), 0);
  return maxTs - r.seconds;
}
function applyFilter(rows) {
  const cutoff = timeCutoff();
  const customFrom = FILTER.time === "custom" ? FILTER.customFrom : null;
  const customTo   = FILTER.time === "custom" ? FILTER.customTo   : null;
  return rows.filter(r => {
    if (FILTER.tenant   && r.tenant   !== FILTER.tenant)   return false;
    if (FILTER.model    && r.model    !== FILTER.model)    return false;
    if (FILTER.endpoint && r.endpoint !== FILTER.endpoint) return false;
    if (cutoff && (r.ts || 0) < cutoff) return false;
    if (customFrom != null && (r.ts || 0) < customFrom) return false;
    if (customTo   != null && (r.ts || 0) > customTo)   return false;
    return true;
  });
}

// Trace search query
let SEARCH_QUERY = "";
let SORT_KEY = "ts";
let SORT_DIR = -1; // -1 = desc, 1 = asc

function rerender() {
  const filtered = applyFilter(DATA.rows);
  // Overview
  renderHero(filtered);
  renderNarrative(filtered);
  renderMetrics(filtered);
  renderActionCards(filtered);
  // Traces tab
  renderTracesTable(filtered);
  // Quality tab
  renderRollingChart(filtered);
  renderHeatmaps(filtered);
  renderDistChart(filtered);
  renderVerdictChart(filtered);
  renderCitationPanel(filtered);
  // Diagnose tab
  renderRecommendations(filtered);
  renderOffenders(filtered);
  // Topbar meta + chips state
  renderTopbarMeta(filtered);
  updateChips();
  const fc = document.getElementById("filter-count");
  if (fc) fc.textContent = filtered.length === DATA.rows.length
    ? `${filtered.length} spans` : `${filtered.length} of ${DATA.rows.length} spans`;
  const tcTraces = document.getElementById("tc-traces");
  if (tcTraces) tcTraces.textContent = String(filtered.length);
  // Setup checklist (recomputed on each render so it tracks live state)
  renderSetupChecklist();
}

// ═══════════════════════════════════════════════════════════════════
//  Utilities
// ═══════════════════════════════════════════════════════════════════
function fmt(v, d=2) { return v == null ? "—" : (+v).toFixed(d); }
function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}
function tierForScore(v) {
  if (v == null) return "ok";
  if (v >= 0.85) return "good";
  if (v >= 0.70) return "ok";
  if (v >= 0.50) return "warning";
  return "critical";
}
function deltaArrow(d) {
  if (d == null) return "·";
  if (d >  0.02) return "▲";
  if (d < -0.02) return "▼";
  return "·";
}
function tsLabel(t, opts = {}) {
  if (!t) return "";
  const d = new Date(t * 1000);
  if (opts.timeOnly) return d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  return d.toLocaleString();
}
function scoreColor(v) {
  if (v == null) return null;
  const c = Math.max(0, Math.min(1, v));
  let r, g, b;
  if (c < 0.5) {
    const t = c / 0.5;
    r = Math.round(248 + (227-248)*t); g = Math.round(81 + (179-81)*t); b = Math.round(73 + (65-73)*t);
  } else {
    const t = (c - 0.5) / 0.5;
    r = Math.round(227 + (63-227)*t); g = Math.round(179 + (185-179)*t); b = Math.round(65 + (80-65)*t);
  }
  return `rgb(${r},${g},${b})`;
}
function debounce(fn, ms) {
  let t; return function(...args) { clearTimeout(t); t = setTimeout(() => fn.apply(this, args), ms); };
}

// ═══════════════════════════════════════════════════════════════════
//  Filter bar + tab navigation
// ═══════════════════════════════════════════════════════════════════
function renderFilterBar() {
  const wrap = document.getElementById("filter-bar");
  const ch = DATA.channels || {};
  let html = '';
  for (const [label, vals] of Object.entries(ch)) {
    if (!vals || !vals.length) continue;
    const chips = vals.map(v => {
      const active = FILTER[label] === v ? "active" : "";
      return `<span class="chip ${active}" data-key="${label}" data-val="${esc(v)}">${esc(v)}</span>`;
    }).join("");
    html += `<div class="filter-group"><span class="filter-label">${label}</span>${chips}</div>`;
  }
  const timeChips = TIME_RANGES.map(r => {
    const active = FILTER.time === r.key ? "active" : "";
    return `<span class="chip ${active}" data-time="${r.key}">${esc(r.label)}</span>`;
  }).join("");
  const customActive = FILTER.time === "custom" ? "active" : "";
  html += `<div class="filter-group"><span class="filter-label">When</span>${timeChips}` +
          `<span class="chip ${customActive}" data-time="custom">Custom…</span>` +
          `<span class="custom-range-row" id="custom-range-wrap" style="display:${FILTER.time === "custom" ? "inline-flex" : "none"}">` +
            `<input type="datetime-local" id="custom-from" value="${esc(localDtValue(FILTER.customFrom))}" />` +
            `<span class="filter-count" style="margin:0">to</span>` +
            `<input type="datetime-local" id="custom-to"   value="${esc(localDtValue(FILTER.customTo))}"   />` +
          `</span></div>`;
  html += `<span class="chip chip-reset" id="chip-reset">Clear filters</span>`;
  html += `<span class="filter-count" id="filter-count"></span>`;
  wrap.innerHTML = html;

  wrap.querySelectorAll(".chip[data-key]").forEach(c => c.addEventListener("click", () => {
    const k = c.dataset.key, v = c.dataset.val;
    FILTER[k] = FILTER[k] === v ? null : v;
    rerender();
  }));
  wrap.querySelectorAll(".chip[data-time]").forEach(c => c.addEventListener("click", () => {
    FILTER.time = c.dataset.time;
    if (FILTER.time === "custom" && FILTER.customFrom == null && FILTER.customTo == null && DATA.rows.length) {
      const tsList = DATA.rows.map(x => x.ts || 0).filter(Boolean);
      FILTER.customTo = Math.max(...tsList);
      FILTER.customFrom = FILTER.customTo - 3600;
    }
    rerender();
  }));
  const fromInput = document.getElementById("custom-from");
  const toInput = document.getElementById("custom-to");
  if (fromInput) fromInput.addEventListener("change", () => { FILTER.customFrom = parseLocalDt(fromInput.value); rerender(); });
  if (toInput)   toInput.addEventListener("change",   () => { FILTER.customTo   = parseLocalDt(toInput.value);   rerender(); });
  document.getElementById("chip-reset").addEventListener("click", clearFilters);
}
function clearFilters() {
  FILTER.tenant = FILTER.model = FILTER.endpoint = null;
  FILTER.time = "all";
  FILTER.customFrom = FILTER.customTo = null;
  rerender();
}
function updateChips() {
  document.querySelectorAll(".chip[data-key]").forEach(c => {
    c.classList.toggle("active", FILTER[c.dataset.key] === c.dataset.val);
  });
  document.querySelectorAll(".chip[data-time]").forEach(c => {
    c.classList.toggle("active", FILTER.time === c.dataset.time);
  });
}
function localDtValue(unixSec) {
  if (unixSec == null) return "";
  const d = new Date(unixSec * 1000);
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}
function parseLocalDt(str) {
  if (!str) return null;
  const d = new Date(str);
  return isNaN(d.getTime()) ? null : d.getTime() / 1000;
}

function switchTab(name) {
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab-pane").forEach(p => p.classList.toggle("active", p.dataset.tab === name));
  if (location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
  // Some charts need a re-render on first activation (their canvases were 0×0
  // while the pane was display:none).
  if (name === "quality") {
    renderRollingChart(applyFilter(DATA.rows));
    renderDistChart(applyFilter(DATA.rows));
    renderVerdictChart(applyFilter(DATA.rows));
  }
  if (name === "traces") {
    const inp = document.getElementById("trace-search");
    if (inp) inp.focus();
  }
}
function setupTabNav() {
  document.querySelectorAll(".tab-btn").forEach(b => b.addEventListener("click", () => switchTab(b.dataset.tab)));
  const initial = (location.hash || "#overview").slice(1);
  if (["overview","traces","quality","diagnose","help"].includes(initial)) switchTab(initial);
}

// ═══════════════════════════════════════════════════════════════════
//  Aggregations
// ═══════════════════════════════════════════════════════════════════
function metricStats(rows, key) {
  const vals = rows.map(r => r[key]).filter(v => v != null);
  if (!vals.length) return null;
  const n = vals.length;
  const k = Math.max(1, Math.floor(n / 3));
  const sorted = [...rows].sort((a, b) => a.ts - b.ts).filter(r => r[key] != null);
  const baseline = sorted.slice(0, k).reduce((s, r) => s + r[key], 0) / Math.min(k, sorted.length);
  const current  = sorted.slice(-k).reduce((s, r) => s + r[key], 0) / Math.min(k, sorted.length);
  return { mean: vals.reduce((s,v)=>s+v,0)/n, n, baseline, current, delta: current - baseline };
}
function rollingMean(arr, w) {
  const out = []; const buf = [];
  for (const v of arr) {
    if (v != null) { buf.push(v); if (buf.length > w) buf.shift(); }
    out.push(buf.length ? buf.reduce((s,x)=>s+x,0)/buf.length : null);
  }
  return out;
}

// ═══════════════════════════════════════════════════════════════════
//  Topbar meta line
// ═══════════════════════════════════════════════════════════════════
function renderTopbarMeta(rows) {
  const meta = document.getElementById("topbar-meta");
  if (!meta) return;
  const scored = rows.filter(r => r.Hallucination != null).length;
  const errs = rows.filter(r => r.status === "error").length;
  meta.innerHTML = `<strong>${rows.length}</strong> spans · <strong>${scored}</strong> scored · <strong>${errs}</strong> errors`;
}

// ═══════════════════════════════════════════════════════════════════
//  HERO
// ═══════════════════════════════════════════════════════════════════
let _heroChart = null;
function renderHero(rows) {
  const stats = metricStats(rows, "Hallucination");
  const wrap = document.getElementById("hero");
  if (!stats) {
    wrap.innerHTML = `<div class="hero-left"><div class="health-dot ok"></div>
      <div class="hero-text"><div class="hero-label">Hallucination health</div>
      <div class="hero-value">—</div><div class="hero-tier">no eval scores yet</div>
      <div class="hero-sub">Add <code>peekr.eval.Hallucination()</code> to your evaluators — see the <strong>Help</strong> tab.</div></div></div>`;
    if (_heroChart) { _heroChart.destroy(); _heroChart = null; }
    return;
  }
  const tier = tierForScore(stats.current);
  const tierLabel = ({good:"healthy", ok:"watch", warning:"needs attention", critical:"regressing"})[tier];
  const flagged = rows.filter(r => r.Hallucination != null && r.Hallucination < 0.5).length;
  const dpct = stats.delta * 100;
  const dWord = stats.delta < -0.02
    ? `<span class="down">↓ ${Math.abs(dpct).toFixed(0)} pts vs baseline ${stats.baseline.toFixed(2)}</span>`
    : stats.delta >  0.02
      ? `<span class="up">↑ ${dpct.toFixed(0)} pts vs baseline ${stats.baseline.toFixed(2)}</span>`
      : `<span>flat vs baseline ${stats.baseline.toFixed(2)}</span>`;

  wrap.innerHTML = `
    <div class="hero-left">
      <div class="health-dot ${tier}"></div>
      <div class="hero-text">
        <div class="hero-label">Hallucination health</div>
        <div class="hero-value">${(stats.current * 100).toFixed(0)}<small>/100</small></div>
        <div class="hero-tier ${tier}">${tierLabel}</div>
        <div class="hero-sub">
          ${flagged} of ${stats.n} scored calls flagged (score &lt; 0.5).<br/>
          ${dWord}
        </div>
      </div>
    </div>
    <div class="hero-right">
      <div class="hero-spark-label">
        <span>Score trend (rolling)</span>
        <span class="hero-spark-meta">${rows.length} spans · ${stats.n} scored</span>
      </div>
      <div class="hero-spark"><canvas id="hero-spark-canvas"></canvas></div>
    </div>`;

  const series = [...rows].sort((a,b)=>a.ts-b.ts).map(r => r.Hallucination);
  const rolling = rollingMean(series, 10);
  if (_heroChart) { _heroChart.destroy(); _heroChart = null; }
  requestAnimationFrame(() => {
    const ctx = document.getElementById("hero-spark-canvas");
    if (!ctx) return;
    _heroChart = new Chart(ctx, {
      type: "line",
      data: { labels: rolling.map((_,i)=>i+1),
              datasets: [{
                data: rolling, borderColor: "#58a6ff", borderWidth: 2,
                backgroundColor: c => {
                  const cv = c.chart.ctx; if (!cv) return "rgba(88,166,255,0.15)";
                  const g = cv.createLinearGradient(0,0,0,cv.canvas.height);
                  g.addColorStop(0,"rgba(88,166,255,0.4)"); g.addColorStop(1,"rgba(88,166,255,0.02)");
                  return g;
                },
                fill: true, tension: 0.3, pointRadius: 0, spanGaps: true,
              }] },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false, resizeDelay: 100,
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        scales: { y: { min: 0, max: 1, display: false }, x: { display: false } },
      },
    });
  });
}

// ═══════════════════════════════════════════════════════════════════
//  NARRATIVE (filtered, re-computed)
// ═══════════════════════════════════════════════════════════════════
function renderNarrative(rows) {
  const ul = document.getElementById("narrative-list");
  const insights = [];
  const stats = metricStats(rows, "Hallucination");
  if (stats && stats.n >= 6) {
    if (stats.delta < -0.1) insights.push(`Hallucination dropped <strong>${(Math.abs(stats.delta)*100).toFixed(0)} points</strong> from baseline (<strong>${stats.baseline.toFixed(2)}</strong> → <strong>${stats.current.toFixed(2)}</strong>) in the current view.`);
    else if (stats.delta > 0.1) insights.push(`Hallucination improved <strong>${(stats.delta*100).toFixed(0)} points</strong> from baseline (<strong>${stats.baseline.toFixed(2)}</strong> → <strong>${stats.current.toFixed(2)}</strong>).`);
  }
  const combos = {};
  for (const r of rows) {
    if (r.Hallucination == null) continue;
    const key = `${r.model || "?"} · ${r.tenant || "?"} · ${r.endpoint || "?"}`;
    (combos[key] = combos[key] || []).push(r.Hallucination);
  }
  let worst = null;
  for (const [k, vs] of Object.entries(combos)) {
    if (vs.length < 3) continue;
    const m = vs.reduce((s,v)=>s+v,0)/vs.length;
    if (!worst || m < worst.mean) worst = { key: k, mean: m, n: vs.length };
  }
  if (worst) insights.push(`Worst channel: <strong>${esc(worst.key)}</strong> — mean <strong>${worst.mean.toFixed(2)}</strong> across <strong>${worst.n}</strong> calls.`);

  let citTotal = 0, citInv = 0;
  for (const r of rows) { const d = r.citation_details; if (d) { citTotal += d.total || 0; citInv += d.invented || 0; } }
  if (citTotal) {
    if (citInv > 0) insights.push(`<strong>${citInv}</strong> of <strong>${citTotal}</strong> citation references were invented (<strong>${(citInv/citTotal*100).toFixed(0)}%</strong>) — URLs, arXiv IDs, or paper titles not in the source context.`);
    else            insights.push(`All <strong>${citTotal}</strong> citation references in the output were grounded in context.`);
  }
  let claimsTotal = 0, contra = 0, unsup = 0;
  for (const r of rows) { const d = r.details; if (d) { claimsTotal += d.total || 0; contra += d.contradicted || 0; unsup += d.unsupported || 0; } }
  if (claimsTotal) insights.push(`Of <strong>${claimsTotal}</strong> atomic claims judged in detailed mode, <strong>${contra}</strong> were contradicted by context and <strong>${unsup}</strong> were unsupported.`);

  const errorRows = rows.filter(r => r.eval_errors && Object.keys(r.eval_errors).length > 0);
  if (errorRows.length) insights.push(`<strong>${errorRows.length}</strong> spans have judge errors (judge unavailable). See the <strong>Diagnose</strong> tab for setup steps.`);

  const errors = rows.filter(r => r.status === "error").length;
  if (errors) insights.push(`<strong>${errors}</strong> LLM call${errors > 1 ? "s" : ""} failed outright (rate-limit, timeout, etc.).`);

  if (!insights.length) insights.push(`All scored signals are within expected ranges for the current filter.`);
  ul.innerHTML = insights.map(s => `<li>${s}</li>`).join("");
}

// ═══════════════════════════════════════════════════════════════════
//  METRIC STRIP + action hints
// ═══════════════════════════════════════════════════════════════════
const METRICS = [
  { key: "Hallucination",    label: "Hallucination", color: "#58a6ff",
    tip: "Mean fraction of the model's factual claims grounded in its source context. 0 = fully hallucinating, 1 = fully grounded." },
  { key: "Rubric",           label: "Rubric",        color: "#bc8cff",
    tip: "Custom rubric score from your LLM-as-judge evaluators. 0-1; what 'good' means depends on the rubric criteria you set." },
  { key: "CitationAccuracy", label: "Citations",     color: "#3fb950",
    tip: "Fraction of citation patterns in the output (URLs, arXiv IDs, paper titles) that appear in the source context. Pure regex — fast and free." },
];
function metricAction(key, s, rows) {
  if (key === "Hallucination") {
    const flagged = rows.filter(r => r.Hallucination != null && r.Hallucination < 0.5).length;
    if (flagged >= 10)        return { text: `→ ${flagged} calls flagged — review worst offenders`, cls: "bad" };
    if (flagged > 0)          return { text: `→ ${flagged} flagged — open the Diagnose tab`, cls: "warn" };
    if (s && s.delta < -0.1)  return { text: `→ regressing — check the heatmap`, cls: "warn" };
    if (s && s.delta > 0.1)   return { text: `→ improving vs baseline`, cls: "ok" };
    return { text: `→ stable, no action needed`, cls: "flat" };
  }
  if (key === "CitationAccuracy") {
    let inv = 0; for (const r of rows) if (r.citation_details) inv += r.citation_details.invented || 0;
    if (inv >= 5) return { text: `→ ${inv} invented refs — likely retrieval miss`, cls: "bad" };
    if (inv > 0)  return { text: `→ ${inv} invented ref${inv>1?"s":""} in offenders`, cls: "warn" };
    if (s && s.n > 0) return { text: `→ all detected references are grounded`, cls: "ok" };
    return { text: `→ no citations detected in outputs`, cls: "flat" };
  }
  if (key === "Rubric") {
    if (s && s.current < 0.6) return { text: `→ below 0.6 — review rubric and outputs`, cls: "warn" };
    if (s && s.delta < -0.1)  return { text: `→ regressing — output quality dropping`, cls: "warn" };
    return { text: `→ within expected range`, cls: "flat" };
  }
  return { text: "", cls: "flat" };
}
const _metricCharts = {};
function renderMetrics(rows) {
  const wrap = document.getElementById("metrics-row");
  let html = "";
  for (const m of METRICS) {
    const s = metricStats(rows, m.key);
    const tier = s ? tierForScore(s.current) : "ok";
    const value = s ? s.current.toFixed(2) : "—";
    const deltaCls = s ? (s.delta > 0.02 ? "up" : (s.delta < -0.02 ? "down" : "flat")) : "flat";
    const deltaTxt = s ? `${deltaArrow(s.delta)} ${s.delta >= 0 ? "+" : ""}${(s.delta*100).toFixed(0)} pts` : "";
    const nTxt = s ? `${s.n} scored` : "no data";
    const baseTxt = s ? `was ${s.baseline.toFixed(2)}` : "—";
    const action = s ? metricAction(m.key, s, rows) : { text: "", cls: "flat" };
    html += `
      <div class="metric">
        <div class="metric-label">${m.label}<span class="help" data-tip="${esc(m.tip)}">?</span></div>
        <div class="metric-row">
          <span class="metric-value ${tier}">${value}</span>
          <span class="metric-delta ${deltaCls}">${deltaTxt}</span>
        </div>
        <div class="metric-spark-wrap"><canvas id="spark-${m.key}"></canvas></div>
        <div class="metric-foot">${nTxt} · ${baseTxt}</div>
        ${action.text ? `<div class="metric-action ${action.cls}">${action.text}</div>` : ""}
      </div>`;
  }
  const err = rows.filter(r => r.status === "error").length;
  const tot = rows.length;
  const errTier = err === 0 ? "good" : (err > tot * 0.05 ? "critical" : "warning");
  const errAction = err === 0 ? { text: `→ no failed calls`, cls: "ok" }
                              : (err > tot * 0.05 ? { text: `→ error rate &gt; 5% — check rate limits`, cls: "bad" }
                                                  : { text: `→ a few transient failures`, cls: "warn" });
  html += `
    <div class="metric">
      <div class="metric-label">Errors<span class="help" data-tip="LLM calls that raised an exception (rate-limit, timeout, network). &gt;5% sustained usually means an upstream issue.">?</span></div>
      <div class="metric-row">
        <span class="metric-value ${errTier}">${err}</span>
        <span class="metric-delta flat">${tot ? ((err/tot)*100).toFixed(1) : 0}% of calls</span>
      </div>
      <div class="metric-foot" style="margin-top:0.95rem">of ${tot} total calls</div>
      ${errAction.text ? `<div class="metric-action ${errAction.cls}">${errAction.text}</div>` : ""}
    </div>`;
  wrap.innerHTML = html;

  for (const m of METRICS) {
    if (_metricCharts[m.key]) { _metricCharts[m.key].destroy(); _metricCharts[m.key] = null; }
  }
  requestAnimationFrame(() => {
    for (const m of METRICS) {
      const ctx = document.getElementById(`spark-${m.key}`);
      if (!ctx) continue;
      const ordered = [...rows].sort((a,b)=>a.ts-b.ts).map(r => r[m.key]);
      const roll = rollingMean(ordered, 10);
      _metricCharts[m.key] = new Chart(ctx, {
        type: "line",
        data: { labels: roll.map((_,i)=>i+1),
                datasets: [{ data: roll, borderColor: m.color, borderWidth: 2, fill: false, tension: 0.3, pointRadius: 0, spanGaps: true }] },
        options: { responsive: true, maintainAspectRatio: false, animation: false, resizeDelay: 100,
                   plugins: { legend: { display: false }, tooltip: { enabled: false } },
                   scales: { y: { min: 0, max: 1, display: false }, x: { display: false } } },
      });
    }
  });
}

// ═══════════════════════════════════════════════════════════════════
//  ACTION CARDS (top 3 recs, shown on Overview)
// ═══════════════════════════════════════════════════════════════════
function renderActionCards(rows) {
  const wrap = document.getElementById("action-cards");
  const recs = diagnoseRecommendations(rows).slice(0, 3);
  if (!recs.length) {
    wrap.innerHTML = `<div class="empty">No urgent actions — keep monitoring.</div>`;
    return;
  }
  wrap.innerHTML = recs.map(r => {
    const cls = ({high:"bad", medium:"warn", low:"warn", info:"", good:"ok"})[r.severity] || "";
    return `
      <div class="action-card ${cls}" data-jump="diagnose">
        <div class="ac-title">${r.title}</div>
        <div class="ac-body">${r.cause}</div>
        <div class="ac-cta">View full recommendation →</div>
      </div>`;
  }).join("");
  wrap.querySelectorAll(".action-card").forEach(c => c.addEventListener("click", () => switchTab(c.dataset.jump)));
}

// ═══════════════════════════════════════════════════════════════════
//  TRACES TABLE
// ═══════════════════════════════════════════════════════════════════
function tracesFiltered(rows) {
  if (!SEARCH_QUERY) return rows;
  const q = SEARCH_QUERY.toLowerCase();
  return rows.filter(r =>
    (r.trace_id || "").toLowerCase().includes(q) ||
    (r.model    || "").toLowerCase().includes(q) ||
    (r.tenant   || "").toLowerCase().includes(q) ||
    (r.endpoint || "").toLowerCase().includes(q) ||
    (r.output   || "").toLowerCase().includes(q) ||
    (r.input    || "").toLowerCase().includes(q) ||
    (r.error    || "").toLowerCase().includes(q)
  );
}
function renderTracesTable(rows) {
  const tbody = document.getElementById("traces-tbody");
  const meta  = document.getElementById("traces-meta");
  const empty = document.getElementById("traces-empty");
  if (!tbody) return;

  const filtered = tracesFiltered(rows);
  meta.textContent = filtered.length === rows.length
    ? `${filtered.length} spans`
    : `${filtered.length} of ${rows.length} match`;
  // Mark sort indicators
  document.querySelectorAll("table.traces thead th").forEach(th => {
    th.classList.toggle("sorted", th.dataset.sort === SORT_KEY);
    const arrow = th.querySelector(".sort");
    arrow.textContent = th.dataset.sort === SORT_KEY ? (SORT_DIR < 0 ? "▾" : "▴") : "";
  });

  const sorted = [...filtered].sort((a, b) => {
    const av = a[SORT_KEY], bv = b[SORT_KEY];
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    if (typeof av === "number" && typeof bv === "number") return (av - bv) * SORT_DIR;
    return String(av).localeCompare(String(bv)) * SORT_DIR;
  });

  if (!sorted.length) {
    tbody.innerHTML = "";
    empty.innerHTML = `<div class="empty-state">
      <div class="es-title">No spans match your filter${SEARCH_QUERY ? ' and search' : ''}.</div>
      <div class="es-body">${SEARCH_QUERY
        ? `Try clearing the search box or your filter chips.`
        : `Either no LLM calls were captured, or your filters exclude everything. Click <em>Clear filters</em> above.`}</div></div>`;
    return;
  }
  empty.innerHTML = "";

  const rowsHtml = sorted.slice(0, 500).map(r => {
    const tier = tierForScore(r.Hallucination);
    const hVal = r.Hallucination != null ? r.Hallucination.toFixed(2) : "—";
    const hCls = r.Hallucination != null ? tier : "na";
    return `<tr data-span="${esc(r.span_id || "")}">
      <td>${tsLabel(r.ts, { timeOnly: true })}</td>
      <td><code>${esc((r.trace_id || "").slice(0, 8))}</code></td>
      <td><span class="tag">${esc(r.model || "—")}</span></td>
      <td><span class="tag">${esc(r.tenant || "—")}</span></td>
      <td><span class="tag">${esc(r.endpoint || "—")}</span></td>
      <td style="text-align:center"><span class="score-pill ${hCls}">${hVal}</span></td>
      <td style="text-align:right;color:var(--muted)">${r.tokens || 0}</td>
      <td style="text-align:right;color:var(--muted)">${Math.round(r.duration_ms || 0)}</td>
      <td>${r.status === "error" ? `<span class="tag err">error</span>` : `<span class="tag">ok</span>`}</td>
    </tr>`;
  }).join("");
  tbody.innerHTML = rowsHtml + (sorted.length > 500
    ? `<tr><td colspan="9" style="text-align:center;color:var(--muted);padding:1rem">Showing first 500 of ${sorted.length}. Use filters or search to narrow.</td></tr>`
    : "");

  // Row click → open side panel
  tbody.querySelectorAll("tr[data-span]").forEach(tr => {
    tr.addEventListener("click", () => openSidePanel(tr.dataset.span));
  });
}
function setupTracesHeaders() {
  document.querySelectorAll("table.traces thead th").forEach(th => th.addEventListener("click", () => {
    const k = th.dataset.sort; if (!k) return;
    if (SORT_KEY === k) SORT_DIR = -SORT_DIR;
    else { SORT_KEY = k; SORT_DIR = (k === "ts" || k === "Hallucination" || k === "tokens" || k === "duration_ms") ? -1 : 1; }
    renderTracesTable(applyFilter(DATA.rows));
  }));
}
function setupTracesSearch() {
  const inp = document.getElementById("trace-search");
  if (!inp) return;
  inp.addEventListener("input", debounce(() => {
    SEARCH_QUERY = inp.value.trim();
    renderTracesTable(applyFilter(DATA.rows));
  }, 120));
}

// ═══════════════════════════════════════════════════════════════════
//  SIDE PANEL (trace detail)
// ═══════════════════════════════════════════════════════════════════
function openSidePanel(spanId) {
  const r = DATA.rows.find(x => x.span_id === spanId);
  if (!r) return;
  const sp = document.getElementById("side-panel");
  const ov = document.getElementById("side-overlay");
  document.getElementById("side-title").textContent = `${r.model || "—"} · ${tsLabel(r.ts)}`;
  document.getElementById("side-body").innerHTML = renderDetailHTML(r);
  // Mark active row
  document.querySelectorAll("table.traces tbody tr").forEach(tr => tr.classList.toggle("active", tr.dataset.span === spanId));
  sp.classList.add("open");
  ov.classList.add("open");
}
function closeSidePanel() {
  document.getElementById("side-panel").classList.remove("open");
  document.getElementById("side-overlay").classList.remove("open");
  document.querySelectorAll("table.traces tbody tr").forEach(tr => tr.classList.remove("active"));
}
function setupSidePanel() {
  document.getElementById("side-close").addEventListener("click", closeSidePanel);
  document.getElementById("side-overlay").addEventListener("click", closeSidePanel);
}
function renderDetailHTML(r) {
  const tier = tierForScore(r.Hallucination);
  const { context, question } = parseInput(r.input, r);
  const claims = r.details && r.details.claims ? r.details.claims : null;
  const answerHtml = claims ? highlightAll(r.output || "", claims) : esc(r.output || "");

  const meta = `<div class="detail-meta">
    <span class="score-pill ${r.Hallucination == null ? "na" : tier}">${r.Hallucination == null ? "no score" : r.Hallucination.toFixed(2)}</span>
    <span class="tag">${esc(r.model || "—")}</span>
    <span class="tag">${esc(r.tenant || "—")}</span>
    <span class="tag">${esc(r.endpoint || "—")}</span>
    ${r.status === "error" ? `<span class="tag err">error</span>` : ""}
    <span style="color:var(--muted);font-size:0.78rem;align-self:center">${tsLabel(r.ts)}</span>
  </div>`;

  let errorsHtml = "";
  const errs = r.eval_errors || {};
  const ek = Object.keys(errs);
  if (ek.length) {
    errorsHtml = `<div class="judge-error">
      <div class="judge-error-label">⚠ Judge unavailable for this span</div>
      ${ek.map(k => `<div class="judge-error-row"><code>${esc(k)}</code> — ${esc(errs[k])}</div>`).join("")}
      <div class="judge-error-hint">Score is missing, not zero. Set <code>OPENAI_API_KEY</code> / <code>ANTHROPIC_API_KEY</code>, or pass <code>judge_provider="anthropic"</code>.</div>
    </div>`;
  }

  let claimsHtml = "";
  if (claims && claims.length) {
    claimsHtml = `<div class="claims-list">` + claims.map(c =>
      `<div class="claim ${c.verdict}"><span class="v">${c.verdict}</span><span>${esc(c.text)}</span></div>`
    ).join("") + `</div>`;
  }

  let citationsHtml = "";
  const cd = r.citation_details;
  if (cd && cd.items && cd.items.length) {
    citationsHtml = `<div class="citations-list">` + cd.items.map(it =>
      `<div class="citation ${it.grounded ? "grounded" : "invented"}">
        <span class="kind">${esc(it.kind)}</span>
        <code>${esc(it.text)}</code>
        <span class="badge">${it.grounded ? "grounded" : "invented"}</span>
      </div>`
    ).join("") + `</div>`;
  }

  const actions = perSpanActions(r);
  let actionsHtml = "";
  if (actions.length) {
    actionsHtml = `<div class="span-actions">
      <div class="span-actions-label">What to try for this call</div>
      <ul>${actions.map(a => `<li>${a.fix}<span class="why">${a.why}</span></li>`).join("")}</ul>
    </div>`;
  }

  return meta
    + (question ? `<div class="detail-q"><strong>Q:</strong>${esc(question)}</div>` : "")
    + `<div class="text-panel"><div class="text-panel-label"><span>Source context</span><span style="color:var(--muted);font-weight:600">${context.length}c</span></div><div class="text-panel-content">${esc(context)}</div></div>`
    + `<div class="text-panel"><div class="text-panel-label"><span>Model answer</span><span style="color:var(--muted);font-weight:600">${(r.output||"").length}c</span></div><div class="text-panel-content">${answerHtml}</div></div>`
    + errorsHtml + claimsHtml + citationsHtml + actionsHtml;
}

// ═══════════════════════════════════════════════════════════════════
//  QUALITY: rolling chart, dist, verdict, heatmap, citation
// ═══════════════════════════════════════════════════════════════════
let _rollingChart = null;
function renderRollingChart(rows) {
  const canvas = document.getElementById("rolling-chart");
  if (!canvas) return;
  const sorted = [...rows].sort((a,b)=>a.ts-b.ts);
  const labels = sorted.map((_, i) => i + 1);
  const window = 20;
  const halRolling = rollingMean(sorted.map(r => r.Hallucination), window);
  const rubRolling = rollingMean(sorted.map(r => r.Rubric), window);
  const citRolling = rollingMean(sorted.map(r => r.CitationAccuracy), window);
  const warningLine  = labels.map(() => 0.7);
  const criticalLine = labels.map(() => 0.5);
  if (_rollingChart) { _rollingChart.destroy(); _rollingChart = null; }
  _rollingChart = new Chart(canvas, {
    type: "line",
    data: { labels, datasets: [
      { label: "Hallucination",    data: halRolling, borderColor: "#58a6ff", tension: 0.25, pointRadius: 0, borderWidth: 2.5, fill: false, spanGaps: true },
      { label: "Rubric",           data: rubRolling, borderColor: "#bc8cff", tension: 0.25, pointRadius: 0, borderWidth: 1.5, fill: false, spanGaps: true },
      { label: "CitationAccuracy", data: citRolling, borderColor: "#3fb950", tension: 0.25, pointRadius: 0, borderWidth: 1.5, fill: false, spanGaps: true },
      { label: "Warning (0.7)",    data: warningLine,  borderColor: "rgba(255,166,87,0.5)", borderDash: [4,4], borderWidth: 1, pointRadius: 0, fill: false },
      { label: "Critical (0.5)",   data: criticalLine, borderColor: "rgba(248,81,73,0.55)", borderDash: [4,4], borderWidth: 1, pointRadius: 0, fill: false },
    ]},
    options: {
      responsive: true, maintainAspectRatio: false, animation: false, resizeDelay: 100,
      interaction: { mode: "nearest", intersect: false, axis: "x" },
      onClick(_evt, els) {
        if (!els.length) return;
        const r = sorted[els[0].index];
        if (r) openSidePanel(r.span_id);
      },
      plugins: {
        legend: { labels: { color: "#e6edf3", font: { size: 12 } } },
        tooltip: {
          backgroundColor: "rgba(13,17,23,0.95)", borderColor: "#30363d", borderWidth: 1,
          titleColor: "#e6edf3", bodyColor: "#c9d1d9", padding: 10,
          callbacks: {
            title: items => { const r = sorted[items[0].dataIndex]; return `Span #${items[0].dataIndex + 1} · ${r && r.trace_id ? r.trace_id.slice(0,8) : ""}`; },
            label: item => {
              const r = sorted[item.dataIndex];
              const dsLabel = item.dataset.label;
              if (dsLabel.startsWith("Warning") || dsLabel.startsWith("Critical")) return null;
              return `${dsLabel}: ${fmt(item.parsed.y)} (raw ${fmt(r ? r[dsLabel] : null)})`;
            },
            afterBody: items => {
              const r = sorted[items[0].dataIndex];
              if (!r) return [];
              return ["", `model:    ${r.model || "—"}`, `tenant:   ${r.tenant || "—"}`, `endpoint: ${r.endpoint || "—"}`, `when:     ${tsLabel(r.ts)}`, "", "click to open trace detail"];
            },
          },
        },
      },
      scales: {
        y: { min: 0, max: 1, ticks: { color: "#8b949e" }, grid: { color: "#30363d" }, title: { display: true, text: "score (0 = hallucinating, 1 = grounded)", color: "#8b949e", font: { size: 11 } } },
        x: { ticks: { color: "#8b949e", maxTicksLimit: 14 }, grid: { color: "rgba(48,54,61,0.5)" }, title: { display: true, text: "span number (oldest → newest)", color: "#8b949e", font: { size: 11 } } },
      },
    },
  });
}

let _distChart = null;
function renderDistChart(rows) {
  const canvas = document.getElementById("dist-chart");
  if (!canvas) return;
  const buckets = Array(10).fill(0);
  for (const r of rows) {
    if (r.Hallucination == null) continue;
    const idx = Math.min(Math.floor(r.Hallucination * 10), 9);
    buckets[idx] += 1;
  }
  if (_distChart) { _distChart.destroy(); _distChart = null; }
  _distChart = new Chart(canvas, {
    type: "bar",
    data: { labels: Array.from({length: 10}, (_, i) => `${(i/10).toFixed(1)}-${((i+1)/10).toFixed(1)}`),
            datasets: [{ data: buckets, backgroundColor: buckets.map((_, i) => scoreColor(i/10 + 0.05)) }] },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false, resizeDelay: 100,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => `${c.parsed.y} span${c.parsed.y === 1 ? "" : "s"}` } } },
      scales: {
        y: { beginAtZero: true, ticks: { color: "#8b949e", precision: 0 }, grid: { color: "#30363d" }, title: { display: true, text: "spans", color: "#8b949e", font: { size: 11 } } },
        x: { ticks: { color: "#8b949e" }, grid: { display: false }, title: { display: true, text: "Hallucination score bucket", color: "#8b949e", font: { size: 11 } } },
      },
    },
  });
}

let _verdictChart = null;
function renderVerdictChart(rows) {
  const canvas = document.getElementById("verdict-chart");
  if (!canvas) return;
  let s = 0, c = 0, u = 0;
  for (const r of rows) { const d = r.details; if (d) { s += d.supported || 0; c += d.contradicted || 0; u += d.unsupported || 0; } }
  if (_verdictChart) { _verdictChart.destroy(); _verdictChart = null; }
  if (s + c + u === 0) {
    canvas.parentElement.innerHTML = `<div class="empty">No detailed-mode claims in this view. Enable with <code>peekr.eval.Hallucination(detailed=True)</code>.</div>`;
    return;
  }
  _verdictChart = new Chart(canvas, {
    type: "doughnut",
    data: { labels: ["supported", "contradicted", "unsupported"], datasets: [{ data: [s, c, u], backgroundColor: ["#3fb950", "#f85149", "#ffa657"], borderColor: "#161b22", borderWidth: 2 }] },
    options: { responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { position: "bottom", labels: { color: "#e6edf3" } } } },
  });
}

function renderHeatmaps(rows) {
  const wrap = document.getElementById("heatmaps");
  if (!wrap) return;
  const buckets = (DATA.channel_heatmap && DATA.channel_heatmap.buckets) || [];
  const nb = buckets.length;
  if (!nb || !rows.length) {
    wrap.innerHTML = '<div class="empty">No channel data in the current filter.</div>';
    return;
  }
  const tMin = buckets[0].start;
  const width = (buckets[nb-1].end - tMin) / nb;

  const fields = [["model","model"], ["tenant","tenant"], ["endpoint","endpoint"]];
  let html = "";
  for (const [field, label] of fields) {
    const bySeg = {};
    for (const r of rows) {
      const seg = r[field]; if (!seg || r.Hallucination == null) continue;
      const idx = width > 0 ? Math.min(Math.floor((r.ts - tMin) / width), nb - 1) : 0;
      (bySeg[seg] = bySeg[seg] || Array.from({length: nb}, () => [])) [idx].push(r.Hallucination);
    }
    const rowsArr = Object.entries(bySeg).map(([seg, bks]) => {
      const cells = bks.map(vs => vs.length ? { mean: vs.reduce((s,v)=>s+v,0)/vs.length, n: vs.length } : { mean: null, n: 0 });
      const tail = cells.slice(-2).map(c => c.mean).filter(x => x != null);
      return { seg, cells, n_total: bks.reduce((s,vs)=>s+vs.length,0), sortKey: tail.length ? Math.min(...tail) : 1.0 };
    });
    rowsArr.sort((a,b) => a.sortKey - b.sortKey);
    if (!rowsArr.length) {
      html += `<div class="heatmap-group"><div class="heatmap-title">${label}</div><div class="empty" style="padding:0.75rem">no data</div></div>`;
      continue;
    }
    const colDef = `220px repeat(${nb}, minmax(40px, 1fr))`;
    let inner = `<div class="heatmap" style="grid-template-columns:${colDef}">`;
    inner += `<div class="heatmap-row-label"></div>`;
    buckets.forEach((b, i) => {
      const lab = new Date(b.start * 1000).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
      const tip = `bucket ${i+1}/${nb}\n${new Date(b.start*1000).toLocaleString()}\nto ${new Date(b.end*1000).toLocaleString()}`;
      inner += `<div class="heatmap-bucket-label" title="${esc(tip)}">${lab}</div>`;
    });
    for (const row of rowsArr) {
      const active = FILTER[label] === row.seg;
      inner += `<div class="heatmap-row-label"${active ? ' style="background:rgba(88,166,255,0.06);border-radius:4px"' : ''}><code>${esc(row.seg)}</code><span class="heatmap-row-n">n=${row.n_total}</span></div>`;
      row.cells.forEach((c, i) => {
        if (c.mean == null) {
          inner += `<div class="heatmap-cell empty" title="${esc(row.seg + ' · bucket ' + (i+1) + ' — no calls')}"></div>`;
        } else {
          const bg = scoreColor(c.mean);
          const textColor = c.mean > 0.4 && c.mean < 0.75 ? "#1f2328" : (c.mean >= 0.75 ? "#0d2515" : "#fff");
          const tip = `${row.seg} · bucket ${i+1}/${nb}\nmean ${c.mean.toFixed(2)} across ${c.n} call${c.n>1?'s':''}\nclick to filter`;
          inner += `<div class="heatmap-cell" data-key="${label}" data-seg="${esc(row.seg)}" style="background:${bg};color:${textColor}" title="${esc(tip)}">${c.mean.toFixed(2)}<span style="display:block;font-size:0.62rem;font-weight:500;opacity:0.7;margin-top:1px">n=${c.n}</span></div>`;
        }
      });
    }
    inner += `</div>`;
    html += `<div class="heatmap-group"><div class="heatmap-title">${label}</div>${inner}</div>`;
  }
  wrap.innerHTML = html;
  wrap.querySelectorAll(".heatmap-cell[data-key]").forEach(cell => {
    cell.addEventListener("click", () => {
      const k = cell.dataset.key, seg = cell.dataset.seg;
      FILTER[k] = FILTER[k] === seg ? null : seg;
      rerender();
    });
  });
}

function renderCitationPanel(rows) {
  const wrap = document.getElementById("citation-card");
  if (!wrap) return;
  let total = 0, grounded = 0, invented = 0;
  for (const r of rows) { const d = r.citation_details; if (!d) continue; total += d.total || 0; grounded += d.grounded || 0; invented += d.invented || 0; }
  if (!total) {
    wrap.innerHTML = '<div class="empty">No citation patterns detected. Add <code>peekr.eval.CitationAccuracy()</code> to your evaluators.</div>';
    return;
  }
  const pct = (grounded / total * 100).toFixed(1);
  const cls = grounded / total >= 0.8 ? "good" : (grounded / total >= 0.5 ? "ok" : "critical");
  wrap.innerHTML = `
    <div class="metrics" style="margin:0">
      <div class="metric"><div class="metric-label">Citations seen</div><div class="metric-value">${total}</div></div>
      <div class="metric"><div class="metric-label">Grounded</div><div class="metric-value good">${grounded}</div></div>
      <div class="metric"><div class="metric-label">Invented</div><div class="metric-value critical">${invented}</div></div>
      <div class="metric"><div class="metric-label">% grounded</div><div class="metric-value ${cls}">${pct}%</div></div>
    </div>
    <p class="hint" style="margin-top:0.75rem;margin-bottom:0">Invented citations are URLs, arXiv IDs, paper titles, or statute numbers that appear in the model's output but not in the source context. They're a frequent failure mode in RAG when retrieval misses, and a strong "look here" signal independent of the LLM-as-judge.</p>`;
}

// ═══════════════════════════════════════════════════════════════════
//  DIAGNOSE: recommendations + offender cards
// ═══════════════════════════════════════════════════════════════════
function _channelConcentration(rows, field) {
  const flagged = rows.filter(r => r.Hallucination != null && r.Hallucination < 0.5 && r[field]);
  if (flagged.length < 4) return null;
  const counts = {};
  for (const r of flagged) counts[r[field]] = (counts[r[field]] || 0) + 1;
  const [val, n] = Object.entries(counts).sort((a,b) => b[1] - a[1])[0];
  return { value: val, share: n / flagged.length, count: n, total: flagged.length };
}
function _claimCounts(rows) {
  let total = 0, supported = 0, contradicted = 0, unsupported = 0;
  for (const r of rows) { const d = r.details; if (!d) continue; total += d.total || 0; supported += d.supported || 0; contradicted += d.contradicted || 0; unsupported += d.unsupported || 0; }
  return { total, supported, contradicted, unsupported };
}
function _citationCounts(rows) {
  let total = 0, grounded = 0, invented = 0;
  for (const r of rows) { const d = r.citation_details; if (!d) continue; total += d.total || 0; grounded += d.grounded || 0; invented += d.invented || 0; }
  return { total, grounded, invented };
}
function diagnoseRecommendations(rows) {
  const recs = [];
  const hStats = metricStats(rows, "Hallucination");
  const errors = rows.filter(r => r.status === "error").length;
  const cit = _citationCounts(rows);
  const claims = _claimCounts(rows);
  const flagged = rows.filter(r => r.Hallucination != null && r.Hallucination < 0.5).length;

  if (!hStats) {
    recs.push({
      severity: "info", title: "No Hallucination scores recorded yet",
      cause: "Either the <code>Hallucination</code> evaluator hasn't been added to your <code>peekr.instrument()</code> call, or the spans in this view didn't pass through it.",
      checks: [
        `Add the evaluator: <code>peekr.instrument(evaluators=[peekr.eval.Hallucination()])</code>`,
        `For RAG, attach the retrieved chunks via <code>context_extractor=lambda s: s.attributes.get("retrieved_docs", "")</code>`,
        `Enable detailed mode for per-claim verdicts: <code>peekr.eval.Hallucination(detailed=True)</code>`,
      ],
    });
    return recs;
  }

  const errorRows = rows.filter(r => r.eval_errors && Object.keys(r.eval_errors).length > 0);
  if (errorRows.length >= Math.max(3, rows.length * 0.1)) {
    const exemplars = new Set();
    for (const r of errorRows) { for (const v of Object.values(r.eval_errors)) { exemplars.add(String(v)); if (exemplars.size >= 2) break; } if (exemplars.size >= 2) break; }
    const exemplarList = [...exemplars].map(e => `<code>${esc(e)}</code>`).join(" · ");
    recs.push({
      severity: "high", title: `LLM judge unavailable for ${errorRows.length} of ${rows.length} spans`,
      cause: `The Hallucination/Rubric evaluator raised on these spans, most likely because the configured provider has no credentials. Spans show no score (not 0.0) — your dashboard scores are sampled from the spans where the judge succeeded, if any. Example errors: ${exemplarList || "(see span details)"}.`,
      checks: [
        `<strong>Set the API key</strong>: <code>export OPENAI_API_KEY=...</code> or <code>export ANTHROPIC_API_KEY=...</code>.`,
        `<strong>Force a provider explicitly</strong>: <code>peekr.eval.Hallucination(judge_provider="anthropic")</code>.`,
        `<strong>Verify with a smoke test:</strong> <code>python -c "from peekr.eval._judge import call_judge; print(call_judge('Return 1.0'))"</code>.`,
      ],
      evidence: `${errorRows.length} spans have eval_errors. Score distributions and drift figures exclude failed judges.`,
    });
  }

  if (cit.total >= 5 && cit.invented / cit.total > 0.3) {
    const rate = cit.invented / cit.total;
    recs.push({
      severity: "high", title: `Model is inventing citations (${(rate*100).toFixed(0)}% of detected references)`,
      cause: `Out of <strong>${cit.total}</strong> citation patterns we saw in outputs, <strong>${cit.invented}</strong> don't appear in the source context. The signature of a RAG flow where retrieval didn't return the actual source, so the model fabricates plausible-looking references.`,
      checks: [
        `<strong>Inspect retrieval:</strong> log returned chunks for a flagged span and confirm cited sources are in there.`,
        `<strong>Tighten the prompt:</strong> <em>"Cite only sources present in the context above."</em>`,
        `<strong>Verify citations post-hoc:</strong> extract URL/arXiv/DOI from output, reject if not in context.`,
        `<strong>Try hybrid retrieval</strong> (BM25 + dense) for keyword-heavy queries.`,
      ],
      evidence: `${cit.invented} invented / ${cit.total} total citations.`,
    });
  }

  if (claims.total >= 6 && claims.contradicted / claims.total > 0.2) {
    const rate = claims.contradicted / claims.total;
    recs.push({
      severity: rate > 0.4 ? "high" : "medium",
      title: `${(rate*100).toFixed(0)}% of claims directly contradict the context`,
      cause: `<strong>${claims.contradicted}</strong> of <strong>${claims.total}</strong> claims actively conflict with the context — model is ignoring or overriding context with training-data priors.`,
      checks: [
        `<strong>Strengthen the prompt:</strong> <em>"Use only the facts in the context. The context wins over what you 'know'."</em>`,
        `<strong>Move context closer to the question</strong> in the prompt (recency bias).`,
        `<strong>Reduce max_tokens / temperature</strong> — long completions drift.`,
        `<strong>Check chunk quality:</strong> try chunk_size 256–512 with 10-20% overlap.`,
      ],
      evidence: `${claims.contradicted} contradicted, ${claims.unsupported} unsupported, ${claims.supported} supported across ${claims.total} claims.`,
    });
  }

  if (claims.total >= 6 && claims.unsupported / claims.total > 0.25 && claims.contradicted / claims.total < 0.2) {
    const rate = claims.unsupported / claims.total;
    recs.push({
      severity: "medium",
      title: `${(rate*100).toFixed(0)}% of claims have no support in context`,
      cause: `<strong>${claims.unsupported}</strong> of <strong>${claims.total}</strong> claims aren't contradicted — the context is silent. The model elaborates beyond what was retrieved.`,
      checks: [
        `<strong>Add a refusal instruction:</strong> <em>"If the context doesn't contain the answer, reply 'I don't have that information.'"</em>`,
        `<strong>Check retrieval recall:</strong> are you missing a corpus?`,
        `<strong>Try a coverage prompt:</strong> ask the model to state which parts it can answer, then answer only those.`,
      ],
      evidence: `${claims.unsupported} unsupported claims with low contradiction — adding detail, not rewriting.`,
    });
  }

  for (const field of ["model", "tenant", "endpoint"]) {
    const conc = _channelConcentration(rows, field);
    if (conc && conc.share > 0.5 && conc.count >= 4) {
      const fieldLabel = field;
      const causeByField = {
        model:    `Most flagged calls came from <strong>one model</strong>. Either intrinsically weaker at grounding, or recently swapped in for this traffic.`,
        tenant:   `Most flagged calls belong to <strong>one tenant</strong>. Their data may be missing from your retrieval index, their system prompt may differ, or their query distribution doesn't match what your RAG was tuned for.`,
        endpoint: `Most flagged calls came from <strong>one endpoint</strong>. Code changes — prompt template, retrieval params, top-k, model override — are the most likely cause.`,
      };
      const checksByField = {
        model: [`Compare hallucination rate per model in the Quality heatmap.`, `Route hard queries to a larger model.`, `Check max_tokens / temperature differs from healthier model's config.`],
        tenant: [`Verify this tenant's documents are in the retrieval index.`, `Diff this tenant's system prompt vs a healthy tenant's.`, `Check for tenant-specific overrides degrading quality.`],
        endpoint: [`Diff last week of deploys touching this endpoint.`, `Compare this endpoint's prompt with a healthier one.`, `Check if top-k or chunk overlap was tweaked recently.`],
      };
      recs.push({
        severity: conc.share > 0.75 ? "high" : "medium",
        title: `Failures concentrated on ${fieldLabel} = ${conc.value}`,
        cause: causeByField[field],
        checks: checksByField[field],
        evidence: `${conc.count} of ${conc.total} flagged calls (${(conc.share*100).toFixed(0)}%).`,
      });
      break;
    }
  }

  if (hStats.n >= 6 && hStats.delta < -0.1) {
    const dropPct = Math.abs(hStats.delta * 100);
    recs.push({
      severity: dropPct > 25 ? "high" : "medium",
      title: `Hallucination regressed ${dropPct.toFixed(0)} points from baseline`,
      cause: `Newest third averages <strong>${hStats.current.toFixed(2)}</strong> vs <strong>${hStats.baseline.toFixed(2)}</strong> baseline. Something changed — a deploy, a model swap, or a retrieval index update.`,
      checks: [
        `Open the Quality tab's heatmap. Find the row that goes green → red.`,
        `Cross-reference that time with your deploy log.`,
        `Replay a regressed call: <code>peekr replay &lt;trace_id&gt;</code>`,
        `Use the time-range chips above (1h, 24h) to bisect when it started.`,
      ],
      evidence: `Δ = ${hStats.delta.toFixed(2)} (baseline ${hStats.baseline.toFixed(2)} → current ${hStats.current.toFixed(2)}, n=${hStats.n})`,
    });
  }

  if (rows.length >= 20 && errors / rows.length > 0.05) {
    recs.push({
      severity: errors / rows.length > 0.2 ? "high" : "medium",
      title: `${errors} of ${rows.length} calls failed outright (${((errors/rows.length)*100).toFixed(1)}%)`,
      cause: `Spans with <code>status="error"</code> — rate limits, timeouts, malformed responses. Error spikes can mask quality drops if fallback models silently kick in.`,
      checks: [
        `<code>SELECT name, COUNT(*) FROM spans WHERE status='error' GROUP BY name</code>`,
        `Increase your LLM client timeout if calls are long.`,
        `Verify any failover provider isn't a weaker model silently lowering quality.`,
      ],
      evidence: `${errors} errors out of ${rows.length} calls.`,
    });
  }

  if (cit.total >= 5 && cit.invented === 0) {
    recs.push({
      severity: "good",
      title: `All ${cit.total} detected citations are grounded`,
      cause: `Every URL, arXiv ID, paper title, and statute reference we found appears in the source context. Your RAG is keeping the model honest about sources.`,
      checks: [`Keep <code>CitationAccuracy</code> in your evaluators — it's a zero-LLM-cost continuous signal.`],
    });
  }

  if (recs.length === 0) {
    recs.push({
      severity: "good", title: "All scored signals are within expected ranges",
      cause: `Current Hallucination is <strong>${hStats.current.toFixed(2)}</strong> (baseline ${hStats.baseline.toFixed(2)}).`,
      checks: [
        `Set up an alert at <code>Hallucination &lt; 0.5</code>.`,
        `Run the offline benchmark: <code>python examples/hallucination_benchmark/run.py</code>`,
      ],
    });
  }
  return recs;
}
function renderRecommendations(rows) {
  const wrap = document.getElementById("recommendations");
  if (!wrap) return;
  const recs = diagnoseRecommendations(rows);
  wrap.innerHTML = recs.map(r => {
    const checks = r.checks && r.checks.length
      ? `<div class="rec-checks"><div class="checks-label">What to try:</div><ol>${r.checks.map(c => `<li>${c}</li>`).join("")}</ol></div>`
      : "";
    const evidence = r.evidence ? `<div class="rec-evidence">${esc(r.evidence)}</div>` : "";
    const sevLabel = ({high:"high", medium:"medium", low:"low", info:"info", good:"good"})[r.severity] || "info";
    return `<div class="rec ${r.severity}">
      <div class="rec-head"><span class="rec-sev ${r.severity}">${sevLabel}</span><span class="rec-title">${r.title}</span></div>
      <div class="rec-cause">${r.cause}</div>
      ${checks}${evidence}
    </div>`;
  }).join("");
  const tcRecs = document.getElementById("tc-recs");
  if (tcRecs) {
    const urgent = recs.filter(r => r.severity === "high" || r.severity === "medium").length;
    tcRecs.textContent = urgent ? String(urgent) : "";
  }
}

// Per-span action items
function perSpanActions(row) {
  const out = [];
  const ctx = (() => { try { return parseInput(row.input, row).context || ""; } catch { return ""; } })();
  const outputText = row.output || "";
  const score = row.Hallucination;
  const details = row.details;
  const claims = (details && details.claims) || [];
  const contraClaims = claims.filter(c => c.verdict === "contradicted");
  const unsupClaims  = claims.filter(c => c.verdict === "unsupported");
  const cd = row.citation_details;
  const inventedCitations = (cd && cd.items) ? cd.items.filter(i => !i.grounded) : [];
  const numericContra = contraClaims.filter(c => /\b\d{2,}\b/.test(c.text));
  const properNounContra = contraClaims.filter(c => /\b[A-Z][a-z]+\s+[A-Z][a-z]+\b/.test(c.text));

  if (ctx.length < 60 && outputText.length > 100) out.push({ fix: `<strong>Likely retrieval miss:</strong> context is only ${ctx.length} chars but the model produced ${outputText.length} chars of answer. Inspect what your retriever returned.`, why: "Sparse context + verbose answer → model filling from training data." });
  if (inventedCitations.length) {
    const kinds = [...new Set(inventedCitations.map(c => c.kind))].join(", ");
    const ex = inventedCitations.slice(0, 2).map(c => `<code>${esc(c.text)}</code>`).join(" and ");
    out.push({ fix: `<strong>Invented ${kinds} citation${inventedCitations.length>1?"s":""}</strong> (${ex}). Add to prompt: <em>"Cite only sources present in the context."</em>`, why: "Model fabricated references — common when retrieval doesn't return real sources." });
  }
  if (numericContra.length) out.push({ fix: `<strong>${numericContra.length} claim${numericContra.length>1?"s":""} with wrong numbers/dates</strong>. Add: <em>"Be exact about numbers and dates — copy them verbatim from context."</em> Lower <code>temperature</code> to 0.`, why: "Models drift on specific numerics without explicit instruction." });
  if (properNounContra.length) out.push({ fix: `<strong>Proper noun substitution.</strong> Add: <em>"Use only names of people/places/orgs that appear in the context. Do not substitute alternatives."</em>`, why: "For less-famous entities, models substitute more-famous ones." });
  if (claims.length >= 2 && unsupClaims.length / claims.length > 0.5 && contraClaims.length === 0) out.push({ fix: `<strong>Out-of-context elaboration</strong>: ${unsupClaims.length} of ${claims.length} claims unsupported. Add: <em>"If the context doesn't contain the answer, say 'I don't have that information.'"</em>`, why: "Context silent; model filled from training data instead of refusing." });
  if (claims.length >= 2 && contraClaims.length / claims.length > 0.5) out.push({ fix: `<strong>Model is overriding the context.</strong> Move context closer to the question (recency bias). Add: <em>"If the context contradicts what you know, the context wins."</em>`, why: `${contraClaims.length} of ${claims.length} claims conflict — model trusts its training prior more.` });
  if (score != null && score < 0.3 && !details) out.push({ fix: `Enable <strong>detailed mode</strong>: <code>peekr.eval.Hallucination(detailed=True)</code> to see which claims failed.`, why: "Simple mode gave one number but no breakdown." });
  if (row.status === "error") out.push({ fix: `<strong>Call failed</strong> (<code>${esc(row.error || "see span")}</code>). Investigate rate limits, timeouts, malformed responses.`, why: "Failed calls correlate with quality regressions if fallback paths exist." });
  if (ctx.length > 60 && outputText.length > ctx.length * 2) out.push({ fix: `Output is <strong>${(outputText.length / ctx.length).toFixed(1)}× longer than context</strong>. Reduce <code>max_tokens</code>.`, why: "Long completions drift once grounded content is exhausted." });
  if (score != null && score >= 0.5 && score < 0.7 && out.length === 0) out.push({ fix: `Score is on the boundary (${score.toFixed(2)}). Review whether your flagging threshold is too strict.`, why: "Borderline cases are often acceptable depending on use case." });
  return out;
}

function renderOffenders(rows) {
  const wrap = document.getElementById("offender-list");
  if (!wrap) return;
  const scored = rows.filter(r => r.Hallucination != null);
  const worst = [...scored].sort((a,b) => a.Hallucination - b.Hallucination).slice(0, 12);
  if (!worst.length) {
    wrap.innerHTML = '<div class="empty">No spans match the current filter, or no Hallucination scores recorded yet.</div>';
    return;
  }
  wrap.innerHTML = worst.map((r, i) => {
    const tier = tierForScore(r.Hallucination);
    return `<div class="offender" id="offender-${esc(r.span_id || ("idx"+i))}">
      <div class="offender-head">
        <div style="display:flex;align-items:center;gap:0.6rem">
          <span class="offender-rank">${i + 1}</span>
          <span class="score-pill ${tier}">${r.Hallucination.toFixed(2)}</span>
          <span style="color:var(--muted);font-size:0.78rem">${tsLabel(r.ts)}</span>
        </div>
        <div style="display:flex;gap:0.4rem;flex-wrap:wrap">
          <span class="tag">${esc(r.model || "—")}</span>
          <span class="tag">${esc(r.tenant || "—")}</span>
          <span class="tag">${esc(r.endpoint || "—")}</span>
          ${r.status === "error" ? '<span class="tag err">error</span>' : ""}
        </div>
      </div>
      <div class="offender-body">${renderDetailHTML(r)}</div>
    </div>`;
  }).join("");
}

// ═══════════════════════════════════════════════════════════════════
//  Detail rendering helpers (used by side panel + offenders)
// ═══════════════════════════════════════════════════════════════════
function highlightAll(text, claims) {
  if (!claims || !claims.length) return esc(text);
  const order = ["contradicted", "unsupported", "supported"];
  let html = esc(text);
  const used = new Set();
  for (const v of order) {
    for (const c of claims.filter(x => x.verdict === v)) {
      const key = (c.text || "").toLowerCase();
      if (used.has(key)) continue;
      used.add(key);
      const cls = v === "contradicted" ? "mark-bad" : v === "unsupported" ? "mark-warn" : "mark-good";
      const candidates = [c.text];
      const words = (c.text || "").split(/\s+/);
      if (words.length > 4) candidates.push(words.slice(0, 4).join(" "));
      const keyword = words.find(w => /^\d|^[A-Z]/.test(w));
      if (keyword) candidates.push(keyword.replace(/[^\w\d]/g, ""));
      for (const t of candidates) {
        if (!t || t.length < 3) continue;
        const escT = esc(t).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
        const re = new RegExp(escT, "i");
        const m = re.exec(html);
        if (m) {
          html = html.slice(0, m.index) + `<span class="${cls}">` + m[0] + "</span>" + html.slice(m.index + m[0].length);
          break;
        }
      }
    }
  }
  return html;
}
function parseInput(inp, row) {
  try {
    const messages = JSON.parse(inp);
    const sys = Array.isArray(messages) ? messages.find(m => m && m.role === "system") : null;
    const usr = Array.isArray(messages) ? messages.find(m => m && m.role === "user")   : null;
    const context = sys ? sys.content : (row && typeof row.system === "string" ? row.system : "");
    return { context, question: usr ? usr.content : "" };
  } catch (_) {
    return { context: inp || "", question: "" };
  }
}

// ═══════════════════════════════════════════════════════════════════
//  Help tab — live setup checklist
// ═══════════════════════════════════════════════════════════════════
function renderSetupChecklist() {
  const ul = document.getElementById("setup-checklist");
  if (!ul) return;
  const r = applyFilter(DATA.rows);
  const hasScores = r.some(x => x.Hallucination != null);
  const hasDetailed = r.some(x => x.details);
  const hasEndpoint = r.some(x => x.endpoint);
  const hasTenant = r.some(x => x.tenant);
  const hasCitations = r.some(x => x.citation_details);
  const judgeErrors = r.some(x => x.eval_errors && Object.keys(x.eval_errors).length > 0);
  const items = [
    { done: hasScores,      text: `Hallucination evaluator is wired (<code>peekr.eval.Hallucination()</code>)` },
    { done: !judgeErrors && hasScores, text: `Judge has credentials (<code>OPENAI_API_KEY</code> or <code>ANTHROPIC_API_KEY</code>)`, warn: judgeErrors },
    { done: hasDetailed,    text: `At least some spans use detailed mode for per-claim verdicts (<code>detailed=True</code>)` },
    { done: hasCitations,   text: `CitationAccuracy evaluator is wired (free, no LLM calls)` },
    { done: hasEndpoint,    text: `Spans tagged with <code>attributes.endpoint</code> for the channel heatmap` },
    { done: hasTenant,      text: `Spans tagged with <code>attributes.user_id</code> via <code>peekr.session(user_id=...)</code> for tenant analysis` },
  ];
  ul.innerHTML = items.map(i => {
    const cls = i.done ? "done" : (i.warn ? "" : "");
    const warnNote = i.warn ? ' <span style="color:var(--red);font-size:0.78em">(judge errors detected — see Diagnose)</span>' : "";
    return `<li class="${cls}">${i.text}${warnNote}</li>`;
  }).join("");
}

// ═══════════════════════════════════════════════════════════════════
//  Keyboard shortcuts
// ═══════════════════════════════════════════════════════════════════
function setupKeyboard() {
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") {
      if (e.key === "Escape") e.target.blur();
      return;
    }
    if (e.key === "Escape") { closeSidePanel(); return; }
    if (e.key === "/")      { e.preventDefault(); switchTab("traces"); const inp = document.getElementById("trace-search"); if (inp) inp.focus(); return; }
    if (e.key === "r" || e.key === "R") { clearFilters(); return; }
    if (e.key >= "1" && e.key <= "5") {
      const order = ["overview", "traces", "quality", "diagnose", "help"];
      switchTab(order[+e.key - 1]);
    }
  });
}

// ═══════════════════════════════════════════════════════════════════
//  Bootstrap
// ═══════════════════════════════════════════════════════════════════
renderFilterBar();
setupTabNav();
setupTracesHeaders();
setupTracesSearch();
setupSidePanel();
setupKeyboard();
rerender();
</script>

</body>
</html>
"""
