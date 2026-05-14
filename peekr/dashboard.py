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

    llm_spans = [s for s in spans if any(s["name"].startswith(p) for p in _LLM_PREFIXES)]
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
    return ((span.get("attributes") or {}).get("eval_scores") or {})


def _summary(all_spans: list[dict], llm_spans: list[dict]) -> dict[str, Any]:
    eval_spans = [s for s in llm_spans if _scores(s)]
    detailed = [s for s in eval_spans if (s.get("attributes") or {}).get("hallucination_details")]
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
        points.append({
            "trace_id": (s.get("trace_id") or "")[:8],
            "ts": s.get("start_time") or 0,
            "Hallucination":    scores.get("Hallucination"),
            "Rubric":           scores.get("Rubric"),
            "CitationAccuracy": scores.get("CitationAccuracy"),
            "NotEmpty":         scores.get("NotEmpty"),
            "NoError":          scores.get("NoError"),
            "error": 1 if s.get("status") == "error" else 0,
            "tokens": attrs.get("tokens_total") or 0,
            "duration_ms": s.get("duration_ms") or 0,
            "tenant":   attrs.get("user_id"),
            "endpoint": attrs.get("endpoint"),
            "model":    attrs.get("model"),
        })
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
        values = [s for s in (_scores(span).get(m) for span in llm_spans) if s is not None]
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
            scores = [
                _scores(x).get("Hallucination") for x in items_sorted
            ]
            scores = [v for v in scores if v is not None]
            n = len(scores)
            if n < 4:
                # Not enough data — still include the segment but mark NA
                rows.append({
                    "segment": segment,
                    "n": n,
                    "current": (sum(scores) / n) if n else None,
                    "baseline": None,
                    "delta": None,
                    "n_total": n,
                })
                continue
            k = max(1, n // 3)
            baseline = sum(scores[:k]) / k
            current = sum(scores[-k:]) / k
            rows.append({
                "segment": segment,
                "n": n,
                "baseline": baseline,
                "current": current,
                "delta": current - baseline,
                "n_baseline": k,
                "n_current": k,
                "n_total": n,
            })
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
        scored.append({
            "trace_id": s.get("trace_id"),
            "span_id": s.get("span_id"),
            "ts": s.get("start_time") or 0,
            "model": attrs.get("model", ""),
            "score": h,
            "output": (attrs.get("output") or "")[:300],
            "input": (attrs.get("input") or "")[:300],
            "details": attrs.get("hallucination_details"),
        })
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
        out.append({
            "trace_id": s.get("trace_id"),
            "span_id":  s.get("span_id"),
            "ts":       s.get("start_time") or 0,
            "model":    attrs.get("model"),
            "tenant":   attrs.get("user_id"),
            "endpoint": attrs.get("endpoint"),
            "status":   s.get("status", "ok"),
            "tokens":   attrs.get("tokens_total") or 0,
            "duration_ms": s.get("duration_ms") or 0,
            "Hallucination":    scores.get("Hallucination"),
            "Rubric":           scores.get("Rubric"),
            "CitationAccuracy": scores.get("CitationAccuracy"),
            "NotEmpty":         scores.get("NotEmpty"),
            "NoError":          scores.get("NoError"),
            "input":  (attrs.get("input")  or "")[:600],
            "output": (attrs.get("output") or "")[:600],
            "details":          attrs.get("hallucination_details"),
            "citation_details": attrs.get("citation_details"),
            "error":  attrs.get("error"),
        })
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
        buckets.append({
            "start": b_start,
            "end":   b_end,
            "label": f"{i + 1}/{n_buckets}",   # short label; tooltip gives the time range
        })

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
                {"mean": sum(c) / len(c), "n": len(c)} if c
                else {"mean": None, "n": 0}
                for c in cell_lists
            ]
            n_total = sum(c["n"] for c in cells)
            rows.append({
                "segment":  str(seg),
                "n_total":  n_total,
                "cells":    cells,
            })
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
            "insights": ["No Hallucination scores recorded — add peekr.eval.Hallucination() to your evaluators."],
        }

    scored.sort(key=lambda x: x[0])
    score_values = [v for _, v, _ in scored]
    n = len(score_values)
    current = sum(score_values[-max(1, n // 3):]) / max(1, n // 3)
    baseline = sum(score_values[:max(1, n // 3)]) / max(1, n // 3) if n >= 6 else current

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
    --border: #30363d; --text: #e6edf3; --muted: #8b949e;
    --accent: #58a6ff; --green: #3fb950; --red: #f85149;
    --orange: #ffa657; --yellow: #e3b341; --purple: #bc8cff;
    --shadow: 0 2px 12px rgba(0,0,0,0.3);
  }
  body { background: var(--bg); color: var(--text); font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; padding: 0; }
  /* Every Chart.js canvas needs a parent with a fixed height — otherwise the chart
     grows a few pixels on every re-render, pushing the page content below it down. */
  canvas { display: block; }

  /* Help / info badge with hover tooltip */
  .help { display: inline-block; margin-left: 0.4em; width: 15px; height: 15px; border-radius: 50%;
          background: var(--bg3); color: var(--muted); font-size: 0.62rem; line-height: 15px;
          text-align: center; cursor: help; position: relative; font-weight: 700; vertical-align: 0.15em; }
  .help:hover { background: var(--accent); color: var(--bg); }
  .help[data-tip]:hover::after { content: attr(data-tip); position: absolute; bottom: 150%; left: 50%;
          transform: translateX(-50%); background: var(--bg4); color: var(--text); padding: 0.6rem 0.8rem;
          border-radius: 6px; font-size: 0.78rem; white-space: pre-wrap; width: 280px;
          box-shadow: 0 6px 18px rgba(0,0,0,0.5); z-index: 100; font-weight: 400; text-align: left;
          line-height: 1.55; border: 1px solid var(--border); pointer-events: none; }
  .help[data-tip]:hover::before { content: ""; position: absolute; bottom: 145%; left: 50%;
          transform: translateX(-50%); border: 6px solid transparent; border-top-color: var(--bg4); z-index: 100; }

  /* Action-hint line under each metric */
  .metric-action { font-size: 0.74rem; color: var(--accent); margin-top: 0.3rem; line-height: 1.4; font-weight: 500; }
  .metric-action.warn { color: var(--orange); }
  .metric-action.bad  { color: var(--red); }
  .metric-action.ok   { color: var(--green); }
  .metric-action.flat { color: var(--muted); font-weight: 400; }

  /* Recommendations panel — diagnose + fix */
  .recs-list { display: flex; flex-direction: column; gap: 0.85rem; }
  .rec { background: var(--bg3); border: 1px solid var(--border); border-left: 4px solid var(--muted); border-radius: 8px; padding: 0.9rem 1.1rem; }
  .rec.high { border-left-color: var(--red); }
  .rec.medium { border-left-color: var(--orange); }
  .rec.low { border-left-color: var(--yellow); }
  .rec.info { border-left-color: var(--accent); }
  .rec.good { border-left-color: var(--green); }
  .rec-head { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.45rem; }
  .rec-sev { font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 700; padding: 0.1em 0.5em; border-radius: 4px; }
  .rec-sev.high   { background: rgba(248,81,73,0.18);  color: var(--red); }
  .rec-sev.medium { background: rgba(255,166,87,0.18); color: var(--orange); }
  .rec-sev.low    { background: rgba(227,179,65,0.18); color: var(--yellow); }
  .rec-sev.info   { background: rgba(88,166,255,0.18); color: var(--accent); }
  .rec-sev.good   { background: rgba(63,185,80,0.18);  color: var(--green); }
  .rec-title { font-weight: 700; font-size: 0.95rem; }
  .rec-cause { font-size: 0.86rem; color: #c9d1d9; line-height: 1.55; margin-bottom: 0.6rem; }
  .rec-cause strong { color: var(--text); }
  .rec-checks { font-size: 0.84rem; line-height: 1.6; }
  .rec-checks .checks-label { font-weight: 600; color: var(--text); margin-bottom: 0.3rem; }
  .rec-checks ol { margin: 0; padding-left: 1.4rem; color: #c9d1d9; }
  .rec-checks li { margin-bottom: 0.25rem; }
  .rec-checks code { background: var(--bg2); padding: 0.1em 0.35em; border-radius: 3px; font-family: "SFMono-Regular", Consolas, monospace; font-size: 0.88em; }
  .rec-evidence { font-size: 0.78rem; color: var(--muted); margin-top: 0.55rem; padding-top: 0.55rem; border-top: 1px dashed var(--border); }

  /* Read-this collapsible */
  details.howto { background: var(--bg2); border: 1px solid var(--border); border-radius: 10px; padding: 0; margin-bottom: 1rem; }
  details.howto summary { padding: 0.7rem 1rem; cursor: pointer; font-size: 0.85rem; font-weight: 600; color: var(--text); list-style: none; display: flex; justify-content: space-between; align-items: center; }
  details.howto summary::-webkit-details-marker { display: none; }
  details.howto summary::after { content: "▾"; color: var(--muted); transition: transform 0.2s; }
  details.howto[open] summary::after { transform: rotate(180deg); }
  details.howto[open] summary { border-bottom: 1px solid var(--border); }
  details.howto .body { padding: 1rem 1.25rem; font-size: 0.86rem; color: #c9d1d9; line-height: 1.7; }
  details.howto .body dl { display: grid; grid-template-columns: max-content 1fr; gap: 0.45rem 1rem; }
  details.howto .body dt { font-weight: 600; color: var(--text); }
  details.howto .body code { background: var(--bg3); padding: 0.1em 0.35em; border-radius: 3px; font-family: "SFMono-Regular", Consolas, monospace; font-size: 0.85em; }
  .wrap { max-width: 1240px; margin: 0 auto; padding: 1.5rem 2rem 4rem; }

  header { padding-bottom: 1rem; border-bottom: 1px solid var(--border); margin-bottom: 1.5rem; }
  h1 { font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0.15rem; }
  .sub { color: var(--muted); font-size: 0.85rem; }
  .sub code { background: var(--bg3); padding: 0.1em 0.35em; border-radius: 3px; font-family: "SFMono-Regular", Consolas, monospace; font-size: 0.85em; }

  .filter-bar { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; margin: 1rem 0 0.25rem; padding: 0.75rem 1rem; background: var(--bg2); border: 1px solid var(--border); border-radius: 10px; }
  .filter-group { display: flex; flex-wrap: wrap; gap: 0.3rem; align-items: center; }
  .filter-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin-right: 0.3rem; padding-right: 0.4rem; border-right: 1px solid var(--border); }
  .filter-group:first-child .filter-label { border-right: none; padding-right: 0; }
  .chip { font-size: 0.78rem; padding: 0.25rem 0.65rem; border-radius: 999px; background: var(--bg3); border: 1px solid var(--border); color: var(--muted); cursor: pointer; transition: all 0.12s; }
  .chip:hover { color: var(--text); border-color: var(--muted); }
  .chip.active { background: var(--accent); color: #0d1117; border-color: var(--accent); font-weight: 600; }
  .chip-reset { color: var(--accent); background: transparent; border: 1px dashed var(--accent); margin-left: auto; }
  .chip-reset:hover { background: rgba(88,166,255,0.12); }
  .filter-count { font-size: 0.78rem; color: var(--muted); margin-left: 0.5rem; }

  .hero { display: grid; grid-template-columns: 1.1fr 1.5fr; gap: 1rem; background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem 1.75rem; margin-bottom: 1rem; box-shadow: var(--shadow); }
  .hero-left { display: flex; gap: 1.25rem; align-items: center; }
  .health-dot { width: 64px; height: 64px; border-radius: 50%; flex-shrink: 0; position: relative; box-shadow: 0 0 0 8px rgba(255,255,255,0.03); }
  .health-dot.good     { background: radial-gradient(circle at 30% 30%, #6fdc8c, var(--green)); }
  .health-dot.ok       { background: radial-gradient(circle at 30% 30%, #f0d674, var(--yellow)); }
  .health-dot.warning  { background: radial-gradient(circle at 30% 30%, #ffbe7a, var(--orange)); }
  .health-dot.critical { background: radial-gradient(circle at 30% 30%, #ff7c75, var(--red)); animation: pulse 1.8s infinite; }
  @keyframes pulse { 0%,100% { box-shadow: 0 0 0 8px rgba(248,81,73,0.15); } 50% { box-shadow: 0 0 0 14px rgba(248,81,73,0.05); } }
  .hero-text .hero-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.2rem; }
  .hero-text .hero-value { font-size: 2.3rem; font-weight: 800; letter-spacing: -0.02em; line-height: 1; margin-bottom: 0.3rem; }
  .hero-text .hero-tier { font-size: 0.88rem; font-weight: 600; }
  .hero-text .hero-tier.good     { color: var(--green); }
  .hero-text .hero-tier.ok       { color: var(--yellow); }
  .hero-text .hero-tier.warning  { color: var(--orange); }
  .hero-text .hero-tier.critical { color: var(--red); }
  .hero-text .hero-sub { color: var(--muted); font-size: 0.82rem; margin-top: 0.5rem; line-height: 1.6; }
  .hero-right { display: flex; flex-direction: column; }
  .hero-spark-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.4rem; }
  .hero-spark-meta { color: var(--text); font-weight: 600; font-size: 0.78rem; letter-spacing: 0; text-transform: none; }
  .hero-spark-meta .down { color: var(--red); } .hero-spark-meta .up { color: var(--green); }
  .hero-spark { position: relative; height: 90px; width: 100%; }

  .narrative { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 1.1rem 1.5rem; margin-bottom: 1.5rem; }
  .narrative h2 { font-size: 0.78rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.6rem; }
  .narrative ul { list-style: none; padding: 0; margin: 0; }
  .narrative li { padding: 0.35rem 0; padding-left: 1.4rem; position: relative; font-size: 0.92rem; line-height: 1.55; }
  .narrative li::before { content: "›"; position: absolute; left: 0.3rem; color: var(--accent); font-weight: 700; }
  .narrative li strong { color: var(--text); font-weight: 600; }

  .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.8rem; margin-bottom: 1.5rem; }
  .metric { background: var(--bg2); border: 1px solid var(--border); border-radius: 10px; padding: 1rem 1.1rem; position: relative; overflow: hidden; }
  .metric-label { font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.35rem; }
  .metric-row { display: flex; align-items: baseline; gap: 0.5rem; }
  .metric-value { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.01em; }
  .metric-value.good { color: var(--green); } .metric-value.ok { color: var(--text); }
  .metric-value.warning { color: var(--orange); } .metric-value.critical { color: var(--red); }
  .metric-delta { font-size: 0.78rem; font-weight: 600; }
  .metric-delta.up { color: var(--green); } .metric-delta.down { color: var(--red); } .metric-delta.flat { color: var(--muted); }
  /* Fixed-height wrapper so Chart.js measures against a stable parent every re-render. */
  .metric-spark-wrap { position: relative; height: 40px; width: 100%; margin-top: 0.4rem; }
  .metric-foot { font-size: 0.72rem; color: var(--muted); margin-top: 0.15rem; }

  section.panel { background: var(--bg2); border: 1px solid var(--border); border-radius: 12px; padding: 1.25rem 1.4rem; margin-bottom: 1.5rem; }
  section.panel h2 { font-size: 1rem; font-weight: 700; margin-bottom: 0.2rem; }
  section.panel .hint { color: var(--muted); font-size: 0.84rem; margin-bottom: 1rem; }

  .chart-wrap { position: relative; height: 320px; overflow: hidden; }
  .chart-wrap.tall { height: 360px; }

  .heatmap-group { margin-bottom: 1.5rem; }
  .heatmap-group:last-child { margin-bottom: 0; }
  .heatmap-title { font-size: 0.76rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.55rem; }
  .heatmap { display: grid; gap: 2px; align-items: stretch; }
  .heatmap-row-label { font-size: 0.78rem; color: var(--text); padding: 0.45rem 0.6rem 0.45rem 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 220px; }
  .heatmap-row-label code { background: var(--bg3); padding: 0.1em 0.35em; border-radius: 3px; font-family: "SFMono-Regular", Consolas, monospace; font-size: 0.82em; }
  .heatmap-row-n { color: var(--muted); font-size: 0.68rem; margin-left: 0.3rem; }
  .heatmap-bucket-label { font-size: 0.68rem; color: var(--muted); text-align: center; padding-bottom: 0.4rem; }
  .heatmap-cell { padding: 0.55rem 0; border-radius: 4px; text-align: center; font-size: 0.78rem; font-weight: 600; cursor: pointer; transition: transform 0.12s; position: relative; }
  .heatmap-cell:hover { transform: scale(1.05); z-index: 1; box-shadow: 0 0 0 2px var(--text); }
  .heatmap-cell.empty { background: var(--bg3); color: var(--muted); }
  .heatmap-legend { display: flex; align-items: center; gap: 0.4rem; margin-top: 0.6rem; font-size: 0.72rem; color: var(--muted); }
  .heatmap-legend .swatch { display: inline-block; width: 14px; height: 14px; border-radius: 3px; }

  .offender-list { display: flex; flex-direction: column; gap: 1rem; }
  .offender { background: var(--bg3); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; transition: border-color 0.15s; }
  .offender:hover { border-color: var(--accent); }
  .offender-head { display: flex; justify-content: space-between; align-items: center; padding: 0.7rem 1rem; background: var(--bg4); border-bottom: 1px solid var(--border); flex-wrap: wrap; gap: 0.5rem; }
  .offender-rank { display: inline-block; background: var(--bg2); border-radius: 50%; width: 24px; height: 24px; text-align: center; font-size: 0.72rem; font-weight: 700; line-height: 24px; color: var(--muted); }
  .offender-score { font-size: 1.05rem; font-weight: 700; padding: 0.15rem 0.6rem; border-radius: 999px; }
  .offender-score.critical { background: rgba(248,81,73,0.18); color: var(--red); }
  .offender-score.warning  { background: rgba(255,166,87,0.18); color: var(--orange); }
  .offender-score.ok       { background: rgba(63,185,80,0.18); color: var(--green); }
  .offender-meta { display: flex; gap: 0.4rem; flex-wrap: wrap; font-size: 0.74rem; color: var(--muted); }
  .offender-meta .tag { background: var(--bg2); border: 1px solid var(--border); padding: 0.12em 0.55em; border-radius: 4px; font-family: "SFMono-Regular", Consolas, monospace; }
  .offender-body { padding: 1rem; }
  .offender-q { color: var(--muted); font-size: 0.82rem; margin-bottom: 0.7rem; }
  .offender-q strong { color: var(--text); font-weight: 600; margin-right: 0.4rem; }
  .offender-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-bottom: 0.85rem; }
  .text-panel { background: var(--bg2); border: 1px solid var(--border); border-radius: 8px; padding: 0.7rem 0.85rem; }
  .text-panel-label { font-size: 0.66rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); margin-bottom: 0.35rem; display: flex; justify-content: space-between; }
  .text-panel-content { font-size: 0.82rem; line-height: 1.55; word-break: break-word; }
  .text-panel-content .mark-bad   { background: rgba(248,81,73,0.25); color: #ffd0cc; padding: 0 2px; border-radius: 2px; }
  .text-panel-content .mark-warn  { background: rgba(255,166,87,0.22); color: #ffe4c8; padding: 0 2px; border-radius: 2px; }
  .text-panel-content .mark-good  { background: rgba(63,185,80,0.22); color: #c5f0c9; padding: 0 2px; border-radius: 2px; }
  .claims-list { display: flex; flex-direction: column; gap: 0.35rem; }
  .claim { display: flex; align-items: flex-start; gap: 0.5rem; padding: 0.45rem 0.65rem; background: var(--bg2); border-radius: 6px; border-left: 3px solid var(--border); font-size: 0.85rem; }
  .claim.supported    { border-left-color: var(--green); }
  .claim.contradicted { border-left-color: var(--red); }
  .claim.unsupported  { border-left-color: var(--orange); }
  .claim .v { display: inline-block; width: 90px; flex-shrink: 0; font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 700; }
  .claim.supported    .v { color: var(--green); }
  .claim.contradicted .v { color: var(--red); }
  .claim.unsupported  .v { color: var(--orange); }
  .citations-list { display: flex; flex-direction: column; gap: 0.3rem; }
  .citation { display: flex; gap: 0.6rem; padding: 0.35rem 0.55rem; border-radius: 4px; font-size: 0.8rem; align-items: center; }
  .citation.invented { background: rgba(248,81,73,0.1); }
  .citation.grounded { background: rgba(63,185,80,0.08); }
  .citation .kind { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); width: 80px; }
  .citation code { font-family: "SFMono-Regular", Consolas, monospace; font-size: 0.85em; }
  .citation .badge { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 700; margin-left: auto; }
  .citation.invented .badge { color: var(--red); }
  .citation.grounded .badge { color: var(--green); }

  /* Per-span action box at the bottom of each offender card */
  .span-actions { margin-top: 0.85rem; padding: 0.7rem 0.85rem; background: rgba(88,166,255,0.06); border: 1px solid rgba(88,166,255,0.25); border-left: 3px solid var(--accent); border-radius: 6px; }
  .span-actions-label { font-size: 0.66rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--accent); margin-bottom: 0.4rem; }
  .span-actions ul { list-style: none; padding: 0; margin: 0; }
  .span-actions li { padding: 0.25rem 0; font-size: 0.82rem; line-height: 1.5; color: #c9d1d9; }
  .span-actions li strong { color: var(--text); }
  .span-actions li code { background: var(--bg2); padding: 0.1em 0.35em; border-radius: 3px; font-family: "SFMono-Regular", Consolas, monospace; font-size: 0.88em; }
  .span-actions .why { color: var(--muted); font-size: 0.76rem; margin-top: 0.1rem; display: block; }

  .empty { color: var(--muted); padding: 2rem 1rem; text-align: center; font-style: italic; font-size: 0.92rem; }
  .empty code { background: var(--bg3); padding: 0.1em 0.35em; border-radius: 3px; font-family: "SFMono-Regular", Consolas, monospace; font-style: normal; }

  @media (max-width: 900px) {
    .wrap { padding: 1rem; }
    .hero { grid-template-columns: 1fr; gap: 1.5rem; }
    .offender-grid { grid-template-columns: 1fr; }
    .heatmap-row-label { max-width: 110px; font-size: 0.72rem; }
  }
</style>
</head>
<body>

<div class="wrap">

  <header>
    <h1>peekr · observability</h1>
    <div class="sub">Source: <code>__SOURCE__</code> · click any cell, chip, or score to drill in</div>
  </header>

  <details class="howto">
    <summary>How to read this dashboard</summary>
    <div class="body">
      <dl>
        <dt>Score (0 → 1)</dt>
        <dd><strong>0</strong> = the model's answer is fully unsupported by its context (hallucinating). <strong>1</strong> = every factual claim is grounded in context. We aggregate per call and per channel.</dd>
        <dt>Health score (0 → 100)</dt>
        <dd>The average Hallucination score over the newest third of the calls in your current filter, multiplied by 100. We compare this <em>current</em> window to the oldest third (<em>baseline</em>) to detect drift.</dd>
        <dt>"134 scored" / "n=134"</dt>
        <dd>Number of calls that have a score for this metric. Some calls are skipped (no context to ground against, or evaluator not configured), so this is &le; the total span count.</dd>
        <dt>Baseline vs current</dt>
        <dd>We split the calls in your filter into thirds by time. <strong>Baseline</strong> = mean over the oldest third, <strong>current</strong> = mean over the newest third. A negative delta means the score is worse than it used to be.</dd>
        <dt>Heatmap colors</dt>
        <dd>Each cell is the mean Hallucination score for that channel in that time window. <span style="color:var(--red)">Red</span> = mostly hallucinating, <span style="color:var(--yellow)">yellow</span> = mixed, <span style="color:var(--green)">green</span> = grounded. Grey = no calls in that bucket.</dd>
        <dt>Action hints</dt>
        <dd>Each metric card shows a short suggestion (e.g. <em>"30 flagged — review below"</em>) computed from the current view. Click the suggested section to drill in.</dd>
        <dt>Filters</dt>
        <dd>Click a chip (tenant / model / endpoint / time range) to filter every panel in the dashboard. Click the same chip again to clear it. Filters stack — e.g. tenant + model + last 24h.</dd>
      </dl>
    </div>
  </details>

  <div class="filter-bar" id="filter-bar"></div>

  <div class="hero" id="hero"></div>

  <div class="narrative" id="narrative-card">
    <h2>What's happening</h2>
    <ul id="narrative-list"></ul>
  </div>

  <div class="metrics" id="metrics-row"></div>

  <section class="panel">
    <h2>Likely causes &amp; next steps<span class="help" data-tip="Auto-generated diagnostic suggestions based on the patterns we see in the current filter. These are starting hypotheses, not certainties — pair them with the worst-offender cards to confirm. The list updates as you change filters, so you can compare healthy vs degraded views.">?</span></h2>
    <p class="hint">Each suggestion below is a hypothesis derived from the failure patterns in your data, paired with concrete things to try. Common in RAG: retrieval misses, chunking issues, missing refusal prompts. Common in agents / memory: stale context, ContextVar bleed, tool-output contamination.</p>
    <div class="recs-list" id="recommendations"></div>
  </section>

  <section class="panel">
    <h2>Score over time<span class="help" data-tip="Rolling 20-call mean for each evaluator. The dashed orange line is the warning threshold (0.7) and the dashed red line is the critical threshold (0.5). Hover a point to see what trace it belongs to. Click a point to jump down to that trace's worst-offender card.">?</span></h2>
    <p class="hint">If the Hallucination line dips below the orange or red dashed line, recent calls are crossing into the warning or critical zone. Look for sudden drops vs gradual decay.</p>
    <div class="chart-wrap tall"><canvas id="rolling-chart"></canvas></div>
  </section>

  <section class="panel">
    <h2>Failure breakdown by channel & time<span class="help" data-tip="Rows = channels (your models, tenants, endpoints). Columns = time buckets, oldest on the left, newest on the right. Cell color = mean Hallucination score for that channel in that window. Red = hallucinating, green = grounded, grey = no calls in that bucket. Click a cell to filter the whole dashboard to that channel.">?</span></h2>
    <p class="hint">
      Use this to localise the regression. <strong>A red row</strong> tells you <em>which</em> channel is failing (which model, tenant, or endpoint).
      <strong>A row that goes from green on the left to red on the right</strong> tells you <em>when</em> it started — usually the moment a deploy, retrieval index update, or model swap landed.
      Click a red cell to filter the rest of the dashboard to that channel and see exactly which calls hallucinated.
    </p>
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

  <section class="panel">
    <h2>What hallucinated — and where<span class="help" data-tip="The 12 lowest-scoring calls in your current filter. Each card shows the question, the source context, the model's answer (with claims highlighted), the per-claim verdicts (when detailed mode is on), and any invented citations. Use this to confirm whether the model is wrong, or whether the source context was missing the fact.">?</span></h2>
    <p class="hint">Red highlights in the answer = claims contradicted by the context. Orange = unsupported (context is silent). Green = supported. Invented citations are flagged separately.</p>
    <div class="offender-list" id="offender-list"></div>
  </section>

</div>

<script>
const DATA = __DATA_JSON__;

const FILTER = { tenant: null, model: null, endpoint: null, time: "all" };

const TIME_RANGES = [
  { key: "all", label: "All time",  seconds: null },
  { key: "1h",  label: "Last 1h",   seconds: 60 * 60 },
  { key: "24h", label: "Last 24h",  seconds: 24 * 60 * 60 },
  { key: "7d",  label: "Last 7d",   seconds: 7 * 24 * 60 * 60 },
  { key: "30d", label: "Last 30d",  seconds: 30 * 24 * 60 * 60 },
];

function timeCutoff() {
  const r = TIME_RANGES.find(x => x.key === FILTER.time);
  if (!r || !r.seconds) return 0;
  // Anchor on the newest timestamp in the dataset (works for both live and historical files).
  const maxTs = DATA.rows.reduce((m, r) => Math.max(m, r.ts || 0), 0);
  return maxTs - r.seconds;
}

function applyFilter(rows) {
  const cutoff = timeCutoff();
  return rows.filter(r => {
    if (FILTER.tenant   && r.tenant   !== FILTER.tenant)   return false;
    if (FILTER.model    && r.model    !== FILTER.model)    return false;
    if (FILTER.endpoint && r.endpoint !== FILTER.endpoint) return false;
    if (cutoff && (r.ts || 0) < cutoff) return false;
    return true;
  });
}

function rerender() {
  const filtered = applyFilter(DATA.rows);
  renderHero(filtered);
  renderNarrative(filtered);
  renderMetrics(filtered);
  renderRecommendations(filtered);
  renderRollingChart(filtered);
  renderHeatmaps(filtered);
  renderOffenders(filtered);
  updateChips();
  const fc = document.getElementById("filter-count");
  if (fc) fc.textContent = filtered.length === DATA.rows.length
    ? `${filtered.length} spans` : `${filtered.length} of ${DATA.rows.length} spans`;
}

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
function tsLabel(t) { if (!t) return ""; return new Date(t * 1000).toLocaleString(); }
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
  // Time-range group
  const timeChips = TIME_RANGES.map(r => {
    const active = FILTER.time === r.key ? "active" : "";
    return `<span class="chip ${active}" data-time="${r.key}">${esc(r.label)}</span>`;
  }).join("");
  html += `<div class="filter-group"><span class="filter-label">When</span>${timeChips}</div>`;
  html += `<span class="chip chip-reset" id="chip-reset">Clear filters</span>`;
  html += `<span class="filter-count" id="filter-count"></span>`;
  wrap.innerHTML = html;
  wrap.querySelectorAll(".chip[data-key]").forEach(c => {
    c.addEventListener("click", () => {
      const k = c.dataset.key, v = c.dataset.val;
      FILTER[k] = FILTER[k] === v ? null : v;
      rerender();
    });
  });
  wrap.querySelectorAll(".chip[data-time]").forEach(c => {
    c.addEventListener("click", () => {
      FILTER.time = c.dataset.time;
      rerender();
    });
  });
  document.getElementById("chip-reset").addEventListener("click", () => {
    FILTER.tenant = FILTER.model = FILTER.endpoint = null;
    FILTER.time = "all";
    rerender();
  });
}

function updateChips() {
  document.querySelectorAll(".chip[data-key]").forEach(c => {
    c.classList.toggle("active", FILTER[c.dataset.key] === c.dataset.val);
  });
  document.querySelectorAll(".chip[data-time]").forEach(c => {
    c.classList.toggle("active", FILTER.time === c.dataset.time);
  });
}

function metricStats(rows, key) {
  const vals = rows.map(r => r[key]).filter(v => v != null);
  if (!vals.length) return null;
  const n = vals.length;
  const k = Math.max(1, Math.floor(n / 3));
  const sortedByTs = [...rows].sort((a, b) => a.ts - b.ts).filter(r => r[key] != null);
  const baseline = sortedByTs.slice(0, k).reduce((s, r) => s + r[key], 0) / Math.min(k, sortedByTs.length);
  const current  = sortedByTs.slice(-k).reduce((s, r) => s + r[key], 0) / Math.min(k, sortedByTs.length);
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

let _heroChart = null;
function renderHero(rows) {
  const stats = metricStats(rows, "Hallucination");
  const wrap = document.getElementById("hero");
  if (!stats) {
    wrap.innerHTML = `<div class="hero-left"><div class="health-dot ok"></div>
      <div class="hero-text"><div class="hero-label">Hallucination health</div>
      <div class="hero-value">—</div><div class="hero-tier">no eval scores yet</div>
      <div class="hero-sub">Add <code>peekr.eval.Hallucination()</code> to your evaluators.</div></div></div>`;
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
        <div class="hero-value">${(stats.current * 100).toFixed(0)}<span style="font-size:1.1rem;font-weight:600;color:var(--muted)">/100</span></div>
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

function renderNarrative(rows) {
  const ul = document.getElementById("narrative-list");
  const insights = [];

  const stats = metricStats(rows, "Hallucination");
  if (stats && stats.n >= 6) {
    if (stats.delta < -0.1) {
      insights.push(`Hallucination dropped <strong>${(Math.abs(stats.delta)*100).toFixed(0)} points</strong> from baseline (<strong>${stats.baseline.toFixed(2)}</strong> → <strong>${stats.current.toFixed(2)}</strong>) in the current view.`);
    } else if (stats.delta > 0.1) {
      insights.push(`Hallucination improved <strong>${(stats.delta*100).toFixed(0)} points</strong> from baseline (<strong>${stats.baseline.toFixed(2)}</strong> → <strong>${stats.current.toFixed(2)}</strong>).`);
    }
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
  if (worst) {
    insights.push(`Worst channel: <strong>${esc(worst.key)}</strong> — mean <strong>${worst.mean.toFixed(2)}</strong> across <strong>${worst.n}</strong> calls.`);
  }

  let citTotal = 0, citInv = 0;
  for (const r of rows) {
    const d = r.citation_details; if (!d) continue;
    citTotal += d.total || 0; citInv += d.invented || 0;
  }
  if (citTotal) {
    if (citInv > 0) {
      const rate = citInv / citTotal;
      insights.push(`<strong>${citInv}</strong> of <strong>${citTotal}</strong> citation references were invented (<strong>${(rate*100).toFixed(0)}%</strong>) — URLs, arXiv IDs, or paper titles not in the source context.`);
    } else {
      insights.push(`All <strong>${citTotal}</strong> citation references in the output were grounded in context.`);
    }
  }

  let claimsTotal = 0, contra = 0, unsup = 0;
  for (const r of rows) {
    const d = r.details; if (!d) continue;
    claimsTotal += d.total || 0; contra += d.contradicted || 0; unsup += d.unsupported || 0;
  }
  if (claimsTotal) {
    insights.push(`Of <strong>${claimsTotal}</strong> atomic claims judged in detailed mode, <strong>${contra}</strong> were contradicted by context and <strong>${unsup}</strong> were unsupported.`);
  }

  const errors = rows.filter(r => r.status === "error").length;
  if (errors) {
    insights.push(`<strong>${errors}</strong> LLM call${errors > 1 ? "s" : ""} failed outright (rate-limit, timeout, etc.).`);
  }

  if (!insights.length) insights.push(`All scored signals are within expected ranges for the current filter.`);
  ul.innerHTML = insights.map(s => `<li>${s}</li>`).join("");
}

const METRICS = [
  { key: "Hallucination",    label: "Hallucination", color: "#58a6ff",
    tip: "Mean fraction of the model's factual claims that are grounded in its source context. 0 = fully hallucinating, 1 = fully grounded. Lower → review the worst-offender cards below." },
  { key: "Rubric",           label: "Rubric",        color: "#bc8cff",
    tip: "Custom rubric score from your LLM-as-judge evaluators. Score 0–1; what 'good' means depends on the rubric criteria you set (e.g. 'be concise', 'cite sources')." },
  { key: "CitationAccuracy", label: "Citations",     color: "#3fb950",
    tip: "Fraction of citation patterns in the output (URLs, arXiv IDs, paper titles, 'Author et al. YEAR') that appear in the source context. Pure regex check — fast and free. A low score is a strong signal of invented references." },
];

function metricAction(key, s, rows) {
  if (key === "Hallucination") {
    const flagged = rows.filter(r => r.Hallucination != null && r.Hallucination < 0.5).length;
    if (flagged >= 10)     return { text: `→ ${flagged} calls flagged — review worst offenders below`, cls: "bad" };
    if (flagged > 0)       return { text: `→ ${flagged} flagged — open the worst-offender card`, cls: "warn" };
    if (s && s.delta < -0.1) return { text: `→ regressing — check the heatmap to find the source`, cls: "warn" };
    if (s && s.delta > 0.1)  return { text: `→ improving vs baseline`, cls: "ok" };
    return { text: `→ stable, no action needed`, cls: "flat" };
  }
  if (key === "CitationAccuracy") {
    let inv = 0;
    for (const r of rows) if (r.citation_details) inv += r.citation_details.invented || 0;
    if (inv >= 5) return { text: `→ ${inv} invented refs — likely RAG retrieval miss`, cls: "bad" };
    if (inv > 0)  return { text: `→ ${inv} invented ref${inv>1?"s":""} — see citation list in offenders`, cls: "warn" };
    if (s && s.n > 0) return { text: `→ all detected references are grounded`, cls: "ok" };
    return { text: `→ no citations detected in outputs`, cls: "flat" };
  }
  if (key === "Rubric") {
    if (s && s.current < 0.6) return { text: `→ below 0.6 — review rubric and outputs`, cls: "warn" };
    if (s && s.delta < -0.1)  return { text: `→ regressing — model output quality dropping`, cls: "warn" };
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
  const errAction = err === 0
    ? { text: `→ no failed calls`, cls: "ok" }
    : (err > tot * 0.05 ? { text: `→ error rate > 5% — check rate limits or upstream API`, cls: "bad" }
                        : { text: `→ a few transient failures`, cls: "warn" });
  html += `
    <div class="metric">
      <div class="metric-label">Errors<span class="help" data-tip="Number of LLM calls that raised an exception (rate-limit, timeout, network, malformed response). These are spans with status='error'. >5% sustained usually means an upstream issue.">?</span></div>
      <div class="metric-row">
        <span class="metric-value ${errTier}">${err}</span>
        <span class="metric-delta flat">${tot ? ((err/tot)*100).toFixed(1) : 0}% of calls</span>
      </div>
      <div class="metric-foot" style="margin-top:0.95rem">of ${tot} total calls</div>
      ${errAction.text ? `<div class="metric-action ${errAction.cls}">${errAction.text}</div>` : ""}
    </div>`;
  wrap.innerHTML = html;

  // Destroy old charts BEFORE the new canvases get measured, then defer creation
  // so Chart.js reads the just-rendered parent height once layout is stable.
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
        options: {
          responsive: true, maintainAspectRatio: false, animation: false, resizeDelay: 100,
          plugins: { legend: { display: false }, tooltip: { enabled: false } },
          scales: { y: { min: 0, max: 1, display: false }, x: { display: false } },
        },
      });
    }
  });
}

let _rollingChart = null;
function renderRollingChart(rows) {
  const sorted = [...rows].sort((a,b)=>a.ts-b.ts);
  const labels = sorted.map((_, i) => i + 1);
  const window = 20;

  const halRolling   = rollingMean(sorted.map(r => r.Hallucination),    window);
  const rubRolling   = rollingMean(sorted.map(r => r.Rubric),           window);
  const citRolling   = rollingMean(sorted.map(r => r.CitationAccuracy), window);
  const warningLine  = labels.map(() => 0.7);
  const criticalLine = labels.map(() => 0.5);

  if (_rollingChart) { _rollingChart.destroy(); _rollingChart = null; }
  const ctx = document.getElementById("rolling-chart");
  if (!ctx) return;

  _rollingChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "Hallucination",    data: halRolling, borderColor: "#58a6ff", backgroundColor: "rgba(88,166,255,0.08)", tension: 0.25, pointRadius: 0, borderWidth: 2.5, fill: false, spanGaps: true },
        { label: "Rubric",           data: rubRolling, borderColor: "#bc8cff", tension: 0.25, pointRadius: 0, borderWidth: 1.5, fill: false, spanGaps: true },
        { label: "CitationAccuracy", data: citRolling, borderColor: "#3fb950", tension: 0.25, pointRadius: 0, borderWidth: 1.5, fill: false, spanGaps: true },
        { label: "Warning (0.7)",    data: warningLine,  borderColor: "rgba(255,166,87,0.5)", borderDash: [4,4], borderWidth: 1, pointRadius: 0, fill: false },
        { label: "Critical (0.5)",   data: criticalLine, borderColor: "rgba(248,81,73,0.55)", borderDash: [4,4], borderWidth: 1, pointRadius: 0, fill: false },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false, resizeDelay: 100,
      interaction: { mode: "nearest", intersect: false, axis: "x" },
      onClick(_evt, els) {
        if (!els.length) return;
        const r = sorted[els[0].index];
        if (!r) return;
        const card = document.getElementById(`offender-${r.span_id}`);
        if (card) { card.scrollIntoView({ behavior: "smooth", block: "center" });
          card.style.outline = "2px solid var(--accent)"; setTimeout(() => card.style.outline = "", 1800); }
      },
      plugins: {
        legend: { labels: { color: "#e6edf3", font: { size: 12 } } },
        tooltip: {
          backgroundColor: "rgba(13,17,23,0.95)", borderColor: "#30363d", borderWidth: 1,
          titleColor: "#e6edf3", bodyColor: "#c9d1d9", padding: 10,
          callbacks: {
            title: items => {
              const r = sorted[items[0].dataIndex];
              return `Span #${items[0].dataIndex + 1} · ${r && r.trace_id ? r.trace_id.slice(0,8) : ""}`;
            },
            label: item => {
              const r = sorted[item.dataIndex];
              const dsLabel = item.dataset.label;
              if (dsLabel.startsWith("Warning") || dsLabel.startsWith("Critical")) return null;
              const raw = r ? r[dsLabel] : null;
              return `${dsLabel}: ${fmt(item.parsed.y)} (raw ${fmt(raw)})`;
            },
            afterBody: items => {
              const r = sorted[items[0].dataIndex];
              if (!r) return [];
              return ["",
                `model:    ${r.model || "—"}`,
                `tenant:   ${r.tenant || "—"}`,
                `endpoint: ${r.endpoint || "—"}`,
                `when:     ${tsLabel(r.ts)}`,
                "",
                "click to jump to this trace"];
            },
          },
        },
      },
      scales: {
        y: { min: 0, max: 1, ticks: { color: "#8b949e" }, grid: { color: "#30363d" },
             title: { display: true, text: "score (0 = hallucinating, 1 = grounded)", color: "#8b949e", font: { size: 11 } } },
        x: { ticks: { color: "#8b949e", maxTicksLimit: 14 }, grid: { color: "rgba(48,54,61,0.5)" },
             title: { display: true, text: "span number (oldest → newest)", color: "#8b949e", font: { size: 11 } } },
      },
    },
  });
}

function renderHeatmaps(rows) {
  const wrap = document.getElementById("heatmaps");
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

// ---------------------------------------------------------------------------
// Recommendations engine — diagnose patterns and suggest concrete fixes
// ---------------------------------------------------------------------------
// Each rule inspects the filtered rows and may emit a recommendation card.
// Designed as a drop-in for RAG and memory/agent pipelines: the suggestions
// focus on retrieval (chunking, top-k, hybrid search), prompts (refusal,
// citation discipline), and ops (deploys, model swaps, rate limits).
//
// Order matters — high-severity / actionable ones first.
// ---------------------------------------------------------------------------

function _channelConcentration(rows, field) {
  // Among flagged rows (Hallucination < 0.5), what fraction share a single value
  // of `field`? Returns the dominant value and its share, or null if not concentrated.
  const flagged = rows.filter(r => r.Hallucination != null && r.Hallucination < 0.5 && r[field]);
  if (flagged.length < 4) return null;
  const counts = {};
  for (const r of flagged) counts[r[field]] = (counts[r[field]] || 0) + 1;
  const [val, n] = Object.entries(counts).sort((a,b) => b[1] - a[1])[0];
  return { value: val, share: n / flagged.length, count: n, total: flagged.length };
}

function _claimCounts(rows) {
  let total = 0, supported = 0, contradicted = 0, unsupported = 0;
  for (const r of rows) {
    const d = r.details; if (!d) continue;
    total       += d.total        || 0;
    supported   += d.supported    || 0;
    contradicted+= d.contradicted || 0;
    unsupported += d.unsupported  || 0;
  }
  return { total, supported, contradicted, unsupported };
}

function _citationCounts(rows) {
  let total = 0, grounded = 0, invented = 0;
  for (const r of rows) {
    const d = r.citation_details; if (!d) continue;
    total    += d.total    || 0;
    grounded += d.grounded || 0;
    invented += d.invented || 0;
  }
  return { total, grounded, invented };
}

function diagnoseRecommendations(rows) {
  const recs = [];
  const hStats = metricStats(rows, "Hallucination");
  const errors = rows.filter(r => r.status === "error").length;
  const cit = _citationCounts(rows);
  const claims = _claimCounts(rows);
  const flagged = rows.filter(r => r.Hallucination != null && r.Hallucination < 0.5).length;

  // ── No data ──
  if (!hStats) {
    recs.push({
      severity: "info", title: "No Hallucination scores recorded yet",
      cause: "Either the <code>Hallucination</code> evaluator hasn't been added to your <code>peekr.instrument()</code> call, or the calls in this view didn't pass through it (no context to ground against, or evaluation crashed).",
      checks: [
        `Add the evaluator: <code>peekr.instrument(evaluators=[peekr.eval.Hallucination()])</code>`,
        `For RAG, attach the retrieved chunks as the grounding context via <code>context_extractor=lambda s: s.attributes.get("retrieved_docs", "")</code>`,
        `Enable detailed mode (<code>Hallucination(detailed=True)</code>) on a sample of calls to get per-claim verdicts in the offender cards`,
      ],
    });
    return recs;
  }

  // ── 1. Invented citations dominate ──
  if (cit.total >= 5 && cit.invented / cit.total > 0.3) {
    const rate = cit.invented / cit.total;
    recs.push({
      severity: "high",
      title: `Model is inventing citations (${(rate*100).toFixed(0)}% of detected references)`,
      cause: `Out of <strong>${cit.total}</strong> citation patterns we saw in outputs (URLs, arXiv IDs, "Author et al. YEAR", section numbers), <strong>${cit.invented}</strong> don't appear in the source context. This is the signature of a RAG flow where <strong>retrieval didn't return the actual source</strong>, so the model fabricates plausible-looking references to seem authoritative.`,
      checks: [
        `<strong>Inspect retrieval:</strong> for a flagged span, log the chunks your retriever returned. Are the cited sources actually in there? If not, the model invented them.`,
        `<strong>Tighten the prompt:</strong> add <em>"Do not cite sources unless they appear in the context above. If you don't have a source, say so."</em>`,
        `<strong>Verify citations post-hoc:</strong> after generation, extract every URL/arXiv/DOI from the output and reject the answer if any are missing from the retrieved chunks.`,
        `<strong>Increase top-k</strong> on retrieval (e.g. 5 → 10) if the right doc is being missed.`,
        `<strong>Try hybrid retrieval</strong> (BM25 + dense embeddings) for keyword-heavy queries like names, dates, and identifiers — dense-only often misses these.`,
      ],
      evidence: `${cit.invented} invented / ${cit.total} total citations detected in the current view.`,
    });
  }

  // ── 2. High contradicted-claim rate ──
  if (claims.total >= 6 && claims.contradicted / claims.total > 0.2) {
    const rate = claims.contradicted / claims.total;
    recs.push({
      severity: rate > 0.4 ? "high" : "medium",
      title: `${(rate*100).toFixed(0)}% of claims directly contradict the context`,
      cause: `<strong>${claims.contradicted}</strong> out of <strong>${claims.total}</strong> atomic claims judged in detailed mode are actively contradicted by the context the model was given. This isn't "model didn't know" — it's "model said the opposite of what's in front of it." That usually means context is being <strong>ignored or overridden</strong> by the model's training-data priors.`,
      checks: [
        `<strong>Strengthen the system prompt:</strong> <em>"Answer using only the facts in the context above. If the context contradicts what you 'know', the context wins."</em>`,
        `<strong>Move retrieved context closer to the question</strong> in the prompt — recency bias means models attend more to text near the end of the prompt.`,
        `<strong>Reduce max_tokens / temperature</strong> — long completions tend to drift into hallucination once the grounded part is exhausted.`,
        `<strong>Check chunk quality:</strong> are chunks being cut mid-sentence so key facts are fragmented? Try chunk_size between 256 and 512 tokens with 10–20% overlap.`,
        `<strong>Verify the right doc is retrieved at all</strong> — log the cosine similarity score for the top result; if it's &lt; 0.6, retrieval probably missed.`,
      ],
      evidence: `${claims.contradicted} contradicted, ${claims.unsupported} unsupported, ${claims.supported} supported across ${claims.total} judged claims.`,
    });
  }

  // ── 3. High unsupported (out-of-context elaboration) ──
  if (claims.total >= 6 && claims.unsupported / claims.total > 0.25 && claims.contradicted / claims.total < 0.2) {
    const rate = claims.unsupported / claims.total;
    recs.push({
      severity: "medium",
      title: `${(rate*100).toFixed(0)}% of claims have no support in context`,
      cause: `<strong>${claims.unsupported}</strong> out of <strong>${claims.total}</strong> claims aren't contradicted — but the context is <strong>silent</strong> about them. The model is elaborating beyond what was retrieved, filling in from training data. This is the classic "smart-sounding answer with no evidence" failure mode.`,
      checks: [
        `<strong>Add a refusal instruction:</strong> <em>"If the context doesn't contain the answer, reply 'I don't have that information.'"</em>`,
        `<strong>Check retrieval recall:</strong> the user's question may need facts that simply aren't in your indexed docs. Are you missing a corpus?`,
        `<strong>Try a coverage prompt:</strong> ask the model to first state which parts of the question it can answer from the context, then answer only those.`,
        `<strong>For agents:</strong> verify the right tool was called. The model may be elaborating because a search/lookup step was skipped.`,
      ],
      evidence: `${claims.unsupported} unsupported claims in a context where contradictions are low — model is adding detail, not rewriting it.`,
    });
  }

  // ── 4. Channel concentration: one model / tenant / endpoint dominates the failures ──
  for (const field of ["model", "tenant", "endpoint"]) {
    const conc = _channelConcentration(rows, field);
    if (conc && conc.share > 0.5 && conc.count >= 4) {
      const fieldLabel = field === "user_id" ? "tenant" : field;
      const causeByField = {
        model:    `Most flagged calls came from <strong>one model</strong>. Either this model is intrinsically weaker at grounding (smaller / older / cheaper variants do this), or it was recently switched in for this traffic.`,
        tenant:   `Most flagged calls belong to <strong>one tenant</strong>. Their data may be missing from your retrieval index, their system prompt may differ, or their query distribution may not match what your RAG was tuned for.`,
        endpoint: `Most flagged calls came from <strong>one endpoint</strong>. Code changes for this route — prompt template, retrieval params, top-k, or model override — are the most likely cause.`,
      };
      const checksByField = {
        model: [
          `Compare hallucination rate per model in the <strong>Failure breakdown</strong> heatmap above.`,
          `If this is a smaller / cheaper model, consider routing hard queries to a larger one.`,
          `Check if max_tokens or temperature differs from your healthier model's config.`,
        ],
        tenant: [
          `Verify this tenant's documents are in the retrieval index. <code>SELECT COUNT(*) FROM chunks WHERE tenant_id = '...'</code>`,
          `Diff this tenant's system prompt and retrieval config vs a healthy tenant's.`,
          `Check for tenant-specific model or chunk-size overrides that might be silently degrading quality.`,
        ],
        endpoint: [
          `Diff the last week of deploys touching this endpoint. Look for prompt template, retrieval, or model-config changes.`,
          `Compare this endpoint's prompt with a healthier one — even one word swapped in the system message changes grounding behavior.`,
          `Check if top-k or chunk overlap was tweaked recently.`,
        ],
      };
      recs.push({
        severity: conc.share > 0.75 ? "high" : "medium",
        title: `Failures concentrated on ${fieldLabel} = ${conc.value}`,
        cause: causeByField[field],
        checks: checksByField[field],
        evidence: `${conc.count} of ${conc.total} flagged calls (${(conc.share*100).toFixed(0)}%) belong to this ${fieldLabel}.`,
      });
      break; // Show only the strongest channel concentration to avoid noise.
    }
  }

  // ── 5. Drift: regressed vs baseline ──
  if (hStats.n >= 6 && hStats.delta < -0.1) {
    const dropPct = Math.abs(hStats.delta * 100);
    recs.push({
      severity: dropPct > 25 ? "high" : "medium",
      title: `Hallucination regressed ${dropPct.toFixed(0)} points from baseline`,
      cause: `In this filter, the newest third of calls averages <strong>${hStats.current.toFixed(2)}</strong> vs <strong>${hStats.baseline.toFixed(2)}</strong> for the oldest third. Something changed between them — a deploy, a model swap, a retrieval index update, or a new user cohort onboarded. The heatmap below pinpoints when and where.`,
      checks: [
        `Open the <strong>Failure breakdown</strong> heatmap below. Find the channel row that goes green → red and note the time bucket.`,
        `Cross-reference that time with your deploy log and retrieval-index update history.`,
        `Replay one of the regressed calls with the older prompt / model / index to isolate the variable: <code>peekr replay &lt;trace_id&gt;</code>`,
        `Bisect: filter to "last 1h" and "last 24h" using the time-range chips above to narrow when the regression started.`,
      ],
      evidence: `Δ = ${hStats.delta.toFixed(2)} (baseline ${hStats.baseline.toFixed(2)} → current ${hStats.current.toFixed(2)}, n=${hStats.n})`,
    });
  }

  // ── 6. Error spikes ──
  if (rows.length >= 20 && errors / rows.length > 0.05) {
    recs.push({
      severity: errors / rows.length > 0.2 ? "high" : "medium",
      title: `${errors} of ${rows.length} calls failed outright (${((errors/rows.length)*100).toFixed(1)}%)`,
      cause: `These are spans with <code>status="error"</code> — rate limits, timeouts, malformed responses, or network failures. Error spikes often correlate with hallucination drops if a fallback model is silently kicking in.`,
      checks: [
        `Sort by error in the SQLite store: <code>SELECT name, COUNT(*) FROM spans WHERE status='error' GROUP BY name</code>`,
        `Check the timeout on your LLM client (default 60s may be too short for long-context calls).`,
        `If using a multi-provider gateway, verify the failover provider isn't a weaker model that's silently lowering quality.`,
        `Add a retry-with-backoff layer if these are transient rate limits.`,
      ],
      evidence: `${errors} errors out of ${rows.length} calls in the current view.`,
    });
  }

  // ── 7. Citations look fine ──
  if (cit.total >= 5 && cit.invented === 0) {
    recs.push({
      severity: "good",
      title: `All ${cit.total} detected citations are grounded`,
      cause: `Every URL, arXiv ID, paper title, and statute reference we found in the outputs appears in the source context. Your RAG retrieval is keeping the model honest about sources.`,
      checks: [
        `Keep <code>CitationAccuracy</code> in your evaluator list — it's a cheap continuous signal (no LLM calls).`,
        `Consider adding alerts on the citation invention rate so a regression here pages you immediately.`,
      ],
    });
  }

  // ── 8. Everything is healthy ──
  if (recs.length === 0) {
    recs.push({
      severity: "good",
      title: "All scored signals are within expected ranges",
      cause: `Current Hallucination is <strong>${hStats.current.toFixed(2)}</strong> (baseline ${hStats.baseline.toFixed(2)}). No obvious failure patterns detected for the current filter.`,
      checks: [
        `Set up an alert at <code>Hallucination &lt; 0.5</code> using <code>peekr.alert.ScoreFloor</code> so regressions page you early.`,
        `Run the offline benchmark periodically: <code>python examples/hallucination_benchmark/run.py</code>`,
        `Try toggling on a tenant or model chip to inspect each channel independently — averages can hide tail issues.`,
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
    return `
      <div class="rec ${r.severity}">
        <div class="rec-head">
          <span class="rec-sev ${r.severity}">${sevLabel}</span>
          <span class="rec-title">${r.title}</span>
        </div>
        <div class="rec-cause">${r.cause}</div>
        ${checks}
        ${evidence}
      </div>`;
  }).join("");
}

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

function parseInput(inp) {
  try {
    const messages = JSON.parse(inp);
    const sys = messages.find(m => m.role === "system");
    const usr = messages.find(m => m.role === "user");
    return { context: sys ? sys.content : "", question: usr ? usr.content : "" };
  } catch (_) {
    return { context: inp || "", question: "" };
  }
}

// ---------------------------------------------------------------------------
// Per-span action items — diagnose ONE flagged call and prescribe fixes
// ---------------------------------------------------------------------------
// Each rule inspects this one row (its claims, citations, score, context size)
// and emits {fix, why} items. Designed so the user can read a single offender
// card and know exactly what to try without scrolling back up to the aggregate
// recommendations panel.
// ---------------------------------------------------------------------------

function perSpanActions(row) {
  const out = [];
  const ctx = (() => { try { return parseInput(row.input).context || ""; } catch { return ""; } })();
  const outputText = row.output || "";
  const score = row.Hallucination;

  const details = row.details;
  const claims = (details && details.claims) || [];
  const contraClaims = claims.filter(c => c.verdict === "contradicted");
  const unsupClaims  = claims.filter(c => c.verdict === "unsupported");

  const cd = row.citation_details;
  const inventedCitations = (cd && cd.items) ? cd.items.filter(i => !i.grounded) : [];

  // Detect if the contradicted claims are about numbers / dates / proper nouns
  const numericContra = contraClaims.filter(c => /\b\d{2,}\b/.test(c.text));
  const properNounContra = contraClaims.filter(c => /\b[A-Z][a-z]+\s+[A-Z][a-z]+\b/.test(c.text)); // "Frank Lloyd"

  // 1. Empty / very short context — likely a retrieval miss
  if (ctx.length < 60 && outputText.length > 100) {
    out.push({
      fix: `<strong>Likely retrieval miss:</strong> context is only ${ctx.length} chars but the model produced ${outputText.length} chars of answer. Inspect what your retriever returned for this query and confirm the right chunks are in the index.`,
      why: "When context is sparse and the answer is verbose, the model is filling from training data."
    });
  }

  // 2. Invented citations — kind-specific advice
  if (inventedCitations.length) {
    const kinds = [...new Set(inventedCitations.map(c => c.kind))];
    const kindList = kinds.join(", ");
    const examples = inventedCitations.slice(0, 2).map(c => `<code>${esc(c.text)}</code>`).join(" and ");
    out.push({
      fix: `<strong>Invented ${kindList} citation${inventedCitations.length>1?"s":""}</strong> (${examples}). Add to the system prompt: <em>"Cite only sources present in the context. Do not invent URLs, arXiv IDs, or paper titles."</em> Then post-process: extract every citation pattern from the output and reject if not in context.`,
      why: `Model fabricated references to seem authoritative. Common when retrieval doesn't return real sources or the prompt asks for citations.`
    });
  }

  // 3. Numeric contradictions — copy-from-context advice
  if (numericContra.length) {
    out.push({
      fix: `<strong>${numericContra.length} claim${numericContra.length>1?"s":""} with wrong numbers/dates</strong>. Add: <em>"Be exact about numbers, dates, and quantities. Copy them verbatim from the context — do not round or approximate."</em> Lower <code>temperature</code> to 0 if not already.`,
      why: "Models drift on specific numerics when the prompt doesn't force exact reproduction."
    });
  }

  // 4. Proper-noun contradictions — entity-replacement pattern
  if (properNounContra.length) {
    out.push({
      fix: `<strong>Proper noun substitution</strong> (e.g. the model swapped a name). Add explicit instruction: <em>"Use only the names of people, places, and organizations that appear in the context. Do not substitute similar-sounding alternatives."</em>`,
      why: "When asked about a less-famous entity, models substitute a more-famous one from training data."
    });
  }

  // 5. Mostly unsupported (no contradictions) — needs refusal prompt
  if (claims.length >= 2 && unsupClaims.length / claims.length > 0.5 && contraClaims.length === 0) {
    out.push({
      fix: `<strong>Out-of-context elaboration</strong>: ${unsupClaims.length} of ${claims.length} claims aren't supported, but none are contradicted — the model is adding detail beyond what was retrieved. Add to the system prompt: <em>"If the context doesn't contain the answer, say 'I don't have that information' instead of guessing."</em>`,
      why: "Context is silent on these claims; model filled from training data instead of refusing."
    });
  }

  // 6. Mostly contradicted (model overriding context)
  if (claims.length >= 2 && contraClaims.length / claims.length > 0.5) {
    out.push({
      fix: `<strong>Model is overriding the context.</strong> Move retrieved context closer to the user question in the prompt (recency bias — models attend more to text near the end). Add: <em>"If the context contradicts what you know, the context wins."</em>`,
      why: `${contraClaims.length} of ${claims.length} claims directly conflict with the source — the model treated its training-data prior as more reliable.`
    });
  }

  // 7. Score very low but no detailed claims captured — recommend enabling detailed mode
  if (score != null && score < 0.3 && !details) {
    out.push({
      fix: `Enable <strong>detailed mode</strong> on this evaluator path to get per-claim verdicts: <code>peekr.eval.Hallucination(detailed=True)</code>. You'll see exactly which sentences failed and why.`,
      why: "Simple mode gave a single score but no breakdown — hard to act on without the claim list."
    });
  }

  // 8. Status = error
  if (row.status === "error") {
    out.push({
      fix: `<strong>Call failed</strong> (<code>${esc(row.error || "see span")}</code>). Investigate rate limits, timeouts, or malformed responses. If a fallback model kicked in elsewhere, it may be silently lowering quality.`,
      why: "Failed calls don't produce useful answers and often correlate with quality regressions if fallback paths exist."
    });
  }

  // 9. Long output relative to context — drift territory
  if (ctx.length > 60 && outputText.length > ctx.length * 2) {
    out.push({
      fix: `Output is <strong>${(outputText.length / ctx.length).toFixed(1)}× longer than context</strong>. Reduce <code>max_tokens</code> to force concision; long completions drift once the grounded part is exhausted.`,
      why: "Models start grounded and become more speculative as the completion grows."
    });
  }

  // 10. Healthy-ish span flagged by a strict threshold
  if (score != null && score >= 0.5 && score < 0.7 && out.length === 0) {
    out.push({
      fix: `Score is on the boundary (${score.toFixed(2)}). Review whether your <strong>flagging threshold</strong> is too strict for your use case, or whether this is a real soft regression. Default threshold is 0.5 critical / 0.7 warning.`,
      why: "Borderline cases are often acceptable in user-facing summarization but unacceptable in factual QA — depends on your product."
    });
  }

  return out;
}

function renderOffenders(rows) {
  const wrap = document.getElementById("offender-list");
  const scored = rows.filter(r => r.Hallucination != null);
  const worst = [...scored].sort((a,b) => a.Hallucination - b.Hallucination).slice(0, 12);
  if (!worst.length) {
    wrap.innerHTML = '<div class="empty">No spans match the current filter, or no Hallucination scores recorded yet.</div>';
    return;
  }
  let html = "";
  worst.forEach((r, i) => {
    const tier = tierForScore(r.Hallucination);
    const { context, question } = parseInput(r.input);
    const claims = r.details && r.details.claims ? r.details.claims : null;
    const answerHtml = claims ? highlightAll(r.output || "", claims) : esc(r.output || "");
    let claimsHtml = "";
    if (claims && claims.length) {
      claimsHtml = `<div class="claims-list" style="margin-top:0.5rem">` + claims.map(c =>
        `<div class="claim ${c.verdict}"><span class="v">${c.verdict}</span><span>${esc(c.text)}</span></div>`
      ).join("") + `</div>`;
    }
    let citationsHtml = "";
    const cd = r.citation_details;
    if (cd && cd.items && cd.items.length) {
      citationsHtml = `<div class="citations-list" style="margin-top:0.7rem">` + cd.items.map(it =>
        `<div class="citation ${it.grounded ? "grounded" : "invented"}">
          <span class="kind">${esc(it.kind)}</span>
          <code>${esc(it.text)}</code>
          <span class="badge">${it.grounded ? "grounded" : "invented"}</span>
        </div>`
      ).join("") + `</div>`;
    }

    // Per-span action items derived from this one call's failure pattern
    const actions = perSpanActions(r);
    let actionsHtml = "";
    if (actions.length) {
      actionsHtml = `
        <div class="span-actions">
          <div class="span-actions-label">What to try for this call</div>
          <ul>${actions.map(a => `<li>${a.fix}<span class="why">${a.why}</span></li>`).join("")}</ul>
        </div>`;
    }

    html += `
      <div class="offender" id="offender-${esc(r.span_id || ("idx"+i))}">
        <div class="offender-head">
          <div style="display:flex;align-items:center;gap:0.6rem">
            <span class="offender-rank">${i + 1}</span>
            <span class="offender-score ${tier}">${r.Hallucination.toFixed(2)}</span>
            <span class="sub" style="font-size:0.78rem">${tsLabel(r.ts)}</span>
          </div>
          <div class="offender-meta">
            <span class="tag">${esc(r.model || "—")}</span>
            <span class="tag">${esc(r.tenant || "—")}</span>
            <span class="tag">${esc(r.endpoint || "—")}</span>
            ${r.status === "error" ? '<span class="tag" style="color:var(--red);border-color:var(--red)">error</span>' : ""}
          </div>
        </div>
        <div class="offender-body">
          ${question ? `<div class="offender-q"><strong>Q:</strong>${esc(question)}</div>` : ""}
          <div class="offender-grid">
            <div class="text-panel">
              <div class="text-panel-label"><span>Source context</span><span style="color:var(--muted);font-weight:600">${context.length}c</span></div>
              <div class="text-panel-content">${esc(context)}</div>
            </div>
            <div class="text-panel">
              <div class="text-panel-label"><span>Model answer</span><span style="color:var(--muted);font-weight:600">${(r.output||"").length}c</span></div>
              <div class="text-panel-content">${answerHtml}</div>
            </div>
          </div>
          ${claimsHtml}
          ${citationsHtml}
          ${actionsHtml}
        </div>
      </div>`;
  });
  wrap.innerHTML = html;
}

renderFilterBar();
rerender();
</script>

</body>
</html>
"""


