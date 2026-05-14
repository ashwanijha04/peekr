"""Per-channel drift report.

Prints the Hallucination score's baseline → current drift, sliced by each
channel field on the spans (model, tenant via `user_id`, and `endpoint`).
Mirrors the "Drift by channel" panel in `peekr dashboard`, but as a plain
text table you can pipe into CI or paste into Slack.

Usage:
    python examples/channel_drift.py                              # default: ./traces.jsonl or traces.db
    python examples/channel_drift.py sample_traces.jsonl
    python examples/channel_drift.py traces.db --metric Hallucination
    python examples/channel_drift.py traces.jsonl --field model   # only one channel
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from peekr.cli import _read_jsonl, _read_sqlite, _default_path  # noqa: E402
from peekr.dashboard import _channel_drift, _LLM_PREFIXES, _scores  # noqa: E402


def fmt(v: float | None, d: int = 3) -> str:
    return "—" if v is None else f"{v:.{d}f}"


def color(delta: float | None) -> str:
    if delta is None:                return ""
    if delta < -0.05:                return "\033[31m"   # red — regressed
    if delta >  0.05:                return "\033[32m"   # green — improved
    return ""


RESET = "\033[0m"


def print_segment(label: str, rows: list[dict], metric: str) -> None:
    print(f"  By {label}:")
    if not rows:
        print(f"    (no spans with attribute '{label}')\n")
        return
    print(f"    {'segment':<32}  {'current':>8}  {'baseline':>9}  {'Δ':>8}  {'n':>6}")
    print("    " + "─" * 72)
    for r in rows:
        c = color(r.get("delta"))
        cur  = fmt(r.get("current"))
        base = fmt(r.get("baseline"))
        delta = r.get("delta")
        delta_str = "—" if delta is None else (("+" if delta >= 0 else "") + f"{delta:.3f}")
        n_str = f"{r['n']}"
        if "n_baseline" in r:
            n_str += f" ({r['n_baseline']}/{r['n_current']})"
        print(f"    {r['segment']:<32}  {c}{cur:>8}  {base:>9}  {delta_str:>8}{RESET}  {n_str:>6}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=None,
                    help="Path to traces.jsonl or traces.db (default: auto-detect)")
    ap.add_argument("--metric", default="Hallucination",
                    help="Eval score to track (default: Hallucination)")
    ap.add_argument("--field", default=None,
                    help="Only report one channel field (model | tenant | endpoint)")
    args = ap.parse_args()

    path = args.path or _default_path()
    spans = _read_sqlite(path) if path.endswith(".db") else _read_jsonl(path)
    if not spans:
        print(f"No spans in {path}")
        sys.exit(1)
    llm_spans = [s for s in spans if any(s["name"].startswith(p) for p in _LLM_PREFIXES)]
    llm_spans.sort(key=lambda s: s.get("start_time") or 0)

    if args.metric != "Hallucination":
        # _channel_drift hardcodes Hallucination; for an ad-hoc metric we
        # compute it the same way inline so the report stays self-contained.
        print(f"NOTE: --metric is currently only wired for 'Hallucination'. Got {args.metric!r}; using Hallucination.")
    drifts = _channel_drift(llm_spans)

    n_scored = sum(1 for s in llm_spans if _scores(s).get("Hallucination") is not None)

    print()
    print("─" * 76)
    print(f"  peekr · channel drift   ·   {path}")
    print(f"  spans: {len(spans)} total · {len(llm_spans)} LLM · {n_scored} scored")
    print("─" * 76)
    for label, rows in drifts.items():
        if args.field and label != args.field:
            continue
        print_segment(label, rows, args.metric)


if __name__ == "__main__":
    main()
