"""Live benchmark for peekr's Hallucination evaluator.

Loads `dataset.jsonl` (context + question + answer + label) and runs the
evaluator against every row, then reports classification metrics treating
"hallucinated" as the positive class.

Threshold rule: a row is *predicted* hallucinated when score < `--threshold`
(default 0.5). Faithful otherwise.

Requires OPENAI_API_KEY or ANTHROPIC_API_KEY in the environment. Detailed
mode (RAGAS-style) uses one judge call per row with up to 800 output tokens;
simple mode uses one judge call with 10 output tokens.

Usage:
    python examples/hallucination_benchmark/run.py
    python examples/hallucination_benchmark/run.py --mode detailed
    python examples/hallucination_benchmark/run.py --mode both --threshold 0.7
    python examples/hallucination_benchmark/run.py --limit 10           # quick smoke
    python examples/hallucination_benchmark/run.py --csv results.csv    # dump per-row
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

# Allow running from the repo root without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from peekr.eval.hallucination import Hallucination  # noqa: E402
from peekr.span import Span  # noqa: E402


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def confusion(rows: list[dict], threshold: float, score_key: str) -> dict:
    """
    Positive class: 'hallucinated'. A row is predicted positive when
    score < threshold (lower score → more hallucinated).
    """
    tp = fp = tn = fn = 0
    for r in rows:
        if r[score_key] is None:
            continue
        pred_halluc = r[score_key] < threshold
        gold_halluc = r["label"] == "hallucinated"
        if pred_halluc and gold_halluc:   tp += 1
        elif pred_halluc and not gold_halluc: fp += 1
        elif not pred_halluc and gold_halluc: fn += 1
        else: tn += 1

    n = tp + fp + tn + fn
    accuracy  = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "n": n}


def roc_auc(rows: list[dict], score_key: str) -> float | None:
    """Simple AUC: probability a faithful row has a higher score than a hallucinated row.
    For two classes; we treat 'hallucinated' as positive (low score)."""
    pos = [r[score_key] for r in rows if r["label"] == "hallucinated" and r[score_key] is not None]
    neg = [r[score_key] for r in rows if r["label"] == "faithful"     and r[score_key] is not None]
    if not pos or not neg:
        return None
    # We want score for negatives > score for positives; AUC is that probability.
    wins = ties = 0
    for n in neg:
        for p in pos:
            if n > p:   wins += 1
            elif n == p: ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def fmt_pct(x: float | None) -> str:
    return "—" if x is None else f"{x*100:5.1f}%"


def fmt_score(x: float | None) -> str:
    return "—" if x is None else f"{x:.3f}"


def print_report(rows: list[dict], threshold: float, mode: str, score_key: str) -> None:
    cm = confusion(rows, threshold, score_key)
    auc = roc_auc(rows, score_key)
    scored = [r for r in rows if r[score_key] is not None]
    pos_scores = [r[score_key] for r in scored if r["label"] == "hallucinated"]
    neg_scores = [r[score_key] for r in scored if r["label"] == "faithful"]

    def m(xs):
        return (f"mean={statistics.mean(xs):.3f}  stdev={statistics.pstdev(xs):.3f}"
                if xs else "no data")

    print()
    print("─" * 72)
    print(f"  Hallucination benchmark   ·   mode={mode}   ·   threshold={threshold}")
    print("─" * 72)
    print(f"  Rows scored: {cm['n']}/{len(rows)}")
    print()
    print(f"               Predicted")
    print(f"               halluc   faithful")
    print(f"  Actual halluc   {cm['tp']:>4}     {cm['fn']:>4}")
    print(f"  Actual faithful {cm['fp']:>4}     {cm['tn']:>4}")
    print()
    print(f"  Accuracy   : {fmt_pct(cm['accuracy'])}")
    print(f"  Precision  : {fmt_pct(cm['precision'])}    (of predicted-halluc, share that truly were)")
    print(f"  Recall     : {fmt_pct(cm['recall'])}    (of true-halluc, share we caught)")
    print(f"  F1         : {fmt_pct(cm['f1'])}")
    print(f"  ROC-AUC    : {fmt_pct(auc)}")
    print()
    print(f"  Score on hallucinated rows : {m(pos_scores)}")
    print(f"  Score on faithful rows     : {m(neg_scores)}")
    print()


def print_per_row(rows: list[dict], threshold: float, score_key: str) -> None:
    print(f"{'#':<3}  {'category':<26}  {'diff':<7}  {'gold':<12}  {'score':<7}  {'pred':<12}  {'✓/✗'}  question")
    print("─" * 130)
    for r in rows:
        s = r[score_key]
        pred = "—" if s is None else ("hallucinated" if s < threshold else "faithful")
        ok = "✓" if pred == r["label"] else "✗" if s is not None else " "
        score_str = fmt_score(s)
        cat = r.get("category", "—")
        diff = r.get("difficulty", "—")
        print(f"{r['id']:<3}  {cat:<26}  {diff:<7}  {r['label']:<12}  {score_str:<7}  {pred:<12}  {ok}    {r['question']}")
    print()


def print_breakdown_by(
    rows: list[dict],
    field: str,
    threshold: float,
    score_key: str,
) -> None:
    """Per-segment metrics — useful for spotting categories the detector fails on."""
    by: dict[str, list[dict]] = {}
    for r in rows:
        key = r.get(field, "—")
        by.setdefault(key, []).append(r)

    print(f"  By {field}:")
    print(f"    {'segment':<28}  {'n':>4}  {'acc':>7}  {'prec':>7}  {'rec':>7}  {'f1':>7}  {'auc':>7}")
    print("    " + "─" * 84)
    for seg, items in sorted(by.items()):
        cm = confusion(items, threshold, score_key)
        auc = roc_auc(items, score_key)
        print(f"    {seg:<28}  {cm['n']:>4}  {fmt_pct(cm['accuracy']):>7}  "
              f"{fmt_pct(cm['precision']):>7}  {fmt_pct(cm['recall']):>7}  "
              f"{fmt_pct(cm['f1']):>7}  {fmt_pct(auc):>7}")
    print()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _row_to_span(row: dict) -> Span:
    """Build a peekr Span the same way our patches would for a real LLM call."""
    messages = [
        {"role": "system", "content": row["context"]},
        {"role": "user",   "content": row["question"]},
    ]
    span = Span(name="openai.chat.completions", trace_id=f"bench-{row['id']}")
    span.attributes["model"] = "benchmark"
    span.attributes["input"] = json.dumps(messages)
    span.attributes["output"] = row["answer"]
    span.finish()
    return span


def run_mode(rows: list[dict], mode: str, model: str | None, sleep: float) -> list[dict]:
    """
    Mutates each row, adding f'score_{mode}' and (for detailed) 'details_{mode}'.
    """
    evaluator = Hallucination(detailed=(mode == "detailed"), model=model)
    score_key = f"score_{mode}"
    detail_key = f"details_{mode}"

    print(f"Running mode={mode} on {len(rows)} rows...")
    for i, row in enumerate(rows, 1):
        span = _row_to_span(row)
        t0 = time.time()
        try:
            score = evaluator.evaluate(span)
            row[score_key] = score
            if mode == "detailed":
                row[detail_key] = span.attributes.get("hallucination_details")
        except Exception as e:
            row[score_key] = None
            row.setdefault("errors", []).append(f"{mode}: {e!r}")
        dt = (time.time() - t0) * 1000
        marker = "·" if row[score_key] is not None else "x"
        print(f"  [{i:>3}/{len(rows)}] {marker} id={row['id']:>2}  gold={row['label']:<12}  score={fmt_score(row[score_key])}  ({dt:.0f}ms)")
        if sleep:
            time.sleep(sleep)
    print()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(Path(__file__).parent / "dataset.jsonl"))
    ap.add_argument("--mode", choices=["simple", "detailed", "both"], default="both",
                    help="Evaluator mode to benchmark (default: both)")
    ap.add_argument("--model", default=None,
                    help="Override judge model (e.g. gpt-4o-mini, gpt-4o, claude-haiku-4-5-20251001)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Score threshold: < threshold → predicted hallucinated (default 0.5)")
    ap.add_argument("--limit", type=int, default=None, help="Cap rows for a quick smoke test")
    ap.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between API calls")
    ap.add_argument("--csv", default=None, help="Write per-row results to this CSV")
    ap.add_argument("--no-per-row", action="store_true", help="Skip the per-row table")
    args = ap.parse_args()

    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("ERROR: set OPENAI_API_KEY or ANTHROPIC_API_KEY before running.")
        sys.exit(2)

    with open(args.dataset) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        rows = rows[: args.limit]

    modes = ["simple", "detailed"] if args.mode == "both" else [args.mode]

    t_start = time.time()
    for mode in modes:
        run_mode(rows, mode, model=args.model, sleep=args.sleep)

    elapsed = time.time() - t_start
    print(f"Total wall-clock: {elapsed:.1f}s")

    for mode in modes:
        score_key = f"score_{mode}"
        print_report(rows, threshold=args.threshold, mode=mode, score_key=score_key)
        print_breakdown_by(rows, "category",   threshold=args.threshold, score_key=score_key)
        print_breakdown_by(rows, "difficulty", threshold=args.threshold, score_key=score_key)
        if not args.no_per_row:
            print_per_row(rows, threshold=args.threshold, score_key=score_key)

    if args.csv:
        keys = ["id", "topic", "label", "question", "answer"]
        for mode in modes:
            keys.append(f"score_{mode}")
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow(row)
        print(f"Per-row results written to {args.csv}")


if __name__ == "__main__":
    main()
