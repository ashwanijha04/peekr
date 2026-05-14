from __future__ import annotations
import json
import sqlite3
import sys
from collections import defaultdict


def main():
    if len(sys.argv) < 2:
        print("Usage: peekr <command> [options]")
        print("Commands: view, replay, cost, dashboard")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "view":
        args = sys.argv[2:]
        show_io = "--io" in args
        args = [a for a in args if not a.startswith("--")]
        path = args[0] if args else _default_path()
        view_traces(path, show_io=show_io)
    elif cmd == "replay":
        _cmd_replay(sys.argv[2:])
    elif cmd == "cost":
        args = sys.argv[2:]
        args = [a for a in args if not a.startswith("--")]
        path = args[0] if args else _default_path()
        _cmd_cost(path)
    elif cmd == "dashboard":
        _cmd_dashboard(sys.argv[2:])
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


def _cmd_dashboard(args: list[str]) -> None:
    """peekr dashboard [path] [-o report.html] — emit a static HTML report."""
    output = "dashboard.html"
    rest: list[str] = []
    i = 0
    while i < len(args):
        if args[i] in ("-o", "--out") and i + 1 < len(args):
            output = args[i + 1]
            i += 2
        else:
            rest.append(args[i])
            i += 1
    path = rest[0] if rest else _default_path()

    from .dashboard import generate_dashboard  # noqa: PLC0415
    out = generate_dashboard(path, output=output)
    print(f"Dashboard written to {out}")
    print(f"Open it with:  open {out}")


def _cmd_replay(args: list[str]) -> None:
    """Handle: peekr replay <trace_id> [--db traces.db] [--jsonl traces.jsonl]"""
    if not args or args[0].startswith("--"):
        print("Usage: peekr replay <trace_id> [--db traces.db] [--jsonl traces.jsonl]")
        sys.exit(1)

    trace_id = args[0]
    rest = args[1:]

    db_path = None
    jsonl_path = None
    i = 0
    while i < len(rest):
        if rest[i] == "--db" and i + 1 < len(rest):
            db_path = rest[i + 1]
            i += 2
        elif rest[i] == "--jsonl" and i + 1 < len(rest):
            jsonl_path = rest[i + 1]
            i += 2
        else:
            i += 1

    from .replay import replay_trace  # noqa: PLC0415
    try:
        new_trace_id = replay_trace(
            trace_id=trace_id,
            db_path=db_path,
            jsonl_path=jsonl_path,
        )
    except Exception as exc:
        print(f"Replay failed: {exc}")
        sys.exit(1)

    print(f"Replayed trace {trace_id[:8]} → new trace {new_trace_id[:8]}")
    print()

    # Show the new trace from the same storage
    storage_path = db_path or jsonl_path or _default_path()
    view_traces(storage_path)


def _cmd_cost(path: str) -> None:
    """peekr cost <traces.jsonl|traces.db> — cost breakdown + top-10 hotspots."""
    if path.endswith(".db"):
        spans = _read_sqlite(path)
    else:
        spans = _read_jsonl(path)

    if not spans:
        return

    # ── per-span cost records (LLM calls only) ────────────────────────────────
    records = []
    for s in spans:
        attrs = s.get("attributes") or {}
        inp = attrs.get("tokens_input", 0)
        out = attrs.get("tokens_output", 0)
        dur = s.get("duration_ms") or 0.0
        cost = (inp / 1_000_000 * 0.80) + (out / 1_000_000 * 4.00)
        records.append({
            "name": s["name"],
            "model": attrs.get("model", ""),
            "status": s.get("status", "ok"),
            "tokens_input": inp,
            "tokens_output": out,
            "tokens_total": inp + out,
            "cost": cost,
            "duration_ms": dur,
        })

    llm_records = [r for r in records if r["tokens_total"] > 0]
    total_cost = sum(r["cost"] for r in llm_records)
    total_input = sum(r["tokens_input"] for r in llm_records)
    total_output = sum(r["tokens_output"] for r in llm_records)
    total_dur = sum(r["duration_ms"] for r in llm_records)
    errors = sum(1 for r in records if r["status"] == "error")

    # ── summary ───────────────────────────────────────────────────────────────
    W = 60
    print()
    print("─" * W)
    print(f"  peekr cost  ·  {path}")
    print("─" * W)
    print(f"  Total spans        : {len(spans):,}")
    print(f"  LLM calls          : {len(llm_records):,}")
    print(f"  Errors             : {errors}")
    print(f"  Total input tokens : {total_input:,}")
    print(f"  Total output tokens: {total_output:,}")
    print(f"  Total LLM time     : {total_dur/1000:.1f}s")
    print(f"  Total cost (est.)  : ${total_cost:.5f}  (Haiku rates: $0.80/$4.00 per M)")
    print("─" * W)

    # ── breakdown by operation ─────────────────────────────────────────────────
    by_op: dict[str, dict] = defaultdict(lambda: {
        "calls": 0, "input": 0, "output": 0, "cost": 0.0,
        "duration_ms": 0.0, "errors": 0,
    })
    for r in records:
        key = f"{r['name']}" + (f"  [{r['model']}]" if r["model"] else "")
        by_op[key]["calls"] += 1
        by_op[key]["input"] += r["tokens_input"]
        by_op[key]["output"] += r["tokens_output"]
        by_op[key]["cost"] += r["cost"]
        by_op[key]["duration_ms"] += r["duration_ms"]
        by_op[key]["errors"] += 1 if r["status"] == "error" else 0

    print()
    print("  Cost by operation:")
    print(f"  {'Operation':<48} {'Calls':>5}  {'Cost':>9}  {'Avg/call':>9}  {'Avg ms':>7}")
    print("  " + "─" * (W - 2))
    for op, s in sorted(by_op.items(), key=lambda x: -x[1]["cost"]):
        avg_cost = s["cost"] / max(s["calls"], 1)
        avg_ms = s["duration_ms"] / max(s["calls"], 1)
        err_flag = "  \033[31m(!)\033[0m" if s["errors"] else ""
        print(f"  {op:<48} {s['calls']:>5}  ${s['cost']:>8.5f}  ${avg_cost:>8.5f}  {avg_ms:>6.0f}ms{err_flag}")

    # ── top 10 hotspots ───────────────────────────────────────────────────────
    def _hotspot_score(r: dict) -> float:
        # normalise cost and latency, weight cost 60% / latency 40%
        max_cost = max((x["cost"] for x in llm_records), default=1) or 1
        max_dur = max((x["duration_ms"] for x in llm_records), default=1) or 1
        return 0.6 * (r["cost"] / max_cost) + 0.4 * (r["duration_ms"] / max_dur)

    if llm_records:
        ranked = sorted(llm_records, key=_hotspot_score, reverse=True)[:10]
        print()
        print("  Top 10 hottest calls  (60% cost · 40% latency):")
        print(f"  {'#':<3} {'Operation':<40} {'In':>7} {'Out':>6} {'Cost':>9} {'ms':>7}  {'Model'}")
        print("  " + "─" * (W - 2))
        for i, r in enumerate(ranked, 1):
            err = " \033[31m!\033[0m" if r["status"] == "error" else ""
            print(
                f"  {i:<3} {r['name']:<40} "
                f"{r['tokens_input']:>7,} {r['tokens_output']:>6,} "
                f"${r['cost']:>8.5f} {r['duration_ms']:>6.0f}ms  "
                f"{r['model']}{err}"
            )

        # top 5 slowest (if different from hottest)
        slowest = sorted(llm_records, key=lambda x: -x["duration_ms"])[:5]
        if slowest[0] != ranked[0]:
            print()
            print("  Top 5 slowest calls:")
            print(f"  {'#':<3} {'Operation':<40} {'ms':>7}  {'Tokens':>7}  {'Cost':>9}")
            print("  " + "─" * (W - 2))
            for i, r in enumerate(slowest, 1):
                print(
                    f"  {i:<3} {r['name']:<40} "
                    f"{r['duration_ms']:>6.0f}ms  "
                    f"{r['tokens_total']:>7,}  "
                    f"${r['cost']:>8.5f}"
                )

    print()


def _default_path() -> str:
    import os
    if os.path.exists("traces.db"):
        return "traces.db"
    return "traces.jsonl"


def view_traces(path: str, show_io: bool = False):
    if path.endswith(".db"):
        spans = _read_sqlite(path)
    else:
        spans = _read_jsonl(path)

    if not spans:
        return

    traces = defaultdict(list)
    for span in spans:
        traces[span["trace_id"]].append(span)

    for i, (trace_id, trace_spans) in enumerate(traces.items()):
        if i > 0:
            print()
        total_ms = sum(s.get("duration_ms") or 0 for s in trace_spans if s["parent_id"] is None)
        total_tokens = sum((s.get("attributes") or {}).get("tokens_total", 0) for s in trace_spans)
        token_str = f"  {total_tokens} tokens" if total_tokens else ""
        print(f"Trace {trace_id[:8]}  {total_ms:.0f}ms{token_str}")
        print("─" * 48)
        roots = [s for s in trace_spans if s["parent_id"] is None]
        for root in roots:
            _print_span(root, trace_spans, indent=0, show_io=show_io)


def _read_jsonl(path: str) -> list[dict]:
    try:
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"No traces file at {path}")
        return []


def _read_sqlite(path: str) -> list[dict]:
    try:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM spans ORDER BY start_time").fetchall()
        conn.close()
        spans = []
        for r in rows:
            s = dict(r)
            s["attributes"] = json.loads(s["attributes"] or "{}")
            spans.append(s)
        return spans
    except (sqlite3.OperationalError, FileNotFoundError):
        print(f"No traces database at {path}")
        return []


def _print_span(span, all_spans, indent, show_io):
    duration = f"{span['duration_ms']:.0f}ms" if span.get("duration_ms") else "  ?"
    attrs = span.get("attributes") or {}
    model = f" [{attrs['model']}]" if "model" in attrs else ""
    tokens = f" {attrs['tokens_total']}tok" if "tokens_total" in attrs else ""
    error = " \033[31mERROR\033[0m" if span["status"] == "error" else ""

    connector = "└─ " if indent > 0 else ""
    prefix = "   " * indent + connector
    print(f"{prefix}\033[1m{span['name']}\033[0m{model}  {duration}{tokens}{error}")

    if show_io:
        io_prefix = "   " * (indent + 1)
        if "input" in attrs:
            print(f"{io_prefix}\033[2min:  {attrs['input'][:120]}\033[0m")
        if "output" in attrs:
            print(f"{io_prefix}\033[2mout: {attrs['output'][:120]}\033[0m")
        if "error" in attrs and span["status"] == "error":
            print(f"{io_prefix}\033[31merr: {attrs['error']}\033[0m")

    children = [s for s in all_spans if s.get("parent_id") == span["span_id"]]
    for child in children:
        _print_span(child, all_spans, indent + 1, show_io)
