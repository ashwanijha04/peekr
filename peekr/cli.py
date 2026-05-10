from __future__ import annotations
import json
import sqlite3
import sys
from collections import defaultdict


def main():
    if len(sys.argv) < 2:
        print("Usage: peekr <command> [options]")
        print("Commands: view, replay")
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
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


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
