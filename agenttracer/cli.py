import json
import sys
from collections import defaultdict


def main():
    if len(sys.argv) < 2:
        print("Usage: agenttracer view <traces.jsonl>")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "view":
        path = sys.argv[2] if len(sys.argv) > 2 else "traces.jsonl"
        view_traces(path)


def view_traces(path: str):
    try:
        with open(path) as f:
            spans = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"No traces file found at {path}")
        return

    # Group by trace_id
    traces = defaultdict(list)
    for span in spans:
        traces[span["trace_id"]].append(span)

    for trace_id, trace_spans in traces.items():
        print(f"\nTrace {trace_id[:8]}")
        print("─" * 40)
        by_id = {s["span_id"]: s for s in trace_spans}
        roots = [s for s in trace_spans if s["parent_id"] is None]
        for root in roots:
            _print_span(root, by_id, trace_spans, indent=0)


def _print_span(span, by_id, all_spans, indent):
    duration = f"{span['duration_ms']:.0f}ms" if span.get("duration_ms") else "?"
    attrs = span.get("attributes", {})
    model = f" [{attrs['model']}]" if "model" in attrs else ""
    tokens = f" {attrs['tokens_total']} tokens" if "tokens_total" in attrs else ""
    status = " ERROR" if span["status"] == "error" else ""
    prefix = "  " * indent + ("└─ " if indent > 0 else "")
    print(f"{prefix}{span['name']}{model} {duration}{tokens}{status}")

    children = [s for s in all_spans if s.get("parent_id") == span["span_id"]]
    for child in children:
        _print_span(child, by_id, all_spans, indent + 1)
