"""Peekr MCP server — expose your local Peekr traces to AI assistants and agents
(Claude Desktop, IDEs, agent frameworks) over the Model Context Protocol (MCP).

The server reads your trace store **read-only** — it never writes spans, never
mutates anything, and never sends data anywhere. It just lets an assistant ask
questions like "show me the worst hallucinations" or "what's my token usage by
model" against the traces Peekr already captured.

Run it::

    peekr-mcp --db traces.db          # SQLite store (default if traces.db exists)
    peekr-mcp --jsonl traces.jsonl    # JSONL store

Then point an MCP client at the ``peekr-mcp`` command. Example (Claude Desktop
``claude_desktop_config.json``)::

    {
      "mcpServers": {
        "peekr": { "command": "peekr-mcp", "args": ["--db", "/path/to/traces.db"] }
      }
    }

The ``mcp`` dependency is only needed to *run* the server (``pip install
"peekr[mcp]"``). The query layer (:class:`TraceStore`) has no such dependency, so
it stays importable and testable everywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import defaultdict
from typing import Any

DEFAULT_DB = "traces.db"
DEFAULT_JSONL = "traces.jsonl"

_SPAN_COLS = (
    "trace_id",
    "span_id",
    "parent_id",
    "name",
    "start_time",
    "end_time",
    "duration_ms",
    "status",
    "attributes",
    "tenant_id",
    "retention_class",
)


def _truncate(value: Any, limit: int = 240) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "…"
    return value


class TraceStore:
    """Read-only reader for a Peekr trace store (SQLite ``traces.db`` or JSONL).

    No ``mcp`` dependency — pure stdlib so it imports and tests anywhere.
    """

    def __init__(self, db_path: str | None = None, jsonl_path: str | None = None):
        self.db_path = db_path
        self.jsonl_path = jsonl_path
        if not db_path and not jsonl_path:
            if os.path.exists(DEFAULT_DB):
                self.db_path = DEFAULT_DB
            elif os.path.exists(DEFAULT_JSONL):
                self.jsonl_path = DEFAULT_JSONL

    # ── loading ──────────────────────────────────────────────────────────────
    def _rows(self) -> list[dict]:
        if self.db_path:
            if not os.path.exists(self.db_path):
                return []
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            conn.row_factory = sqlite3.Row
            try:
                rows = [dict(r) for r in conn.execute("SELECT * FROM spans")]
            except sqlite3.OperationalError:
                return []  # no spans table yet
            finally:
                conn.close()
        elif self.jsonl_path:
            if not os.path.exists(self.jsonl_path):
                return []
            rows = []
            with open(self.jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        else:
            return []

        for r in rows:
            a = r.get("attributes")
            if isinstance(a, str):
                try:
                    r["attributes"] = json.loads(a) if a else {}
                except json.JSONDecodeError:
                    r["attributes"] = {}
            elif a is None:
                r["attributes"] = {}
        return rows

    @staticmethod
    def _hallucination(attrs: dict) -> float | None:
        scores = attrs.get("eval_scores")
        if isinstance(scores, dict):
            v = scores.get("Hallucination")
            if isinstance(v, (int, float)):
                return float(v)
        return None

    # ── tools ────────────────────────────────────────────────────────────────
    def recent_traces(self, limit: int = 20) -> list[dict]:
        """Most recent traces with a one-line summary each."""
        by_trace: dict[str, list[dict]] = defaultdict(list)
        for r in self._rows():
            by_trace[r["trace_id"]].append(r)

        out = []
        for trace_id, spans in by_trace.items():
            spans.sort(key=lambda s: s.get("start_time") or 0)
            root = next((s for s in spans if not s.get("parent_id")), spans[0])
            tokens = sum((s["attributes"].get("tokens_total") or 0) for s in spans)
            models = sorted(
                {
                    s["attributes"].get("model")
                    for s in spans
                    if s["attributes"].get("model")
                }
            )
            out.append(
                {
                    "trace_id": trace_id,
                    "name": root.get("name"),
                    "spans": len(spans),
                    "tokens_total": tokens,
                    "models": models,
                    "has_error": any(s.get("status") == "error" for s in spans),
                    "start_time": root.get("start_time"),
                    "duration_ms": root.get("duration_ms"),
                }
            )
        out.sort(key=lambda t: t.get("start_time") or 0, reverse=True)
        return out[:limit]

    def get_trace(self, trace_id: str) -> list[dict]:
        """Every span in one trace (the waterfall), oldest first."""
        spans = [r for r in self._rows() if r["trace_id"] == trace_id]
        spans.sort(key=lambda s: s.get("start_time") or 0)
        result = []
        for s in spans:
            a = s["attributes"]
            result.append(
                {
                    "span_id": s.get("span_id"),
                    "parent_id": s.get("parent_id"),
                    "name": s.get("name"),
                    "status": s.get("status"),
                    "duration_ms": s.get("duration_ms"),
                    "model": a.get("model"),
                    "tokens_total": a.get("tokens_total"),
                    "hallucination": self._hallucination(a),
                    "input": _truncate(a.get("input")),
                    "output": _truncate(a.get("output")),
                    "error": a.get("error"),
                }
            )
        return result

    def worst_hallucinations(
        self, limit: int = 10, max_score: float = 0.5
    ) -> list[dict]:
        """Spans whose claim-level hallucination score is below ``max_score`` (lower = worse), worst first."""
        out = []
        for s in self._rows():
            score = self._hallucination(s["attributes"])
            if score is not None and score < max_score:
                out.append(
                    {
                        "trace_id": s["trace_id"],
                        "span_id": s.get("span_id"),
                        "name": s.get("name"),
                        "hallucination": score,
                        "output": _truncate(s["attributes"].get("output")),
                    }
                )
        out.sort(key=lambda x: x["hallucination"])
        return out[:limit]

    def token_usage_by_model(self) -> dict[str, int]:
        """Total tokens used, grouped by model name."""
        usage: dict[str, int] = defaultdict(int)
        for s in self._rows():
            a = s["attributes"]
            model = a.get("model")
            if model:
                usage[model] += a.get("tokens_total") or 0
        return dict(sorted(usage.items(), key=lambda kv: kv[1], reverse=True))

    def error_spans(self, limit: int = 20) -> list[dict]:
        """Spans that errored, most recent first."""
        out = [s for s in self._rows() if s.get("status") == "error"]
        out.sort(key=lambda s: s.get("start_time") or 0, reverse=True)
        return [
            {
                "trace_id": s["trace_id"],
                "span_id": s.get("span_id"),
                "name": s.get("name"),
                "error": s["attributes"].get("error"),
                "model": s["attributes"].get("model"),
            }
            for s in out[:limit]
        ]

    def search_spans(
        self,
        name_contains: str | None = None,
        model: str | None = None,
        limit: int = 25,
    ) -> list[dict]:
        """Find spans by substring of the span name and/or exact model."""
        out = []
        for s in self._rows():
            if (
                name_contains
                and name_contains.lower() not in (s.get("name") or "").lower()
            ):
                continue
            if model and s["attributes"].get("model") != model:
                continue
            out.append(
                {
                    "trace_id": s["trace_id"],
                    "span_id": s.get("span_id"),
                    "name": s.get("name"),
                    "status": s.get("status"),
                    "duration_ms": s.get("duration_ms"),
                    "model": s["attributes"].get("model"),
                    "tokens_total": s["attributes"].get("tokens_total"),
                }
            )
            if len(out) >= limit:
                break
        return out


def build_server(store: TraceStore):
    """Wire the TraceStore methods up as MCP tools. Imports ``mcp`` lazily."""
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("peekr")

    @server.tool()
    def recent_traces(limit: int = 20) -> list[dict]:
        """List the most recent Peekr traces with a summary (span count, tokens, models, errors)."""
        return store.recent_traces(limit)

    @server.tool()
    def get_trace(trace_id: str) -> list[dict]:
        """Get every span in a single trace (the full waterfall) by trace_id."""
        return store.get_trace(trace_id)

    @server.tool()
    def worst_hallucinations(limit: int = 10, max_score: float = 0.5) -> list[dict]:
        """List spans with the lowest claim-level hallucination scores (most likely hallucinated), worst first."""
        return store.worst_hallucinations(limit, max_score)

    @server.tool()
    def token_usage_by_model() -> dict:
        """Total token usage grouped by model name."""
        return store.token_usage_by_model()

    @server.tool()
    def error_spans(limit: int = 20) -> list[dict]:
        """List spans that errored, most recent first."""
        return store.error_spans(limit)

    @server.tool()
    def search_spans(
        name_contains: str = "", model: str = "", limit: int = 25
    ) -> list[dict]:
        """Search spans by a substring of the span name and/or an exact model name."""
        return store.search_spans(name_contains or None, model or None, limit)

    return server


def run(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="peekr-mcp",
        description="Peekr MCP server — expose your local Peekr traces to AI assistants over MCP (read-only).",
    )
    ap.add_argument("--db", help="Path to a SQLite trace store (traces.db).")
    ap.add_argument("--jsonl", help="Path to a JSONL trace store (traces.jsonl).")
    args = ap.parse_args(argv)
    store = TraceStore(db_path=args.db, jsonl_path=args.jsonl)
    build_server(store).run()  # stdio transport


if __name__ == "__main__":
    run()
