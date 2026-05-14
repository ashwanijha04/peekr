"""Read-only storage adapter for the peekr web dashboard.

Reads from the same on-disk formats that :class:`peekr.exporters.SQLiteExporter`
and :class:`peekr.exporters.JSONLExporter` write. No new schema; no writes.

SQLite is preferred when available because it supports filtering at the query
layer. JSONL is the fallback: traces are loaded fully into memory and filtered
in Python.
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


# Same per-million-token rates as `peekr cost`. Used to estimate cost from
# the tokens_input / tokens_output attributes captured by the SDK patches.
COST_PER_M_INPUT = 0.80
COST_PER_M_OUTPUT = 4.00


def estimate_cost(tokens_input: int, tokens_output: int) -> float:
    return (tokens_input / 1_000_000) * COST_PER_M_INPUT + (
        tokens_output / 1_000_000
    ) * COST_PER_M_OUTPUT


@dataclass
class TraceSummary:
    trace_id: str
    start_time: float
    span_count: int
    root_duration_ms: float
    tokens_input: int
    tokens_output: int
    tokens_total: int
    cost: float
    error_count: int
    user_id: Optional[str]
    session_id: Optional[str]
    has_evals: bool
    has_guardrails: bool
    root_name: Optional[str] = None


@dataclass
class Filters:
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    since: Optional[float] = None
    until: Optional[float] = None
    min_cost: Optional[float] = None
    has_evals: bool = False
    has_guardrails: bool = False
    has_errors: bool = False
    q: Optional[str] = None  # substring match against span names

    def is_empty(self) -> bool:
        return (
            self.user_id is None
            and self.session_id is None
            and self.since is None
            and self.until is None
            and self.min_cost is None
            and not self.has_evals
            and not self.has_guardrails
            and not self.has_errors
            and not self.q
        )


@dataclass
class FacetValues:
    """Distinct values harvested from spans to populate the filter dropdowns."""

    users: list[str] = field(default_factory=list)
    sessions: list[str] = field(default_factory=list)


# ───────────────────────── storage abstraction ──────────────────────────────


class TraceStore:
    """Common interface returned by :func:`open_store`."""

    def list_traces(
        self,
        filters: Filters,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[TraceSummary], int]:
        raise NotImplementedError

    def get_trace(self, trace_id: str) -> list[dict]:
        raise NotImplementedError

    def facets(self) -> FacetValues:
        raise NotImplementedError

    @property
    def source(self) -> str:
        raise NotImplementedError


# ───────────────────────── SQLite store ─────────────────────────────────────


class SQLiteStore(TraceStore):
    def __init__(self, path: str) -> None:
        self.path = path

    @property
    def source(self) -> str:
        return self.path

    def _connect(self) -> sqlite3.Connection:
        # uri=True + mode=ro is best-effort read-only; falls back fine if the
        # file lives on a writable disk anyway.
        try:
            conn = sqlite3.connect(
                f"file:{self.path}?mode=ro", uri=True, check_same_thread=False
            )
        except sqlite3.OperationalError:
            conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def list_traces(
        self,
        filters: Filters,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[TraceSummary], int]:
        # Aggregate one row per trace_id. json_extract pulls user_id /
        # session_id / eval / guardrail flags out of the JSON attributes blob.
        select = """
            SELECT
                trace_id,
                MIN(start_time)                                        AS start_time,
                COUNT(*)                                               AS span_count,
                COALESCE(SUM(CASE WHEN parent_id IS NULL THEN duration_ms ELSE 0 END), 0)            AS root_duration_ms,
                COALESCE(SUM(CAST(json_extract(attributes,'$.tokens_input')  AS INTEGER)), 0) AS tokens_input,
                COALESCE(SUM(CAST(json_extract(attributes,'$.tokens_output') AS INTEGER)), 0) AS tokens_output,
                COALESCE(SUM(CAST(json_extract(attributes,'$.tokens_total')  AS INTEGER)), 0) AS tokens_total,
                SUM(CASE WHEN status='error' THEN 1 ELSE 0 END)        AS error_count,
                MAX(json_extract(attributes,'$.user_id'))              AS user_id,
                MAX(json_extract(attributes,'$.session_id'))           AS session_id,
                MAX(CASE WHEN json_extract(attributes,'$.eval_scores') IS NOT NULL THEN 1 ELSE 0 END)         AS has_evals,
                MAX(CASE WHEN json_extract(attributes,'$.guardrail_findings') IS NOT NULL THEN 1 ELSE 0 END)  AS has_guardrails
            FROM spans
            GROUP BY trace_id
        """

        having_clauses: list[str] = []
        params: list[Any] = []

        if filters.user_id:
            having_clauses.append("user_id = ?")
            params.append(filters.user_id)
        if filters.session_id:
            having_clauses.append("session_id = ?")
            params.append(filters.session_id)
        if filters.since is not None:
            having_clauses.append("start_time >= ?")
            params.append(filters.since)
        if filters.until is not None:
            having_clauses.append("start_time <= ?")
            params.append(filters.until)
        if filters.has_evals:
            having_clauses.append("has_evals = 1")
        if filters.has_guardrails:
            having_clauses.append("has_guardrails = 1")
        if filters.has_errors:
            having_clauses.append("error_count > 0")
        if filters.q:
            # Match a substring against any span name in this trace.
            having_clauses.append(
                "trace_id IN (SELECT trace_id FROM spans WHERE name LIKE ?)"
            )
            params.append(f"%{filters.q}%")

        if having_clauses:
            select += " HAVING " + " AND ".join(having_clauses)

        select += " ORDER BY start_time DESC"

        with self._connect() as conn:
            rows = conn.execute(select, params).fetchall()

        results: list[TraceSummary] = []
        for row in rows:
            cost = estimate_cost(row["tokens_input"], row["tokens_output"])
            if filters.min_cost is not None and cost < filters.min_cost:
                continue
            results.append(
                TraceSummary(
                    trace_id=row["trace_id"],
                    start_time=row["start_time"] or 0.0,
                    span_count=row["span_count"],
                    root_duration_ms=row["root_duration_ms"] or 0.0,
                    tokens_input=row["tokens_input"] or 0,
                    tokens_output=row["tokens_output"] or 0,
                    tokens_total=row["tokens_total"] or 0,
                    cost=cost,
                    error_count=row["error_count"] or 0,
                    user_id=row["user_id"],
                    session_id=row["session_id"],
                    has_evals=bool(row["has_evals"]),
                    has_guardrails=bool(row["has_guardrails"]),
                )
            )

        total = len(results)
        return results[offset : offset + limit], total

    def get_trace(self, trace_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_time",
                (trace_id,),
            ).fetchall()
        spans = []
        for r in rows:
            s = dict(r)
            s["attributes"] = json.loads(s["attributes"] or "{}")
            spans.append(s)
        return spans

    def facets(self) -> FacetValues:
        with self._connect() as conn:
            users = [
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT json_extract(attributes,'$.user_id') "
                    "FROM spans WHERE json_extract(attributes,'$.user_id') IS NOT NULL "
                    "ORDER BY 1"
                )
            ]
            sessions = [
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT json_extract(attributes,'$.session_id') "
                    "FROM spans WHERE json_extract(attributes,'$.session_id') IS NOT NULL "
                    "ORDER BY 1"
                )
            ]
        return FacetValues(users=users, sessions=sessions)


# ───────────────────────── JSONL store ──────────────────────────────────────


class JSONLStore(TraceStore):
    """Fallback for users who have not enabled SQLite storage."""

    def __init__(self, path: str) -> None:
        self.path = path

    @property
    def source(self) -> str:
        return self.path

    def _read_all(self) -> list[dict]:
        spans: list[dict] = []
        if not os.path.exists(self.path):
            return spans
        with open(self.path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    spans.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return spans

    def _grouped(self) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {}
        for span in self._read_all():
            out.setdefault(span["trace_id"], []).append(span)
        return out

    def list_traces(
        self,
        filters: Filters,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[TraceSummary], int]:
        summaries: list[TraceSummary] = []
        for trace_id, spans in self._grouped().items():
            summary = _summarize(trace_id, spans)
            if _matches(summary, spans, filters):
                summaries.append(summary)

        summaries.sort(key=lambda s: s.start_time, reverse=True)
        return summaries[offset : offset + limit], len(summaries)

    def get_trace(self, trace_id: str) -> list[dict]:
        spans = [s for s in self._read_all() if s.get("trace_id") == trace_id]
        spans.sort(key=lambda s: s.get("start_time", 0))
        return spans

    def facets(self) -> FacetValues:
        users: set[str] = set()
        sessions: set[str] = set()
        for span in self._read_all():
            attrs = span.get("attributes") or {}
            if attrs.get("user_id"):
                users.add(str(attrs["user_id"]))
            if attrs.get("session_id"):
                sessions.add(str(attrs["session_id"]))
        return FacetValues(users=sorted(users), sessions=sorted(sessions))


def _summarize(trace_id: str, spans: Iterable[dict]) -> TraceSummary:
    spans = list(spans)
    tokens_input = 0
    tokens_output = 0
    tokens_total = 0
    error_count = 0
    has_evals = False
    has_guardrails = False
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    root_duration_ms = 0.0
    start_time = min((s.get("start_time") or 0.0 for s in spans), default=0.0)

    for s in spans:
        attrs = s.get("attributes") or {}
        tokens_input += int(attrs.get("tokens_input") or 0)
        tokens_output += int(attrs.get("tokens_output") or 0)
        tokens_total += int(attrs.get("tokens_total") or 0)
        if s.get("status") == "error":
            error_count += 1
        if attrs.get("eval_scores"):
            has_evals = True
        if attrs.get("guardrail_findings"):
            has_guardrails = True
        if attrs.get("user_id") and user_id is None:
            user_id = str(attrs["user_id"])
        if attrs.get("session_id") and session_id is None:
            session_id = str(attrs["session_id"])
        if s.get("parent_id") is None:
            root_duration_ms += float(s.get("duration_ms") or 0)

    return TraceSummary(
        trace_id=trace_id,
        start_time=start_time,
        span_count=len(spans),
        root_duration_ms=root_duration_ms,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        tokens_total=tokens_total,
        cost=estimate_cost(tokens_input, tokens_output),
        error_count=error_count,
        user_id=user_id,
        session_id=session_id,
        has_evals=has_evals,
        has_guardrails=has_guardrails,
    )


def _matches(summary: TraceSummary, spans: list[dict], filters: Filters) -> bool:
    if filters.user_id and summary.user_id != filters.user_id:
        return False
    if filters.session_id and summary.session_id != filters.session_id:
        return False
    if filters.since is not None and summary.start_time < filters.since:
        return False
    if filters.until is not None and summary.start_time > filters.until:
        return False
    if filters.min_cost is not None and summary.cost < filters.min_cost:
        return False
    if filters.has_evals and not summary.has_evals:
        return False
    if filters.has_guardrails and not summary.has_guardrails:
        return False
    if filters.has_errors and summary.error_count == 0:
        return False
    if filters.q:
        needle = filters.q.lower()
        if not any(needle in (s.get("name") or "").lower() for s in spans):
            return False
    return True


# ───────────────────────── factory ──────────────────────────────────────────


def open_store(db: Optional[str] = None, jsonl: Optional[str] = None) -> TraceStore:
    """Pick the right store given the caller's preferences.

    Order: explicit ``db`` → explicit ``jsonl`` → first of ``traces.db`` /
    ``traces.jsonl`` that exists in the current working directory.
    """
    if db:
        return SQLiteStore(db)
    if jsonl:
        return JSONLStore(jsonl)
    if os.path.exists("traces.db"):
        return SQLiteStore("traces.db")
    return JSONLStore("traces.jsonl")


# ───────────────────────── tree helpers (used by templates) ─────────────────


def build_tree(spans: list[dict]) -> list[dict]:
    """Return root spans, each with a ``children`` list (recursive).

    Spans that reference a missing parent get promoted to roots so they are
    still rendered.
    """
    by_id: dict[str, dict] = {}
    for span in spans:
        s = dict(span)
        s["children"] = []
        # Normalise attributes to a dict in case the caller passed a row with
        # the original JSON string still in it.
        attrs = s.get("attributes")
        if isinstance(attrs, str):
            try:
                s["attributes"] = json.loads(attrs)
            except json.JSONDecodeError:
                s["attributes"] = {}
        elif attrs is None:
            s["attributes"] = {}
        by_id[s["span_id"]] = s

    roots: list[dict] = []
    for span in by_id.values():
        parent_id = span.get("parent_id")
        if parent_id and parent_id in by_id:
            by_id[parent_id]["children"].append(span)
        else:
            roots.append(span)
    return roots


def trace_overview(spans: list[dict]) -> dict:
    """Aggregate stats for the trace-detail header (cost, tokens, errors)."""
    tokens_input = 0
    tokens_output = 0
    error_count = 0
    span_count = len(spans)
    root_duration_ms = 0.0
    eval_scores: dict[str, list[float]] = {}
    guardrail_findings: list[dict] = []
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    for s in spans:
        attrs = s.get("attributes") or {}
        tokens_input += int(attrs.get("tokens_input") or 0)
        tokens_output += int(attrs.get("tokens_output") or 0)
        if s.get("status") == "error":
            error_count += 1
        if s.get("parent_id") is None:
            root_duration_ms += float(s.get("duration_ms") or 0)
        if user_id is None and attrs.get("user_id"):
            user_id = str(attrs["user_id"])
        if session_id is None and attrs.get("session_id"):
            session_id = str(attrs["session_id"])
        for name, score in (attrs.get("eval_scores") or {}).items():
            eval_scores.setdefault(name, []).append(float(score))
        findings = attrs.get("guardrail_findings")
        if isinstance(findings, list):
            for f in findings:
                guardrail_findings.append({"span": s.get("name"), "finding": f})

    return {
        "span_count": span_count,
        "root_duration_ms": root_duration_ms,
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
        "tokens_total": tokens_input + tokens_output,
        "cost": estimate_cost(tokens_input, tokens_output),
        "error_count": error_count,
        "eval_scores": {k: sum(v) / len(v) for k, v in eval_scores.items()},
        "guardrail_findings": guardrail_findings,
        "user_id": user_id,
        "session_id": session_id,
    }
