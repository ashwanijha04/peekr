from __future__ import annotations
import json
import sqlite3
import threading
from .span import Span


class JSONLExporter:
    def __init__(self, path: str = "traces.jsonl"):
        self.path = path

    def export(self, span: Span) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(span.to_dict()) + "\n")


class ConsoleExporter:
    def export(self, span: Span) -> None:
        duration = f"{span.duration_ms:.1f}ms" if span.duration_ms else "?"
        indent = "  " if span.parent_id else ""
        attrs = ""
        if "model" in span.attributes:
            attrs += f" model={span.attributes['model']}"
        if "tokens_total" in span.attributes:
            attrs += f" tokens={span.attributes['tokens_total']}"
        print(f"{indent}[{span.name}] {duration}{attrs}")


class SQLiteExporter:
    def __init__(self, path: str = "traces.db"):
        self.path = path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS spans (
                    trace_id    TEXT NOT NULL,
                    span_id     TEXT NOT NULL PRIMARY KEY,
                    parent_id   TEXT,
                    name        TEXT NOT NULL,
                    start_time  REAL NOT NULL,
                    end_time    REAL,
                    duration_ms REAL,
                    status      TEXT NOT NULL DEFAULT 'ok',
                    attributes  TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trace_id   ON spans(trace_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_name       ON spans(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_start_time ON spans(start_time)")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def export(self, span: Span) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO spans
                        (trace_id, span_id, parent_id, name,
                         start_time, end_time, duration_ms, status, attributes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    span.trace_id,
                    span.span_id,
                    span.parent_id,
                    span.name,
                    span.start_time,
                    span.end_time,
                    span.duration_ms,
                    span.status,
                    json.dumps(span.attributes),
                ))

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]


_exporters: list = []


def add_exporter(exporter) -> None:
    _exporters.append(exporter)


def export_span(span: Span) -> None:
    for exporter in _exporters:
        exporter.export(span)
