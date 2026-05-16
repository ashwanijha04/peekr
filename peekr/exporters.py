from __future__ import annotations
import json
import sqlite3
import threading
from .span import Span


class JSONLExporter:
    # Storage exporters are subject to sampling — see context.should_persist.
    # Mutating exporters (EvalExporter, AlertExporter) do NOT set this flag,
    # so they always run on every span (evaluators score the full trace;
    # alerts compute on the true error rate).
    _is_storage = True

    def __init__(self, path: str = "traces.jsonl"):
        self.path = path

    def export(self, span: Span) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(span.to_dict()) + "\n")


class ConsoleExporter:
    _is_storage = True

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
    _is_storage = True

    def __init__(self, path: str = "traces.db"):
        self.path = path
        self._lock = threading.Lock()
        self._init_db()

    # Bump when the schema changes. Migrations below are additive and idempotent.
    SCHEMA_VERSION = 1

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS spans (
                    trace_id        TEXT NOT NULL,
                    span_id         TEXT NOT NULL PRIMARY KEY,
                    parent_id       TEXT,
                    name            TEXT NOT NULL,
                    start_time      REAL NOT NULL,
                    end_time        REAL,
                    duration_ms     REAL,
                    status          TEXT NOT NULL DEFAULT 'ok',
                    attributes      TEXT,
                    tenant_id       TEXT,
                    retention_class TEXT
                )
            """)
            self._migrate(conn)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trace_id       ON spans(trace_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_name           ON spans(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_start_time     ON spans(start_time)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tenant_id      ON spans(tenant_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_retention      ON spans(retention_class)")
            conn.execute(f"PRAGMA user_version = {self.SCHEMA_VERSION}")

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Bring an existing DB up to SCHEMA_VERSION. Additive only."""
        current = conn.execute("PRAGMA user_version").fetchone()[0]
        if current >= self.SCHEMA_VERSION:
            return
        existing_cols = {r[1] for r in conn.execute("PRAGMA table_info(spans)").fetchall()}
        if "tenant_id" not in existing_cols:
            conn.execute("ALTER TABLE spans ADD COLUMN tenant_id TEXT")
        if "retention_class" not in existing_cols:
            conn.execute("ALTER TABLE spans ADD COLUMN retention_class TEXT")

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
                         start_time, end_time, duration_ms, status, attributes,
                         tenant_id, retention_class)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    span.tenant_id,
                    span.retention_class,
                ))

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]


class HTTPExporter:
    """Ship spans to a Peekr Cloud (or self-hosted) ingestion endpoint over HTTPS.

    Reserved public surface — implementation lands with Peekr Cloud GA.
    The constructor signature is stable as of v0.3 so you can wire it in
    today and the call site won't change when the body is filled in.

    Usage (works once Peekr Cloud is live):

        peekr.instrument(
            tenant_id="acme",
            exporter=peekr.HTTPExporter(
                endpoint="https://ingest.peekr.cloud",
                api_key="pk_live_…",
            ),
        )

    Until then this class raises NotImplementedError on .export() so a
    misconfigured pipeline fails loudly rather than silently dropping spans.
    Get on the waitlist: https://github.com/ashwanijha04/peekr/discussions
    """
    _is_storage = True

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        *,
        batch_size: int = 100,
        flush_interval_seconds: float = 5.0,
        timeout_seconds: float = 10.0,
    ) -> None:
        if not endpoint:
            raise ValueError("HTTPExporter: endpoint is required")
        if not api_key:
            raise ValueError("HTTPExporter: api_key is required")
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.batch_size = batch_size
        self.flush_interval_seconds = flush_interval_seconds
        self.timeout_seconds = timeout_seconds

    def export(self, span: Span) -> None:
        raise NotImplementedError(
            "HTTPExporter ships with Peekr Cloud (Phase 1). "
            "Use JSONLExporter or SQLiteExporter today; "
            "see https://github.com/ashwanijha04/peekr/discussions for the waitlist."
        )


_exporters: list = []


def add_exporter(exporter) -> None:
    _exporters.append(exporter)


def export_span(span: Span) -> None:
    # Mutators (EvalExporter, AlertExporter) always run — they need every
    # span to compute correct eval scores and alert rates. Storage exporters
    # respect sampling via the _is_storage marker.
    keep = None  # resolved lazily; avoids importing context for no-sampling case
    for exporter in _exporters:
        if getattr(exporter, "_is_storage", False):
            if keep is None:
                from .context import should_persist
                keep = should_persist(span)
            if not keep:
                continue
        exporter.export(span)
