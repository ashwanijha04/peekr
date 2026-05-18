from __future__ import annotations
import atexit
import json
import sqlite3
import sys
import threading
import time
import urllib.error
import urllib.request
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

    Buffers spans in memory and POSTs them in batches on a background daemon
    thread. A flush happens whenever the buffer reaches `batch_size`, or every
    `flush_interval_seconds`, or at interpreter exit (registered via atexit).

    Usage:

        peekr.instrument(
            tenant_id="acme",
            exporter=peekr.HTTPExporter(
                endpoint="https://ingest.peekr.cloud",
                api_key="pk_live_…",
            ),
        )

    Failures (timeouts, 429, 5xx) are retried once, then logged and dropped.
    Spans are upserted server-side on (project_id, span_id), so retries are
    idempotent and a span re-sent with end_time set will overwrite the open row.
    """
    _is_storage = True

    _RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

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

        self._queue: list[dict] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._flush_thread: threading.Thread | None = None
        self._started = False

    def export(self, span: Span) -> None:
        if not self._started:
            self._start()
        batch: list[dict] | None = None
        with self._lock:
            self._queue.append(span.to_dict())
            if len(self._queue) >= self.batch_size:
                batch, self._queue = self._queue, []
        if batch is not None:
            self._post(batch)

    def shutdown(self) -> None:
        """Flush any buffered spans and stop the background thread.
        Called automatically at interpreter exit; safe to call manually."""
        self._stop_event.set()
        self._flush_pending()
        if self._flush_thread is not None and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=self.timeout_seconds + 1.0)

    def _start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        self._flush_thread = threading.Thread(
            target=self._flusher_loop,
            name="peekr-http-exporter",
            daemon=True,
        )
        self._flush_thread.start()
        atexit.register(self.shutdown)

    def _flusher_loop(self) -> None:
        while not self._stop_event.wait(self.flush_interval_seconds):
            self._flush_pending()

    def _flush_pending(self) -> None:
        with self._lock:
            if not self._queue:
                return
            batch, self._queue = self._queue, []
        self._post(batch)

    def _post(self, batch: list[dict], attempt: int = 0) -> None:
        body = json.dumps({"spans": batch}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.endpoint}/v1/spans",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "peekr-python",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                resp.read()
        except urllib.error.HTTPError as e:
            if e.code in self._RETRY_STATUSES and attempt < 1:
                time.sleep(1.0)
                self._post(batch, attempt=attempt + 1)
                return
            self._log_failure(e.code, e.reason, len(batch))
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < 1:
                time.sleep(1.0)
                self._post(batch, attempt=attempt + 1)
                return
            self._log_failure(None, str(e), len(batch))

    @staticmethod
    def _log_failure(code: int | None, reason: str, n: int) -> None:
        code_str = code if code is not None else "network"
        sys.stderr.write(
            f"[peekr] HTTPExporter: dropped {n} spans ({code_str}: {reason})\n"
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
