"""First-class tenant_id / retention_class on Span.

These fields are first-class (not in attributes) so the hosted backend can
route + index without JSON extraction. Resolution priority for each field:
peekr.session(...) > instrument(...) > env var > None.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager

import pytest

import peekr
from peekr.context import set_process_defaults, start_span, end_span
from peekr.exporters import JSONLExporter, SQLiteExporter, HTTPExporter
from peekr.span import Span


@contextmanager
def _clean_env(*keys: str):
    """Remove env keys for the duration; restore after."""
    saved = {k: os.environ.pop(k, None) for k in keys}
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


@contextmanager
def _clean_defaults():
    """Reset process defaults around a test."""
    from peekr import context as ctx
    saved = (ctx._default_tenant_id, ctx._default_retention_class)
    ctx._default_tenant_id = None
    ctx._default_retention_class = None
    try:
        yield
    finally:
        ctx._default_tenant_id, ctx._default_retention_class = saved


# ---------------------------------------------------------------------------
# 1. Schema
# ---------------------------------------------------------------------------

class TestSchema:
    def test_span_has_first_class_fields(self):
        s = Span(name="x", trace_id="t")
        assert s.tenant_id is None
        assert s.retention_class is None

    def test_to_dict_includes_first_class_fields(self):
        s = Span(name="x", trace_id="t", tenant_id="acme", retention_class="long")
        s.finish()
        d = s.to_dict()
        assert d["tenant_id"] == "acme"
        assert d["retention_class"] == "long"
        # Must NOT be silently moved into attributes
        assert "tenant_id" not in d.get("attributes", {})


# ---------------------------------------------------------------------------
# 2. Resolution order: session > instrument > env > None
# ---------------------------------------------------------------------------

class TestResolutionOrder:
    def test_no_signal_yields_none(self):
        with _clean_defaults(), _clean_env("PEEKR_TENANT_ID", "PEEKR_RETENTION_CLASS"):
            span, tok = start_span("op")
            try:
                assert span.tenant_id is None
                assert span.retention_class is None
            finally:
                end_span(span, tok)

    def test_env_var_picked_up(self):
        with _clean_defaults(), _clean_env("PEEKR_TENANT_ID", "PEEKR_RETENTION_CLASS"):
            os.environ["PEEKR_TENANT_ID"] = "from-env"
            os.environ["PEEKR_RETENTION_CLASS"] = "short"
            span, tok = start_span("op")
            try:
                assert span.tenant_id == "from-env"
                assert span.retention_class == "short"
            finally:
                end_span(span, tok)

    def test_instrument_default_overrides_env(self):
        with _clean_defaults(), _clean_env("PEEKR_TENANT_ID"):
            os.environ["PEEKR_TENANT_ID"] = "from-env"
            set_process_defaults(tenant_id="from-instrument")
            span, tok = start_span("op")
            try:
                assert span.tenant_id == "from-instrument"
            finally:
                end_span(span, tok)

    def test_session_overrides_everything(self):
        with _clean_defaults(), _clean_env("PEEKR_TENANT_ID", "PEEKR_RETENTION_CLASS"):
            os.environ["PEEKR_TENANT_ID"] = "from-env"
            set_process_defaults(tenant_id="from-instrument", retention_class="default")
            with peekr.session(tenant_id="from-session", retention_class="long"):
                span, tok = start_span("op")
                try:
                    assert span.tenant_id == "from-session"
                    assert span.retention_class == "long"
                finally:
                    end_span(span, tok)

    def test_session_tenant_distinct_from_user_id(self):
        """user_id is the end-user; tenant_id is the customer org. Both must coexist."""
        with _clean_defaults(), _clean_env("PEEKR_TENANT_ID"):
            with peekr.session(tenant_id="acme", user_id="alice"):
                span, tok = start_span("op")
                try:
                    assert span.tenant_id == "acme"           # first-class
                    assert span.attributes["user_id"] == "alice"  # legacy attribute
                finally:
                    end_span(span, tok)


# ---------------------------------------------------------------------------
# 3. Persistence — JSONL + SQLite
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_jsonl_emits_top_level(self, tmp_path):
        path = tmp_path / "traces.jsonl"
        exp = JSONLExporter(str(path))
        s = Span(name="op", trace_id="t", tenant_id="acme", retention_class="long")
        s.finish()
        exp.export(s)
        line = json.loads(path.read_text().strip())
        assert line["tenant_id"] == "acme"
        assert line["retention_class"] == "long"

    def test_sqlite_indexes_new_columns(self, tmp_path):
        path = tmp_path / "traces.db"
        SQLiteExporter(str(path))
        with sqlite3.connect(path) as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(spans)").fetchall()}
            indexes = {r[1] for r in conn.execute("PRAGMA index_list(spans)").fetchall()}
        assert "tenant_id" in cols
        assert "retention_class" in cols
        assert "idx_tenant_id" in indexes
        assert "idx_retention" in indexes

    def test_sqlite_stores_and_queries_by_tenant(self, tmp_path):
        path = tmp_path / "traces.db"
        exp = SQLiteExporter(str(path))
        for tenant in ("acme", "globex", "acme"):
            s = Span(name="op", trace_id=f"t-{tenant}", tenant_id=tenant)
            s.finish()
            exp.export(s)
        rows = exp.query("SELECT tenant_id FROM spans WHERE tenant_id = ?", ("acme",))
        assert len(rows) == 2
        assert all(r["tenant_id"] == "acme" for r in rows)

    def test_http_exporter_signature_is_stable(self):
        """Reserved public surface for Peekr Cloud — must accept the
        documented kwargs and reject missing required ones, even though
        .export() raises NotImplementedError until cloud GA."""
        exp = HTTPExporter(
            endpoint="https://ingest.peekr.cloud",
            api_key="pk_live_test",
            batch_size=50,
            flush_interval_seconds=2.5,
            timeout_seconds=15.0,
        )
        assert exp.endpoint == "https://ingest.peekr.cloud"  # trailing slash stripped
        assert exp.api_key == "pk_live_test"
        assert exp.batch_size == 50

        # Defaults
        exp2 = HTTPExporter(endpoint="https://x/", api_key="k")
        assert exp2.endpoint == "https://x"
        assert exp2.batch_size == 100

        # Required
        with pytest.raises(ValueError):
            HTTPExporter(endpoint="", api_key="k")
        with pytest.raises(ValueError):
            HTTPExporter(endpoint="https://x", api_key="")

        # Until Cloud GA, .export() must fail loudly so misconfigured
        # pipelines don't silently drop spans.
        s = Span(name="x", trace_id="t")
        s.finish()
        with pytest.raises(NotImplementedError):
            exp.export(s)

    def test_sqlite_migrates_existing_db_additively(self, tmp_path):
        """Pre-v1 DBs without the new columns must migrate cleanly without losing data."""
        path = tmp_path / "traces.db"
        # Simulate a pre-migration DB
        with sqlite3.connect(path) as conn:
            conn.execute("""
                CREATE TABLE spans (
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
            conn.execute(
                "INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("t1", "s1", None, "old.span", 1.0, 1.5, 500.0, "ok", "{}"),
            )

        # Opening through the exporter should migrate, not blow up
        exp = SQLiteExporter(str(path))
        with sqlite3.connect(path) as conn:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(spans)").fetchall()}
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            row = conn.execute("SELECT span_id, tenant_id FROM spans WHERE span_id = 's1'").fetchone()

        assert "tenant_id" in cols and "retention_class" in cols
        assert version == SQLiteExporter.SCHEMA_VERSION
        assert row[0] == "s1"        # legacy row preserved
        assert row[1] is None        # back-filled as NULL

        # And new writes work
        s = Span(name="new.span", trace_id="t2", tenant_id="acme", retention_class="long")
        s.finish()
        exp.export(s)
        with sqlite3.connect(path) as conn:
            r = conn.execute(
                "SELECT tenant_id, retention_class FROM spans WHERE span_id = ?", (s.span_id,)
            ).fetchone()
        assert r == ("acme", "long")
