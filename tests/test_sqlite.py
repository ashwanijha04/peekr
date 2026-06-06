import json
import threading
import pytest
from peekr.span import Span
from peekr.exporters import SQLiteExporter


def make_span(name="test", trace_id="trace1", parent_id=None):
    s = Span(name=name, trace_id=trace_id, parent_id=parent_id)
    s.finish()
    return s


@pytest.fixture
def db(tmp_path):
    return SQLiteExporter(path=str(tmp_path / "traces.db"))


def test_creates_table(db):
    rows = db.query(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='spans'"
    )
    assert rows, "spans table not created"


def test_export_and_read_back(db):
    span = make_span("openai.chat", trace_id="abc123")
    span.attributes["model"] = "gpt-4o"
    span.attributes["tokens_total"] = 42
    db.export(span)

    rows = db.query("SELECT * FROM spans WHERE span_id = ?", (span.span_id,))
    assert len(rows) == 1
    row = rows[0]
    assert row["name"] == "openai.chat"
    assert row["trace_id"] == "abc123"
    assert row["status"] == "ok"
    attrs = json.loads(row["attributes"])
    assert attrs["model"] == "gpt-4o"
    assert attrs["tokens_total"] == 42


def test_duration_stored(db):
    span = make_span()
    db.export(span)
    row = db.query("SELECT duration_ms FROM spans")[0]
    assert row["duration_ms"] is not None
    assert row["duration_ms"] >= 0


def test_parent_child_stored(db):
    parent = make_span("agent.run", trace_id="t1")
    child = make_span("tool.search", trace_id="t1", parent_id=parent.span_id)
    db.export(parent)
    db.export(child)

    rows = db.query("SELECT * FROM spans WHERE trace_id='t1' ORDER BY start_time")
    assert len(rows) == 2
    child_row = next(r for r in rows if r["name"] == "tool.search")
    assert child_row["parent_id"] == parent.span_id


def test_error_span(db):
    span = make_span("tool.boom")
    span.status = "error"
    span.attributes["error"] = "something went wrong"
    db.export(span)

    row = db.query("SELECT * FROM spans")[0]
    assert row["status"] == "error"
    assert json.loads(row["attributes"])["error"] == "something went wrong"


def test_query_by_trace_id(db):
    for i in range(3):
        db.export(make_span(f"span-{i}", trace_id="trace-A"))
    db.export(make_span("other", trace_id="trace-B"))

    rows = db.query("SELECT * FROM spans WHERE trace_id = ?", ("trace-A",))
    assert len(rows) == 3


def test_wal_mode_enabled(db):
    rows = db.query("PRAGMA journal_mode")
    assert rows[0]["journal_mode"] == "wal"


def test_concurrent_writes(db):
    errors = []

    def write(n):
        try:
            for i in range(10):
                db.export(make_span(f"span-{n}-{i}", trace_id=f"trace-{n}"))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=write, args=(n,)) for n in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent write errors: {errors}"
    rows = db.query("SELECT COUNT(*) as c FROM spans")
    assert rows[0]["c"] == 50


def test_idempotent_export(db):
    span = make_span()
    db.export(span)
    db.export(span)  # same span_id — should upsert, not duplicate
    rows = db.query("SELECT COUNT(*) as c FROM spans")
    assert rows[0]["c"] == 1
