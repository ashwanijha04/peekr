from __future__ import annotations

import json
import os
import sqlite3
import tempfile

import pytest

from peekr.feedback import feedback, export_feedback


@pytest.fixture()
def db(tmp_path):
    """Return a path to a temporary SQLite DB."""
    return str(tmp_path / "test_traces.db")


@pytest.fixture()
def db_with_spans(db):
    """Create a DB with both spans and feedback tables pre-populated."""
    conn = sqlite3.connect(db)
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
        "INSERT INTO spans VALUES (?,?,?,?,?,?,?,?,?)",
        (
            "trace-good-1",
            "span-1",
            None,
            "openai.chat.completions",
            1.0,
            2.0,
            1000.0,
            "ok",
            json.dumps({"input": "What is 2+2?", "output": "4"}),
        ),
    )
    conn.execute(
        "INSERT INTO spans VALUES (?,?,?,?,?,?,?,?,?)",
        (
            "trace-bad-1",
            "span-2",
            None,
            "anthropic.messages",
            3.0,
            4.0,
            1000.0,
            "error",
            json.dumps({"input": "Tell me about X", "output": "I don't know"}),
        ),
    )
    conn.commit()
    conn.close()
    return db


# ---------------------------------------------------------------------------
# feedback() tests
# ---------------------------------------------------------------------------

class TestFeedbackInsert:
    def test_creates_feedback_table(self, db):
        feedback(trace_id="abc", rating="good", db_path=db)
        conn = sqlite3.connect(db)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        conn.close()
        assert "feedback" in tables

    def test_inserts_row(self, db):
        feedback(trace_id="abc", rating="good", note="great", db_path=db)
        conn = sqlite3.connect(db)
        rows = conn.execute("SELECT * FROM feedback WHERE trace_id='abc'").fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0][2] == "good"   # rating column
        assert rows[0][3] == "great"  # note column

    def test_multiple_feedbacks(self, db):
        feedback(trace_id="t1", rating="good", db_path=db)
        feedback(trace_id="t2", rating="bad", note="wrong", db_path=db)
        feedback(trace_id="t1", rating="bad", note="changed mind", db_path=db)
        conn = sqlite3.connect(db)
        count = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        conn.close()
        assert count == 3

    def test_default_note_is_empty(self, db):
        feedback(trace_id="t1", rating="good", db_path=db)
        conn = sqlite3.connect(db)
        row = conn.execute("SELECT note FROM feedback").fetchone()
        conn.close()
        assert row[0] == ""

    def test_unique_ids(self, db):
        feedback(trace_id="t1", rating="good", db_path=db)
        feedback(trace_id="t1", rating="good", db_path=db)
        conn = sqlite3.connect(db)
        ids = [r[0] for r in conn.execute("SELECT id FROM feedback").fetchall()]
        conn.close()
        assert ids[0] != ids[1]


# ---------------------------------------------------------------------------
# export_feedback() tests
# ---------------------------------------------------------------------------

class TestExportFeedback:
    def test_export_all_raw(self, db_with_spans, tmp_path):
        feedback(trace_id="trace-good-1", rating="good", db_path=db_with_spans)
        feedback(trace_id="trace-bad-1", rating="bad", db_path=db_with_spans)
        out = str(tmp_path / "out.jsonl")
        export_feedback(db_path=db_with_spans, filter=None, output=out, format="raw")
        with open(out) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 2
        ratings = {l["rating"] for l in lines}
        assert ratings == {"good", "bad"}

    def test_export_filter_good(self, db_with_spans, tmp_path):
        feedback(trace_id="trace-good-1", rating="good", db_path=db_with_spans)
        feedback(trace_id="trace-bad-1", rating="bad", db_path=db_with_spans)
        out = str(tmp_path / "out.jsonl")
        export_feedback(db_path=db_with_spans, filter="good", output=out, format="raw")
        with open(out) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 1
        assert lines[0]["rating"] == "good"

    def test_export_filter_bad(self, db_with_spans, tmp_path):
        feedback(trace_id="trace-good-1", rating="good", db_path=db_with_spans)
        feedback(trace_id="trace-bad-1", rating="bad", db_path=db_with_spans)
        out = str(tmp_path / "out.jsonl")
        export_feedback(db_path=db_with_spans, filter="bad", output=out, format="raw")
        with open(out) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 1
        assert lines[0]["rating"] == "bad"

    def test_export_openai_ft_format(self, db_with_spans, tmp_path):
        feedback(trace_id="trace-good-1", rating="good", db_path=db_with_spans)
        out = str(tmp_path / "out.jsonl")
        export_feedback(db_path=db_with_spans, filter="good", output=out, format="openai-ft")
        with open(out) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 1
        record = lines[0]
        assert "messages" in record
        assert len(record["messages"]) == 2
        assert record["messages"][0]["role"] == "user"
        assert record["messages"][1]["role"] == "assistant"

    def test_openai_ft_extracts_span_attributes(self, db_with_spans, tmp_path):
        feedback(trace_id="trace-good-1", rating="good", db_path=db_with_spans)
        out = str(tmp_path / "out.jsonl")
        export_feedback(db_path=db_with_spans, filter="good", output=out, format="openai-ft")
        with open(out) as f:
            record = json.loads(f.readline())
        # The span for trace-good-1 has input="What is 2+2?" and output="4"
        assert record["messages"][0]["content"] == "What is 2+2?"
        assert record["messages"][1]["content"] == "4"

    def test_raw_format_includes_spans(self, db_with_spans, tmp_path):
        feedback(trace_id="trace-good-1", rating="good", db_path=db_with_spans)
        out = str(tmp_path / "out.jsonl")
        export_feedback(db_path=db_with_spans, filter="good", output=out, format="raw")
        with open(out) as f:
            record = json.loads(f.readline())
        assert "spans" in record
        assert len(record["spans"]) == 1
        assert record["spans"][0]["name"] == "openai.chat.completions"

    def test_no_feedback_yields_empty_file(self, db_with_spans, tmp_path):
        # No feedback inserted at all
        out = str(tmp_path / "out.jsonl")
        export_feedback(db_path=db_with_spans, filter=None, output=out, format="raw")
        with open(out) as f:
            content = f.read().strip()
        assert content == ""

    def test_export_without_spans_table(self, db, tmp_path):
        """Feedback-only DB (no spans table) should not crash."""
        feedback(trace_id="t1", rating="good", db_path=db)
        out = str(tmp_path / "out.jsonl")
        export_feedback(db_path=db, filter=None, output=out, format="raw")
        with open(out) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 1
        assert lines[0]["trace_id"] == "t1"
