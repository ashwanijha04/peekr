from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from typing import Optional


_lock = threading.Lock()


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_feedback_table(db_path: str) -> None:
    with _connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id         TEXT NOT NULL PRIMARY KEY,
                trace_id   TEXT NOT NULL,
                rating     TEXT NOT NULL,
                note       TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_trace_id ON feedback(trace_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_rating   ON feedback(rating)"
        )


def feedback(
    trace_id: str,
    rating: str,
    note: str = "",
    db_path: str = "traces.db",
) -> None:
    """Insert a feedback row for a given trace_id.

    Args:
        trace_id: The trace to annotate.
        rating:   "good" or "bad" (or any string label).
        note:     Optional free-text note.
        db_path:  Path to the SQLite database (default: traces.db).
    """
    _init_feedback_table(db_path)
    with _lock:
        with _connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO feedback (id, trace_id, rating, note, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (uuid.uuid4().hex, trace_id, rating, note, time.time()),
            )


def export_feedback(
    db_path: str = "traces.db",
    filter: Optional[str] = None,
    output: str = "training_data.jsonl",
    format: str = "openai-ft",
) -> None:
    """Export feedback-annotated traces to a JSONL file.

    Args:
        db_path: Path to the SQLite database.
        filter:  "good", "bad", or None for all ratings.
        output:  Path to the output JSONL file.
        format:  "openai-ft" or "raw".
    """
    _init_feedback_table(db_path)

    with _connect(db_path) as conn:
        # Check if spans table exists (it may not if SQLiteExporter was never used)
        table_check = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='spans'"
        ).fetchone()
        spans_exist = table_check is not None

        if spans_exist:
            if filter is not None:
                rows = conn.execute(
                    """
                    SELECT f.trace_id, f.rating, f.note, f.created_at,
                           s.span_id, s.parent_id, s.name, s.start_time, s.end_time,
                           s.duration_ms, s.status, s.attributes
                    FROM feedback f
                    LEFT JOIN spans s ON f.trace_id = s.trace_id
                    WHERE f.rating = ?
                    ORDER BY f.created_at, s.start_time
                    """,
                    (filter,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT f.trace_id, f.rating, f.note, f.created_at,
                           s.span_id, s.parent_id, s.name, s.start_time, s.end_time,
                           s.duration_ms, s.status, s.attributes
                    FROM feedback f
                    LEFT JOIN spans s ON f.trace_id = s.trace_id
                    ORDER BY f.created_at, s.start_time
                    """
                ).fetchall()
        else:
            # No spans table — just return feedback rows
            if filter is not None:
                rows = conn.execute(
                    """
                    SELECT f.trace_id, f.rating, f.note, f.created_at,
                           NULL AS span_id, NULL AS parent_id, NULL AS name,
                           NULL AS start_time, NULL AS end_time,
                           NULL AS duration_ms, NULL AS status, NULL AS attributes
                    FROM feedback f
                    WHERE f.rating = ?
                    ORDER BY f.created_at
                    """,
                    (filter,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT f.trace_id, f.rating, f.note, f.created_at,
                           NULL AS span_id, NULL AS parent_id, NULL AS name,
                           NULL AS start_time, NULL AS end_time,
                           NULL AS duration_ms, NULL AS status, NULL AS attributes
                    FROM feedback f
                    ORDER BY f.created_at
                    """
                ).fetchall()

    # Group spans per trace so we can build one record per trace
    traces: dict[str, dict] = {}
    for row in rows:
        d = dict(row)
        tid = d["trace_id"]
        if tid not in traces:
            traces[tid] = {
                "trace_id": tid,
                "rating": d["rating"],
                "note": d["note"],
                "spans": [],
            }
        if d.get("span_id"):
            attrs_raw = d.get("attributes")
            attrs = json.loads(attrs_raw) if attrs_raw else {}
            traces[tid]["spans"].append(
                {
                    "span_id": d["span_id"],
                    "parent_id": d["parent_id"],
                    "name": d["name"],
                    "start_time": d["start_time"],
                    "end_time": d["end_time"],
                    "duration_ms": d["duration_ms"],
                    "status": d["status"],
                    "attributes": attrs,
                }
            )

    with open(output, "w") as f:
        for tid, trace_data in traces.items():
            if format == "openai-ft":
                record = _to_openai_ft(trace_data)
            else:
                record = _to_raw(trace_data)
            f.write(json.dumps(record) + "\n")


def _to_openai_ft(trace_data: dict) -> dict:
    """Convert a trace to OpenAI fine-tuning message format."""
    spans = trace_data["spans"]

    # Extract input and output from span attributes (prefer LLM spans)
    input_text = ""
    output_text = ""

    llm_prefixes = ("openai.", "anthropic.", "bedrock.")
    llm_spans = [s for s in spans if s["name"].startswith(llm_prefixes)]
    target_spans = llm_spans if llm_spans else spans

    for span in target_spans:
        attrs = span.get("attributes", {})
        if not input_text and attrs.get("input"):
            input_text = attrs["input"]
        if not output_text and attrs.get("output"):
            output_text = attrs["output"]

    return {
        "messages": [
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_text},
        ]
    }


def _to_raw(trace_data: dict) -> dict:
    """Convert a trace to raw format including all span data and feedback rating."""
    return {
        "trace_id": trace_data["trace_id"],
        "rating": trace_data["rating"],
        "note": trace_data["note"],
        "spans": trace_data["spans"],
    }
