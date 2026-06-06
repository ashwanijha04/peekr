"""Tests for the Peekr MCP server's read-only query layer (TraceStore).

These exercise the SQLite + JSONL readers and every tool method without
requiring the optional ``mcp`` dependency to be installed.
"""

import json
import sqlite3

from peekr.mcp_server import TraceStore

_DDL = """
CREATE TABLE spans (
    trace_id TEXT NOT NULL, span_id TEXT PRIMARY KEY, parent_id TEXT, name TEXT,
    start_time REAL, end_time REAL, duration_ms REAL, status TEXT,
    attributes TEXT, tenant_id TEXT, retention_class TEXT
)
"""

_ROWS = [
    # trace t1: a 2-span agent run; the LLM span hallucinated (0.30)
    (
        "t1",
        "s1",
        None,
        "agent.run",
        1.0,
        2.0,
        1000.0,
        "ok",
        json.dumps(
            {
                "model": "gpt-4o",
                "tokens_total": 100,
                "eval_scores": {"Hallucination": 0.30},
                "output": "Coverage includes unlimited visits.",
            }
        ),
        None,
        None,
    ),
    (
        "t1",
        "s2",
        "s1",
        "llm.call",
        1.1,
        1.5,
        400.0,
        "ok",
        json.dumps({"model": "gpt-4o", "tokens_total": 80}),
        None,
        None,
    ),
    # trace t2: a single errored span on a different model
    (
        "t2",
        "s3",
        None,
        "agent.run",
        3.0,
        3.2,
        200.0,
        "error",
        json.dumps({"model": "claude-opus-4-8", "tokens_total": 50, "error": "boom"}),
        None,
        None,
    ),
]


def _make_db(path: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute(_DDL)
    conn.executemany("INSERT INTO spans VALUES (?,?,?,?,?,?,?,?,?,?,?)", _ROWS)
    conn.commit()
    conn.close()


def test_sqlite_reader_and_tools(tmp_path):
    db = str(tmp_path / "traces.db")
    _make_db(db)
    store = TraceStore(db_path=db)

    rows = store._rows()
    assert len(rows) == 3
    assert isinstance(rows[0]["attributes"], dict)  # attributes JSON parsed

    recent = store.recent_traces(10)
    assert {t["trace_id"] for t in recent} == {"t1", "t2"}
    t1 = next(t for t in recent if t["trace_id"] == "t1")
    assert t1["spans"] == 2
    assert t1["tokens_total"] == 180
    assert t1["models"] == ["gpt-4o"]
    assert recent[0]["trace_id"] == "t2"  # newest first (start_time 3.0 > 1.0)

    trace = store.get_trace("t1")
    assert len(trace) == 2
    assert trace[0]["name"] == "agent.run"
    assert trace[0]["hallucination"] == 0.30

    worst = store.worst_hallucinations(limit=5, max_score=0.5)
    assert len(worst) == 1 and worst[0]["trace_id"] == "t1"

    usage = store.token_usage_by_model()
    assert usage["gpt-4o"] == 180 and usage["claude-opus-4-8"] == 50

    errors = store.error_spans(10)
    assert len(errors) == 1 and errors[0]["trace_id"] == "t2"

    found = store.search_spans(name_contains="llm")
    assert len(found) == 1 and found[0]["span_id"] == "s2"


def test_jsonl_reader(tmp_path):
    path = tmp_path / "traces.jsonl"
    with open(path, "w") as f:
        for r in _ROWS:
            f.write(
                json.dumps(
                    {
                        "trace_id": r[0],
                        "span_id": r[1],
                        "parent_id": r[2],
                        "name": r[3],
                        "start_time": r[4],
                        "end_time": r[5],
                        "duration_ms": r[6],
                        "status": r[7],
                        "attributes": json.loads(r[8]),
                    }
                )
                + "\n"
            )
    store = TraceStore(jsonl_path=str(path))
    assert len(store._rows()) == 3
    assert store.token_usage_by_model()["gpt-4o"] == 180


def test_missing_store_is_empty(tmp_path):
    store = TraceStore(db_path=str(tmp_path / "nope.db"))
    assert store._rows() == []
    assert store.recent_traces() == []
