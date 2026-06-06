from __future__ import annotations
import json
import os
import sqlite3
import sys
import tempfile
import types
import uuid
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub out SDK modules so replay never constructs real clients (which raise
# on missing API keys at construction time in current SDK versions).
#
# sys.modules.setdefault is NOT enough here: in a full-suite run the real
# openai/anthropic modules are already imported by earlier test files, so the
# stubs silently lose and replay hits real clients. The autouse fixture below
# force-installs the stubs per test via monkeypatch, which also restores the
# real modules afterwards so other test files are unaffected.
# ---------------------------------------------------------------------------

def _make_openai_stub():
    mod = types.ModuleType("openai")
    chat = types.ModuleType("openai.chat")
    completions = types.ModuleType("openai.chat.completions")
    completions.create = MagicMock()
    chat.completions = completions
    mod.chat = chat
    return mod


def _make_anthropic_stub():
    mod = types.ModuleType("anthropic")
    mod.Anthropic = MagicMock()
    return mod


def _make_boto3_stub():
    mod = types.ModuleType("boto3")
    mod.client = MagicMock()
    return mod


_openai_stub = _make_openai_stub()
_anthropic_stub = _make_anthropic_stub()
_boto3_stub = _make_boto3_stub()


@pytest.fixture(autouse=True)
def _stub_llm_sdks(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", _openai_stub)
    monkeypatch.setitem(sys.modules, "anthropic", _anthropic_stub)
    monkeypatch.setitem(sys.modules, "boto3", _boto3_stub)


from peekr.exporters import _exporters
from peekr.replay import load_trace, replay_trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_span_dict(
    name: str,
    trace_id: str,
    span_id: str | None = None,
    attributes: dict | None = None,
) -> dict:
    return {
        "trace_id": trace_id,
        "span_id": span_id or uuid.uuid4().hex,
        "parent_id": None,
        "name": name,
        "start_time": 1_000_000.0,
        "end_time": 1_000_001.0,
        "duration_ms": 1000.0,
        "status": "ok",
        "attributes": attributes or {},
    }


def _write_sqlite(path: str, spans: list[dict]) -> None:
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS spans (
            trace_id TEXT, span_id TEXT PRIMARY KEY, parent_id TEXT,
            name TEXT, start_time REAL, end_time REAL,
            duration_ms REAL, status TEXT, attributes TEXT
        )
    """)
    for s in spans:
        conn.execute(
            "INSERT INTO spans VALUES (?,?,?,?,?,?,?,?,?)",
            (
                s["trace_id"], s["span_id"], s.get("parent_id"),
                s["name"], s["start_time"], s["end_time"],
                s["duration_ms"], s["status"],
                json.dumps(s.get("attributes", {})),
            ),
        )
    conn.commit()
    conn.close()


def _write_jsonl(path: str, spans: list[dict]) -> None:
    with open(path, "w") as fh:
        for s in spans:
            d = dict(s)
            # attributes stored as dict in JSONL (peekr writes them as JSON string
            # inside the span dict, but to_dict() returns the dict directly)
            fh.write(json.dumps(d) + "\n")


# ---------------------------------------------------------------------------
# load_trace tests
# ---------------------------------------------------------------------------

class TestLoadTrace:
    def test_load_from_sqlite(self, tmp_path):
        trace_id = uuid.uuid4().hex
        spans = [
            _make_span_dict("openai.chat.completions", trace_id),
            _make_span_dict("agent.run", trace_id),
        ]
        db = str(tmp_path / "traces.db")
        _write_sqlite(db, spans)

        result = load_trace(trace_id, db_path=db)
        assert len(result) == 2
        names = {s["name"] for s in result}
        assert "openai.chat.completions" in names

    def test_load_from_jsonl(self, tmp_path):
        trace_id = uuid.uuid4().hex
        spans = [
            _make_span_dict("anthropic.messages", trace_id),
        ]
        jsonl = str(tmp_path / "traces.jsonl")
        _write_jsonl(jsonl, spans)

        result = load_trace(trace_id, jsonl_path=jsonl)
        assert len(result) == 1
        assert result[0]["name"] == "anthropic.messages"

    def test_load_filters_by_trace_id(self, tmp_path):
        tid1 = uuid.uuid4().hex
        tid2 = uuid.uuid4().hex
        spans = [
            _make_span_dict("openai.chat.completions", tid1),
            _make_span_dict("openai.chat.completions", tid2),
        ]
        db = str(tmp_path / "traces.db")
        _write_sqlite(db, spans)

        result = load_trace(tid1, db_path=db)
        assert len(result) == 1
        assert result[0]["trace_id"] == tid1

    def test_load_jsonl_filters_by_trace_id(self, tmp_path):
        tid1 = uuid.uuid4().hex
        tid2 = uuid.uuid4().hex
        spans = [
            _make_span_dict("openai.chat.completions", tid1),
            _make_span_dict("agent.run", tid2),
        ]
        jsonl = str(tmp_path / "traces.jsonl")
        _write_jsonl(jsonl, spans)

        result = load_trace(tid1, jsonl_path=jsonl)
        assert len(result) == 1

    def test_load_no_path_raises(self):
        with pytest.raises(ValueError, match="db_path or jsonl_path"):
            load_trace("abc")

    def test_load_attributes_are_dict(self, tmp_path):
        tid = uuid.uuid4().hex
        spans = [_make_span_dict("openai.chat.completions", tid, attributes={"model": "gpt-4o"})]
        db = str(tmp_path / "traces.db")
        _write_sqlite(db, spans)

        result = load_trace(tid, db_path=db)
        assert isinstance(result[0]["attributes"], dict)
        assert result[0]["attributes"]["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# replay_trace tests (mocked LLM calls)
# ---------------------------------------------------------------------------

class TestReplayTrace:
    """All LLM SDK calls are mocked — no real API calls are made."""

    # -- OpenAI replay -------------------------------------------------------

    def test_replay_openai_sqlite(self, tmp_path):
        tid = uuid.uuid4().hex
        messages = [{"role": "user", "content": "hi"}]
        spans = [
            _make_span_dict(
                "openai.chat.completions", tid,
                attributes={"model": "gpt-4o", "input": json.dumps(messages)},
            )
        ]
        db = str(tmp_path / "traces.db")
        _write_sqlite(db, spans)

        mock_create = MagicMock()
        _exporters.clear()
        _openai_stub.chat.completions.create = mock_create

        new_tid = replay_trace(tid, db_path=db)

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["messages"] == messages
        assert isinstance(new_tid, str)
        assert new_tid != tid

    def test_replay_openai_jsonl(self, tmp_path):
        tid = uuid.uuid4().hex
        messages = [{"role": "user", "content": "hello"}]
        spans = [
            _make_span_dict(
                "openai.chat.completions", tid,
                attributes={"model": "gpt-4o-mini", "input": json.dumps(messages)},
            )
        ]
        jsonl = str(tmp_path / "traces.jsonl")
        _write_jsonl(jsonl, spans)

        mock_create = MagicMock()
        _exporters.clear()
        _openai_stub.chat.completions.create = mock_create

        new_tid = replay_trace(tid, jsonl_path=jsonl)

        mock_create.assert_called_once()
        assert new_tid != tid

    # -- Anthropic replay ----------------------------------------------------

    def test_replay_anthropic(self, tmp_path):
        tid = uuid.uuid4().hex
        messages = [{"role": "user", "content": "ping"}]
        spans = [
            _make_span_dict(
                "anthropic.messages", tid,
                attributes={
                    "model": "claude-3-5-haiku-20241022",
                    "input": json.dumps(messages),
                },
            )
        ]
        db = str(tmp_path / "traces.db")
        _write_sqlite(db, spans)

        _exporters.clear()

        mock_client = MagicMock()
        _anthropic_stub.Anthropic = MagicMock(return_value=mock_client)

        new_tid = replay_trace(tid, db_path=db)

        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-3-5-haiku-20241022"
        assert call_kwargs["messages"] == messages
        assert new_tid != tid

    # -- Bedrock replay ------------------------------------------------------

    def test_replay_bedrock(self, tmp_path):
        tid = uuid.uuid4().hex
        messages = [{"role": "user", "content": [{"text": "hi"}]}]
        spans = [
            _make_span_dict(
                "bedrock.converse", tid,
                attributes={
                    "model": "amazon.titan-text-lite-v1",
                    "input": json.dumps(messages),
                },
            )
        ]
        db = str(tmp_path / "traces.db")
        _write_sqlite(db, spans)

        _exporters.clear()

        mock_boto_client = MagicMock()
        mock_boto_client.converse.return_value = {
            "output": {"message": {"content": [{"text": "hello"}]}},
            "usage": {"inputTokens": 2, "outputTokens": 3, "totalTokens": 5},
        }
        _boto3_stub.client = MagicMock(return_value=mock_boto_client)

        new_tid = replay_trace(tid, db_path=db)

        mock_boto_client.converse.assert_called_once()
        assert new_tid != tid

    # -- Multi-span replay ---------------------------------------------------

    def test_replay_multiple_llm_spans(self, tmp_path):
        tid = uuid.uuid4().hex
        messages = [{"role": "user", "content": "q"}]
        spans = [
            _make_span_dict(
                "openai.chat.completions", tid,
                attributes={"model": "gpt-4o", "input": json.dumps(messages)},
            ),
            _make_span_dict(
                "openai.chat.completions", tid,
                attributes={"model": "gpt-4o", "input": json.dumps(messages)},
            ),
        ]
        db = str(tmp_path / "traces.db")
        _write_sqlite(db, spans)

        mock_create = MagicMock()
        _exporters.clear()
        _openai_stub.chat.completions.create = mock_create

        new_tid = replay_trace(tid, db_path=db)

        assert mock_create.call_count == 2

    # -- Shared new trace_id -------------------------------------------------

    def test_replay_returns_consistent_trace_id(self, tmp_path):
        """All replayed spans should share a single new trace_id."""
        tid = uuid.uuid4().hex
        messages = [{"role": "user", "content": "q"}]
        spans = [
            _make_span_dict(
                "openai.chat.completions", tid,
                attributes={"model": "gpt-4o", "input": json.dumps(messages)},
            ),
        ]
        db = str(tmp_path / "traces.db")
        _write_sqlite(db, spans)

        _exporters.clear()
        collector_spans = []

        class Collector:
            def export(self, span):
                collector_spans.append(span)

        _exporters.append(Collector())
        _openai_stub.chat.completions.create = MagicMock()

        new_tid = replay_trace(tid, db_path=db)

        assert new_tid != tid
        # Any spans exported during replay should carry the new_tid
        for s in collector_spans:
            assert s.trace_id == new_tid

    # -- Error cases ---------------------------------------------------------

    def test_replay_no_llm_spans_raises(self, tmp_path):
        tid = uuid.uuid4().hex
        spans = [_make_span_dict("agent.run", tid)]
        db = str(tmp_path / "traces.db")
        _write_sqlite(db, spans)

        with pytest.raises(ValueError, match="No LLM spans"):
            replay_trace(tid, db_path=db)

    def test_replay_span_without_input_skipped(self, tmp_path):
        """Spans that have no 'input' attribute should be silently skipped."""
        tid = uuid.uuid4().hex
        spans = [
            _make_span_dict(
                "openai.chat.completions", tid,
                attributes={"model": "gpt-4o"},  # no 'input' key
            )
        ]
        db = str(tmp_path / "traces.db")
        _write_sqlite(db, spans)

        _exporters.clear()
        mock_create = MagicMock()
        _openai_stub.chat.completions.create = mock_create

        # Should not raise even though there's nothing to replay
        new_tid = replay_trace(tid, db_path=db)

        mock_create.assert_not_called()
        assert isinstance(new_tid, str)
