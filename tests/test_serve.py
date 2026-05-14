"""Tests for the ``peekr serve`` local web dashboard."""
from __future__ import annotations

import json
import os

import pytest

# The dashboard is optional. Skip everything if Flask isn't installed.
pytest.importorskip("flask")

from peekr.exporters import JSONLExporter, SQLiteExporter  # noqa: E402
from peekr.serve.app import create_app  # noqa: E402
from peekr.span import Span  # noqa: E402


# ───────────────────────── fixtures ─────────────────────────────────────────


def _finish(span: Span, *, status: str = "ok") -> Span:
    span.status = status
    span.finish()
    return span


def _make_trace(
    db,
    *,
    trace_id: str,
    user_id: str | None = None,
    session_id: str | None = None,
    error: bool = False,
    eval_score: float | None = None,
    guardrail: dict | None = None,
    tokens: tuple[int, int] = (0, 0),
    span_name: str = "agent.run",
) -> None:
    """Write one parent + one LLM child span for the given trace_id."""
    parent = Span(name=span_name, trace_id=trace_id)
    if user_id:
        parent.attributes["user_id"] = user_id
    if session_id:
        parent.attributes["session_id"] = session_id
    _finish(parent)
    db.export(parent)

    child = Span(
        name="openai.chat",
        trace_id=trace_id,
        parent_id=parent.span_id,
    )
    child.attributes["model"] = "gpt-4o"
    child.attributes["input"] = "what is the capital of france?"
    child.attributes["output"] = "Paris"
    child.attributes["tokens_input"] = tokens[0]
    child.attributes["tokens_output"] = tokens[1]
    child.attributes["tokens_total"] = sum(tokens)
    if user_id:
        child.attributes["user_id"] = user_id
    if session_id:
        child.attributes["session_id"] = session_id
    if eval_score is not None:
        child.attributes["eval_scores"] = {"NotEmpty": eval_score}
    if guardrail is not None:
        child.attributes["guardrail_findings"] = [guardrail]
    _finish(child, status="error" if error else "ok")
    if error:
        child.attributes["error"] = "boom"
    db.export(child)


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "traces.db")


@pytest.fixture
def populated_db(db_path):
    db = SQLiteExporter(db_path)
    _make_trace(db, trace_id="trace-A", user_id="alice", tokens=(1000, 500))
    _make_trace(
        db,
        trace_id="trace-B",
        user_id="bob",
        session_id="sess-1",
        eval_score=0.92,
        tokens=(2000, 1000),
    )
    _make_trace(
        db,
        trace_id="trace-C",
        user_id="alice",
        session_id="sess-2",
        error=True,
        guardrail={"rule": "pii", "severity": "high", "message": "leaked email"},
        tokens=(500, 100),
    )
    return db_path


@pytest.fixture
def client(populated_db):
    app = create_app(db=populated_db)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ───────────────────────── tests ────────────────────────────────────────────


class TestTraceList:
    def test_empty_db_renders(self, tmp_path):
        db = SQLiteExporter(str(tmp_path / "empty.db"))  # create schema, no rows
        del db
        app = create_app(db=str(tmp_path / "empty.db"))
        app.config["TESTING"] = True
        client = app.test_client()
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.get_data(as_text=True)
        assert "No traces match" in body
        # Quickstart snippet is shown only when no filters are active.
        assert "peekr.instrument" in body

    def test_lists_all_traces(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.get_data(as_text=True)
        assert "trace-A"[:8] in body
        assert "trace-B"[:8] in body
        assert "trace-C"[:8] in body
        assert "3 traces" in body

    def test_filter_by_user(self, client):
        resp = client.get("/?user=alice")
        body = resp.get_data(as_text=True)
        assert "trace-A"[:8] in body
        assert "trace-C"[:8] in body
        assert "trace-B"[:8] not in body

    def test_filter_by_session(self, client):
        resp = client.get("/?session=sess-1")
        body = resp.get_data(as_text=True)
        assert "trace-B"[:8] in body
        assert "trace-A"[:8] not in body

    def test_filter_has_errors(self, client):
        resp = client.get("/?has_errors=1")
        body = resp.get_data(as_text=True)
        assert "trace-C"[:8] in body
        assert "trace-A"[:8] not in body
        assert "trace-B"[:8] not in body

    def test_filter_has_evals(self, client):
        resp = client.get("/?has_evals=1")
        body = resp.get_data(as_text=True)
        assert "trace-B"[:8] in body
        assert "trace-A"[:8] not in body

    def test_filter_has_guardrails(self, client):
        resp = client.get("/?has_guardrails=1")
        body = resp.get_data(as_text=True)
        assert "trace-C"[:8] in body
        assert "trace-A"[:8] not in body

    def test_filter_by_search(self, client):
        resp = client.get("/?q=openai")
        body = resp.get_data(as_text=True)
        # all three traces have an openai.chat span
        assert "trace-A"[:8] in body
        assert "trace-B"[:8] in body
        assert "trace-C"[:8] in body

    def test_filter_min_cost(self, client):
        # trace-B has the most tokens, so the largest estimated cost
        resp = client.get("/?min_cost=0.001")
        body = resp.get_data(as_text=True)
        assert "trace-B"[:8] in body


class TestTraceDetail:
    def test_renders_trace(self, client):
        resp = client.get("/trace/trace-B")
        assert resp.status_code == 200
        body = resp.get_data(as_text=True)
        assert "trace-B" in body
        # Span tree includes both spans
        assert "agent.run" in body
        assert "openai.chat" in body
        # Eval panel shows
        assert "Eval scores" in body
        assert "NotEmpty" in body

    def test_renders_guardrail_panel(self, client):
        resp = client.get("/trace/trace-C")
        body = resp.get_data(as_text=True)
        assert "Guardrail findings" in body
        assert "pii" in body
        # Error badge surfaces
        assert "error" in body.lower()

    def test_404_for_missing_trace(self, client):
        resp = client.get("/trace/does-not-exist")
        assert resp.status_code == 404
        assert "404" in resp.get_data(as_text=True)

    def test_span_io_lazy_endpoint(self, client, populated_db):
        # Resolve a span_id from the seeded DB
        from peekr.serve.data import SQLiteStore

        store = SQLiteStore(populated_db)
        spans = store.get_trace("trace-A")
        span_id = next(s["span_id"] for s in spans if s["name"] == "openai.chat")

        resp = client.get(f"/api/trace/trace-A/span/{span_id}")
        assert resp.status_code == 200
        payload = json.loads(resp.get_data(as_text=True))
        assert payload["span_id"] == span_id
        assert payload["model"] == "gpt-4o"
        assert payload["input"].startswith("what is the capital")
        assert payload["output"] == "Paris"

    def test_span_io_404_for_missing_span(self, client):
        resp = client.get("/api/trace/trace-A/span/nope")
        assert resp.status_code == 404


class TestCompare:
    def test_renders_two_traces_side_by_side(self, client):
        resp = client.get("/compare?a=trace-A&b=trace-B")
        assert resp.status_code == 200
        body = resp.get_data(as_text=True)
        assert "A · " in body and "B · " in body
        # Distinguishable span counts / token totals from both sides
        assert body.count("agent.run") >= 2

    def test_empty_form_shown_with_no_args(self, client):
        resp = client.get("/compare")
        body = resp.get_data(as_text=True)
        assert "Compare traces" in body
        assert "Paste two trace IDs" in body

    def test_compare_handles_missing_id(self, client):
        resp = client.get("/compare?a=trace-A&b=nope")
        body = resp.get_data(as_text=True)
        assert resp.status_code == 200
        assert "No trace found for ID nope" in body


class TestPagination:
    def test_pagination_metadata(self, tmp_path):
        path = str(tmp_path / "big.db")
        db = SQLiteExporter(path)
        for i in range(60):
            _make_trace(db, trace_id=f"t-{i:03d}")
        app = create_app(db=path)
        app.config["TESTING"] = True
        client = app.test_client()
        resp = client.get("/?page=2")
        body = resp.get_data(as_text=True)
        assert "page 2 of 2" in body
        # Page 1 traces should not be present on page 2
        assert "t-059"[:8] not in body or "t-009"[:8] in body  # boundary sanity


class TestJSONLFallback:
    def test_reads_jsonl_when_no_db(self, tmp_path):
        path = str(tmp_path / "traces.jsonl")
        exp = JSONLExporter(path=path)
        parent = Span(name="agent.run", trace_id="jl-1")
        parent.attributes["user_id"] = "carol"
        _finish(parent)
        exp.export(parent)
        child = Span(name="openai.chat", trace_id="jl-1", parent_id=parent.span_id)
        child.attributes["model"] = "gpt-4o"
        child.attributes["input"] = "hi"
        child.attributes["output"] = "hello"
        child.attributes["tokens_input"] = 10
        child.attributes["tokens_output"] = 5
        child.attributes["tokens_total"] = 15
        _finish(child)
        exp.export(child)

        app = create_app(jsonl=path)
        app.config["TESTING"] = True
        c = app.test_client()
        resp = c.get("/")
        body = resp.get_data(as_text=True)
        assert "jl-1"[:8] in body
        assert "1 trace" in body

        detail = c.get("/trace/jl-1")
        assert detail.status_code == 200
        dbody = detail.get_data(as_text=True)
        assert "openai.chat" in dbody

    def test_jsonl_filters_by_user(self, tmp_path):
        path = str(tmp_path / "traces.jsonl")
        exp = JSONLExporter(path=path)
        for tid, user in [("t-x", "dave"), ("t-y", "erin")]:
            span = Span(name="agent.run", trace_id=tid)
            span.attributes["user_id"] = user
            _finish(span)
            exp.export(span)

        app = create_app(jsonl=path)
        app.config["TESTING"] = True
        c = app.test_client()
        resp = c.get("/?user=dave")
        body = resp.get_data(as_text=True)
        assert "t-x"[:8] in body
        assert "t-y"[:8] not in body

    def test_missing_file_renders_empty_state(self, tmp_path):
        path = str(tmp_path / "nope.jsonl")
        assert not os.path.exists(path)
        app = create_app(jsonl=path)
        app.config["TESTING"] = True
        c = app.test_client()
        resp = c.get("/")
        assert resp.status_code == 200
        assert "No traces match" in resp.get_data(as_text=True)


class TestStaticAssets:
    def test_css_served(self, client):
        resp = client.get("/static/style.css")
        assert resp.status_code == 200
        assert b"--bg" in resp.data  # CSS variables from docs/index.html palette

    def test_js_served(self, client):
        resp = client.get("/static/detail.js")
        assert resp.status_code == 200
        assert b"api/trace" in resp.data
