"""
Tests for the LlamaIndex callback handler. We don't import llama_index —
we drive `PeekrLlamaIndexHandler` directly using the same payload shapes
LlamaIndex passes to `on_event_start` / `on_event_end`.
"""
from __future__ import annotations
import pytest

from peekr.exporters import _exporters
from peekr.patches.llamaindex_patch import PeekrLlamaIndexHandler


class CollectingExporter:
    def __init__(self):
        self.spans = []

    def export(self, span):
        self.spans.append(span)


@pytest.fixture(autouse=True)
def isolated_exporters():
    _exporters.clear()
    col = CollectingExporter()
    _exporters.append(col)
    yield col
    _exporters.clear()


# ── LLM events ──────────────────────────────────────────────────────────────

class _Usage:
    def __init__(self, prompt, completion):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion


class _Raw:
    def __init__(self, prompt, completion):
        self.usage = _Usage(prompt, completion)


class _Response:
    def __init__(self, text, prompt, completion):
        self.text = text
        self.raw = _Raw(prompt, completion)


def test_llm_event_captures_usage(isolated_exporters):
    h = PeekrLlamaIndexHandler()
    h.on_event_start("llm",
                     payload={"messages": [{"role": "user", "content": "hi"}],
                              "serialized": {"model": "gpt-4o"}},
                     event_id="e1")
    h.on_event_end("llm",
                   payload={"response": _Response("hello", 10, 4)},
                   event_id="e1")

    s = isolated_exporters.spans[0]
    assert s.name == "llamaindex.llm"
    assert s.attributes["model"] == "gpt-4o"
    assert s.attributes["tokens_input"] == 10
    assert s.attributes["tokens_output"] == 4
    assert s.attributes["tokens_total"] == 14
    assert s.attributes["output"] == "hello"


def test_llm_event_no_usage_doesnt_crash(isolated_exporters):
    h = PeekrLlamaIndexHandler()
    h.on_event_start("llm", payload={"prompt": "hi"}, event_id="e1")
    h.on_event_end("llm", payload={}, event_id="e1")
    s = isolated_exporters.spans[0]
    assert s.status == "ok"


# ── Retrieve events ─────────────────────────────────────────────────────────

def test_retrieve_event_captures_documents(isolated_exporters):
    h = PeekrLlamaIndexHandler()
    h.on_event_start("retrieve", payload={"query_str": "climate"}, event_id="r1")
    h.on_event_end("retrieve",
                   payload={"nodes": [{"text": "n1"}, {"text": "n2"}, {"text": "n3"}]},
                   event_id="r1")

    s = isolated_exporters.spans[0]
    assert s.name == "llamaindex.retrieve"
    assert s.attributes["input"] == "climate"
    assert s.attributes["document_count"] == 3


# ── Query / Agent step ──────────────────────────────────────────────────────

def test_query_event_captures_input_output(isolated_exporters):
    h = PeekrLlamaIndexHandler()
    h.on_event_start("query", payload={"query_str": "what is X"}, event_id="q1")
    h.on_event_end("query", payload={"response": "X is Y"}, event_id="q1")

    s = isolated_exporters.spans[0]
    assert s.name == "llamaindex.query"
    assert s.attributes["input"] == "what is X"
    assert s.attributes["output"] == "X is Y"


def test_agent_step_event(isolated_exporters):
    h = PeekrLlamaIndexHandler()
    h.on_event_start("agent_step",
                     payload={"messages": [{"role": "user", "content": "do thing"}]},
                     event_id="a1")
    h.on_event_end("agent_step", payload={}, event_id="a1")

    s = isolated_exporters.spans[0]
    assert s.name == "llamaindex.agent_step"
    assert "do thing" in s.attributes["input"]


# ── Function call ───────────────────────────────────────────────────────────

def test_function_call_event(isolated_exporters):
    h = PeekrLlamaIndexHandler()
    h.on_event_start("function_call",
                     payload={"tool": "calculator", "function_call": "add(1,2)"},
                     event_id="f1")
    h.on_event_end("function_call",
                   payload={"function_call_response": "3"},
                   event_id="f1")

    s = isolated_exporters.spans[0]
    assert s.name == "llamaindex.function_call"
    assert s.attributes["output"] == "3"


# ── Parent/child nesting ────────────────────────────────────────────────────

def test_events_nest_via_parent_id(isolated_exporters):
    h = PeekrLlamaIndexHandler()
    h.on_event_start("query", payload={"query_str": "q"}, event_id="root")
    h.on_event_start("retrieve", payload={"query_str": "q"},
                     event_id="ret", parent_id="root")
    h.on_event_end("retrieve", payload={"nodes": []}, event_id="ret")
    h.on_event_start("llm", payload={"messages": []},
                     event_id="llm", parent_id="root")
    h.on_event_end("llm",
                   payload={"response": _Response("ok", 1, 1)},
                   event_id="llm")
    h.on_event_end("query", payload={"response": "ok"}, event_id="root")

    by_name = {s.name: s for s in isolated_exporters.spans}
    root = by_name["llamaindex.query"]
    ret = by_name["llamaindex.retrieve"]
    llm = by_name["llamaindex.llm"]
    assert ret.parent_id == root.span_id
    assert llm.parent_id == root.span_id
    assert root.parent_id is None


# ── Error propagation ───────────────────────────────────────────────────────

def test_payload_exception_marks_error(isolated_exporters):
    h = PeekrLlamaIndexHandler()
    h.on_event_start("llm", payload={}, event_id="e")
    h.on_event_end("llm", payload={"exception": RuntimeError("oops")},
                   event_id="e")
    s = isolated_exporters.spans[0]
    assert s.status == "error"
    assert "oops" in s.attributes["error"]


def test_end_without_start_is_noop(isolated_exporters):
    h = PeekrLlamaIndexHandler()
    h.on_event_end("llm", payload={}, event_id="nope")
    assert isolated_exporters.spans == []


# ── Enum-style event types ──────────────────────────────────────────────────

class _FakeEventType:
    def __init__(self, v): self.value = v


def test_enum_event_type_lowercased(isolated_exporters):
    h = PeekrLlamaIndexHandler()
    h.on_event_start(_FakeEventType("LLM"), payload={"prompt": "x"}, event_id="e")
    h.on_event_end(_FakeEventType("LLM"), payload={}, event_id="e")
    assert isolated_exporters.spans[0].name == "llamaindex.llm"
