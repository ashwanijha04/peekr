"""
Tests for the LangChain callback handler. The framework is not installed —
we drive `PeekrLangChainHandler` directly, mimicking the exact signatures
LangChain uses when it dispatches callbacks.
"""
from __future__ import annotations
import uuid
import pytest

from peekr.exporters import _exporters
from peekr.patches.langchain_patch import PeekrLangChainHandler


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


def _uuid():
    return uuid.uuid4()


# ── Chain ────────────────────────────────────────────────────────────────────

def test_chain_start_end_emits_span(isolated_exporters):
    h = PeekrLangChainHandler()
    rid = _uuid()
    h.on_chain_start({"id": ["langchain", "MyChain"]},
                     {"q": "what is 2+2"}, run_id=rid)
    h.on_chain_end({"result": "4"}, run_id=rid)

    spans = isolated_exporters.spans
    assert len(spans) == 1
    assert spans[0].name == "langchain.chain.MyChain"
    assert spans[0].status == "ok"
    assert "what is 2+2" in spans[0].attributes["input"]
    assert "4" in spans[0].attributes["output"]


def test_chain_error_marks_span_error(isolated_exporters):
    h = PeekrLangChainHandler()
    rid = _uuid()
    h.on_chain_start({"id": ["x"]}, {}, run_id=rid)
    h.on_chain_error(RuntimeError("boom"), run_id=rid)

    assert isolated_exporters.spans[0].status == "error"
    assert "boom" in isolated_exporters.spans[0].attributes["error"]


def test_chain_nests_under_parent(isolated_exporters):
    h = PeekrLangChainHandler()
    parent = _uuid()
    child = _uuid()
    h.on_chain_start({"id": ["Outer"]}, {}, run_id=parent)
    h.on_chain_start({"id": ["Inner"]}, {}, run_id=child, parent_run_id=parent)
    h.on_chain_end({}, run_id=child)
    h.on_chain_end({}, run_id=parent)

    by_name = {s.name: s for s in isolated_exporters.spans}
    inner = by_name["langchain.chain.Inner"]
    outer = by_name["langchain.chain.Outer"]
    assert inner.parent_id == outer.span_id
    assert outer.parent_id is None


# ── Tool ─────────────────────────────────────────────────────────────────────

def test_tool_start_end(isolated_exporters):
    h = PeekrLangChainHandler()
    rid = _uuid()
    h.on_tool_start({"id": ["search_web"], "name": "search_web"},
                    "climate policy", run_id=rid)
    h.on_tool_end("['result-1', 'result-2']", run_id=rid)

    s = isolated_exporters.spans[0]
    assert s.name == "langchain.tool.search_web"
    assert s.attributes["input"] == "climate policy"
    assert "result-1" in s.attributes["output"]


def test_tool_error(isolated_exporters):
    h = PeekrLangChainHandler()
    rid = _uuid()
    h.on_tool_start({"id": ["t"]}, "x", run_id=rid)
    h.on_tool_error(ValueError("bad input"), run_id=rid)
    assert isolated_exporters.spans[0].status == "error"


# ── Retriever ────────────────────────────────────────────────────────────────

def test_retriever_captures_documents(isolated_exporters):
    h = PeekrLangChainHandler()
    rid = _uuid()
    h.on_retriever_start({"id": ["vectorstore_retriever"]}, "vector query", run_id=rid)
    h.on_retriever_end(["doc-a", "doc-b", "doc-c"], run_id=rid)

    s = isolated_exporters.spans[0]
    assert s.name == "langchain.retriever.vectorstore_retriever"
    assert s.attributes["document_count"] == 3


# ── LLM / Chat ──────────────────────────────────────────────────────────────

class _Gen:
    def __init__(self, text):
        self.text = text


class _LLMResult:
    def __init__(self, text, prompt_tokens, completion_tokens):
        self.generations = [[_Gen(text)]]
        self.llm_output = {
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "model_name": "gpt-4o",
        }


def test_llm_captures_tokens_and_output(isolated_exporters):
    h = PeekrLangChainHandler()
    rid = _uuid()
    h.on_llm_start({"id": ["openai", "ChatOpenAI"]},
                   ["hi"], run_id=rid,
                   invocation_params={"model": "gpt-4o"})
    h.on_llm_end(_LLMResult("hi back", 12, 4), run_id=rid)

    s = isolated_exporters.spans[0]
    assert s.attributes["tokens_input"] == 12
    assert s.attributes["tokens_output"] == 4
    assert s.attributes["tokens_total"] == 16
    assert s.attributes["model"] == "gpt-4o"
    assert s.attributes["output"] == "hi back"


def test_chat_model_start_uses_chat_name(isolated_exporters):
    h = PeekrLangChainHandler()
    rid = _uuid()
    h.on_chat_model_start({"id": ["chat", "ChatOpenAI"]},
                          [{"role": "user", "content": "hi"}],
                          run_id=rid,
                          invocation_params={"model": "gpt-4o-mini"})
    h.on_llm_end(_LLMResult("hello", 5, 2), run_id=rid)
    assert isolated_exporters.spans[0].name == "langchain.chat.ChatOpenAI"


def test_llm_error(isolated_exporters):
    h = PeekrLangChainHandler()
    rid = _uuid()
    h.on_llm_start({"id": ["openai"]}, ["hi"], run_id=rid)
    h.on_llm_error(RuntimeError("rate limit"), run_id=rid)
    assert isolated_exporters.spans[0].status == "error"
    assert "rate limit" in isolated_exporters.spans[0].attributes["error"]


# ── Agent ───────────────────────────────────────────────────────────────────

class _AgentAction:
    def __init__(self, tool, tool_input):
        self.tool = tool
        self.tool_input = tool_input


class _AgentFinish:
    def __init__(self, return_values):
        self.return_values = return_values


def test_agent_action_emits_immediate_span(isolated_exporters):
    h = PeekrLangChainHandler()
    rid = _uuid()
    h.on_agent_action(_AgentAction("search_web", "climate policy"), run_id=rid)
    s = isolated_exporters.spans[0]
    assert s.name == "langchain.agent.action"
    assert s.attributes["tool"] == "search_web"


def test_agent_finish_emits_span(isolated_exporters):
    h = PeekrLangChainHandler()
    rid = _uuid()
    h.on_agent_finish(_AgentFinish({"output": "done"}), run_id=rid)
    s = isolated_exporters.spans[0]
    assert s.name == "langchain.agent.finish"
    assert "done" in s.attributes["output"]


# ── Tree ────────────────────────────────────────────────────────────────────

def test_full_tree_chain_tool_llm(isolated_exporters):
    """Realistic shape: outer chain → tool call + LLM call, both nested."""
    h = PeekrLangChainHandler()
    chain = _uuid()
    tool = _uuid()
    llm = _uuid()

    h.on_chain_start({"id": ["AgentExecutor"]}, {"input": "q"}, run_id=chain)
    h.on_tool_start({"id": ["search"]}, "q", run_id=tool, parent_run_id=chain)
    h.on_tool_end("result", run_id=tool)
    h.on_llm_start({"id": ["openai"]}, ["q"], run_id=llm, parent_run_id=chain,
                   invocation_params={"model": "gpt-4o"})
    h.on_llm_end(_LLMResult("answer", 8, 3), run_id=llm)
    h.on_chain_end({"output": "answer"}, run_id=chain)

    by_name = {s.name: s for s in isolated_exporters.spans}
    chain_span = by_name["langchain.chain.AgentExecutor"]
    tool_span = by_name["langchain.tool.search"]
    llm_span = by_name["langchain.llm.openai"]

    assert tool_span.parent_id == chain_span.span_id
    assert llm_span.parent_id == chain_span.span_id
    assert chain_span.parent_id is None


def test_end_without_start_is_noop(isolated_exporters):
    h = PeekrLangChainHandler()
    h.on_chain_end({}, run_id=_uuid())
    assert isolated_exporters.spans == []
