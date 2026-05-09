"""
Tests for streaming token capture.
We don't need real API keys — we mock the stream objects directly.
"""
import pytest
from peekr.exporters import _exporters
from peekr.patches.openai_patch import _OpenAIStreamWrapper
from peekr.patches.anthropic_patch import _AnthropicStreamWrapper
from peekr.context import start_span, end_span


class CollectingExporter:
    def __init__(self):
        self.spans = []

    def export(self, span):
        self.spans.append(span)


@pytest.fixture(autouse=True)
def isolated_exporters():
    _exporters.clear()
    collector = CollectingExporter()
    _exporters.append(collector)
    yield collector
    _exporters.clear()


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_openai_chunk(content=None, usage=None):
    """Minimal mock of an OpenAI stream chunk."""
    class Delta:
        pass
    class Choice:
        delta = Delta()
    class Chunk:
        choices = [Choice()]
        pass

    chunk = Chunk()
    if content is not None:
        chunk.choices[0].delta.content = content
    if usage is not None:
        class Usage:
            prompt_tokens = usage[0]
            completion_tokens = usage[1]
            total_tokens = usage[0] + usage[1]
        chunk.usage = Usage()
    else:
        chunk.usage = None
    return chunk


def _make_anthropic_event(type_, input_tokens=None, output_tokens=None):
    """Minimal mock of an Anthropic stream event."""
    class Event:
        pass

    event = Event()
    event.type = type_

    if type_ == "message_start" and input_tokens is not None:
        class Usage:
            pass
        class Message:
            pass
        u = Usage()
        u.input_tokens = input_tokens
        m = Message()
        m.usage = u
        event.message = m

    if type_ == "message_delta" and output_tokens is not None:
        class Usage:
            pass
        u = Usage()
        u.output_tokens = output_tokens
        event.usage = u

    return event


def _span_and_token():
    return start_span("test")


# ── OpenAI stream tests ───────────────────────────────────────────────────────

def test_openai_stream_captures_tokens(isolated_exporters):
    chunks = [
        _make_openai_chunk(content="Hello"),
        _make_openai_chunk(content=" world"),
        _make_openai_chunk(usage=(10, 5)),   # final usage chunk
    ]
    span, token = _span_and_token()
    wrapper = _OpenAIStreamWrapper(iter(chunks), span, token)

    list(wrapper)  # consume

    assert isolated_exporters.spans[0].attributes["tokens_input"]  == 10
    assert isolated_exporters.spans[0].attributes["tokens_output"] == 5
    assert isolated_exporters.spans[0].attributes["tokens_total"]  == 15


def test_openai_stream_status_ok(isolated_exporters):
    span, token = _span_and_token()
    wrapper = _OpenAIStreamWrapper(iter([_make_openai_chunk(content="hi")]), span, token)
    list(wrapper)
    assert isolated_exporters.spans[0].status == "ok"


def test_openai_stream_status_error(isolated_exporters):
    def bad_stream():
        yield _make_openai_chunk(content="hi")
        raise RuntimeError("stream died")

    span, token = _span_and_token()
    wrapper = _OpenAIStreamWrapper(bad_stream(), span, token)

    with pytest.raises(RuntimeError):
        list(wrapper)

    assert isolated_exporters.spans[0].status == "error"
    assert "stream died" in isolated_exporters.spans[0].attributes["error"]


def test_openai_stream_span_exported_once(isolated_exporters):
    span, token = _span_and_token()
    wrapper = _OpenAIStreamWrapper(iter([_make_openai_chunk(content="x")]), span, token)
    list(wrapper)
    list(wrapper)  # exhaust again — should not double-export
    assert len(isolated_exporters.spans) == 1


def test_openai_stream_context_manager(isolated_exporters):
    chunks = [_make_openai_chunk(content="hi"), _make_openai_chunk(usage=(8, 4))]

    class FakeStream:
        def __iter__(self): return iter(chunks)
        def __enter__(self): return self
        def __exit__(self, *a): return False

    span, token = _span_and_token()
    wrapper = _OpenAIStreamWrapper(FakeStream(), span, token)

    with wrapper as s:
        list(s)

    assert isolated_exporters.spans[0].attributes["tokens_total"] == 12


# ── Anthropic stream tests ────────────────────────────────────────────────────

def test_anthropic_stream_captures_tokens(isolated_exporters):
    events = [
        _make_anthropic_event("message_start", input_tokens=20),
        _make_anthropic_event("content_block_delta"),
        _make_anthropic_event("message_delta", output_tokens=8),
        _make_anthropic_event("message_stop"),
    ]
    span, token = _span_and_token()
    wrapper = _AnthropicStreamWrapper(iter(events), span, token)

    list(wrapper)

    assert isolated_exporters.spans[0].attributes["tokens_input"]  == 20
    assert isolated_exporters.spans[0].attributes["tokens_output"] == 8
    assert isolated_exporters.spans[0].attributes["tokens_total"]  == 28


def test_anthropic_stream_status_ok(isolated_exporters):
    span, token = _span_and_token()
    wrapper = _AnthropicStreamWrapper(
        iter([_make_anthropic_event("message_start", input_tokens=5)]),
        span, token
    )
    list(wrapper)
    assert isolated_exporters.spans[0].status == "ok"


def test_anthropic_stream_status_error(isolated_exporters):
    def bad_stream():
        yield _make_anthropic_event("message_start", input_tokens=5)
        raise ValueError("anthropic stream error")

    span, token = _span_and_token()
    wrapper = _AnthropicStreamWrapper(bad_stream(), span, token)

    with pytest.raises(ValueError):
        list(wrapper)

    assert isolated_exporters.spans[0].status == "error"


def test_anthropic_stream_exported_once(isolated_exporters):
    span, token = _span_and_token()
    wrapper = _AnthropicStreamWrapper(iter([_make_anthropic_event("message_stop")]), span, token)
    list(wrapper)
    list(wrapper)
    assert len(isolated_exporters.spans) == 1
