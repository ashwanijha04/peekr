import asyncio
import pytest
from peekr.decorators import trace
from peekr.exporters import _exporters


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


def test_trace_sync_captures_span(isolated_exporters):
    @trace
    def add(a, b):
        return a + b

    result = add(1, 2)
    assert result == 3
    assert len(isolated_exporters.spans) == 1
    span = isolated_exporters.spans[0]
    assert span.status == "ok"
    assert span.duration_ms is not None


def test_trace_captures_io(isolated_exporters):
    @trace
    def greet(name):
        return f"hello {name}"

    greet("world")
    span = isolated_exporters.spans[0]
    assert "input" in span.attributes
    assert "output" in span.attributes
    assert "world" in span.attributes["input"]
    assert "hello world" in span.attributes["output"]


def test_trace_no_io(isolated_exporters):
    @trace(capture_io=False)
    def greet(name):
        return f"hello {name}"

    greet("world")
    span = isolated_exporters.spans[0]
    assert "input" not in span.attributes
    assert "output" not in span.attributes


def test_trace_custom_name(isolated_exporters):
    @trace(name="tool.custom")
    def my_func():
        pass

    my_func()
    assert isolated_exporters.spans[0].name == "tool.custom"


def test_trace_error_captured(isolated_exporters):
    @trace
    def boom():
        raise ValueError("oops")

    with pytest.raises(ValueError):
        boom()

    span = isolated_exporters.spans[0]
    assert span.status == "error"
    assert span.attributes["error"] == "oops"


def test_trace_parent_child(isolated_exporters):
    @trace(name="tool.inner")
    def inner():
        return 42

    @trace(name="agent.outer")
    def outer():
        return inner()

    outer()
    spans = isolated_exporters.spans
    assert len(spans) == 2
    inner_span = next(s for s in spans if s.name == "tool.inner")
    outer_span = next(s for s in spans if s.name == "agent.outer")
    assert inner_span.parent_id == outer_span.span_id


def test_trace_async(isolated_exporters):
    @trace
    async def async_op():
        await asyncio.sleep(0)
        return "done"

    result = asyncio.run(async_op())
    assert result == "done"
    span = isolated_exporters.spans[0]
    assert span.status == "ok"
    assert span.attributes["output"] == '"done"'


def test_trace_async_error(isolated_exporters):
    @trace
    async def async_boom():
        raise RuntimeError("async fail")

    with pytest.raises(RuntimeError):
        asyncio.run(async_boom())

    assert isolated_exporters.spans[0].status == "error"
