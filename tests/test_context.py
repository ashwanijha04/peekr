import asyncio
from agenttracer.context import start_span, end_span, get_current_span, get_or_create_trace_id


def test_no_active_span():
    assert get_current_span() is None


def test_span_becomes_current():
    span, token = start_span("test")
    assert get_current_span() is span
    end_span(span, token)
    assert get_current_span() is None


def test_parent_child_linking():
    parent, pt = start_span("parent")
    child, ct = start_span("child")
    assert child.parent_id == parent.span_id
    end_span(child, ct)
    assert get_current_span() is parent
    end_span(parent, pt)


def test_same_trace_id_within_run():
    span1, t1 = start_span("a")
    trace_id = span1.trace_id
    span2, t2 = start_span("b")
    assert span2.trace_id == trace_id
    end_span(span2, t2)
    end_span(span1, t1)


def test_async_context_propagation():
    async def inner():
        parent, pt = start_span("parent")
        child, ct = start_span("child")
        assert child.parent_id == parent.span_id
        end_span(child, ct)
        end_span(parent, pt)

    asyncio.run(inner())
