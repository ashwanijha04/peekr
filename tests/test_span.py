import time
from peekai.span import Span


def test_span_defaults():
    s = Span(name="test", trace_id="abc")
    assert s.name == "test"
    assert s.trace_id == "abc"
    assert s.parent_id is None
    assert s.status == "ok"
    assert s.end_time is None
    assert s.duration_ms is None


def test_span_finish():
    s = Span(name="test", trace_id="abc")
    time.sleep(0.01)
    s.finish()
    assert s.end_time is not None
    assert s.duration_ms >= 10


def test_span_to_dict():
    s = Span(name="test", trace_id="abc")
    s.attributes["model"] = "gpt-4o"
    s.finish()
    d = s.to_dict()
    assert d["name"] == "test"
    assert d["attributes"]["model"] == "gpt-4o"
    assert "duration_ms" in d


def test_span_unique_ids():
    s1 = Span(name="a", trace_id="t")
    s2 = Span(name="b", trace_id="t")
    assert s1.span_id != s2.span_id
