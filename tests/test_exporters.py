import json
import os
import tempfile
import pytest
from agenttracer.span import Span
from agenttracer.exporters import JSONLExporter, ConsoleExporter, _exporters, add_exporter, export_span


@pytest.fixture(autouse=True)
def clear_exporters():
    _exporters.clear()
    yield
    _exporters.clear()


def make_span(name="test"):
    s = Span(name=name, trace_id="trace123")
    s.finish()
    return s


def test_jsonl_exporter_writes_line():
    with tempfile.NamedTemporaryFile(mode="r", suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        exporter = JSONLExporter(path=path)
        exporter.export(make_span("op.one"))
        exporter.export(make_span("op.two"))
        with open(path) as f:
            lines = [json.loads(l) for l in f]
        assert len(lines) == 2
        assert lines[0]["name"] == "op.one"
        assert lines[1]["name"] == "op.two"
    finally:
        os.unlink(path)


def test_jsonl_exporter_valid_json():
    with tempfile.NamedTemporaryFile(mode="r", suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        s = make_span()
        s.attributes["model"] = "gpt-4o"
        s.attributes["tokens_total"] = 42
        JSONLExporter(path=path).export(s)
        with open(path) as f:
            data = json.loads(f.read())
        assert data["attributes"]["model"] == "gpt-4o"
        assert data["duration_ms"] is not None
    finally:
        os.unlink(path)


def test_console_exporter_prints(capsys):
    ConsoleExporter().export(make_span("my.op"))
    out = capsys.readouterr().out
    assert "my.op" in out


def test_export_span_dispatches_to_all():
    collected = []

    class FakeExporter:
        def export(self, span):
            collected.append(span.name)

    add_exporter(FakeExporter())
    add_exporter(FakeExporter())
    export_span(make_span("dispatched"))
    assert collected == ["dispatched", "dispatched"]
