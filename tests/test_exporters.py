import json
import os
import tempfile
import pytest
from peekr.span import Span
from peekr.exporters import (
    JSONLExporter,
    ConsoleExporter,
    _exporters,
    add_exporter,
    export_span,
)


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
            lines = [json.loads(line) for line in f]
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


def test_clear_exporters_public_api():
    """clear_exporters() empties the registry — no private _exporters poking."""
    from peekr.exporters import clear_exporters as clear_fn

    add_exporter(ConsoleExporter())
    add_exporter(ConsoleExporter())
    assert len(_exporters) == 2
    clear_fn()
    assert _exporters == []
    # Also exported at package top level.
    import peekr

    assert peekr.clear_exporters is clear_fn


def test_instrument_none_path_disables_storage(tmp_path):
    """jsonl_path=None / db_path=None must not register broken exporters.

    JSONLExporter(None) raises TypeError on every span export, so a None
    path means "this backend is disabled".
    """
    from peekr import instrument

    instrument(console=False, jsonl_path=None)
    assert not any(isinstance(e, JSONLExporter) for e in _exporters)

    _exporters.clear()
    from peekr.exporters import SQLiteExporter

    instrument(console=False, storage="both", jsonl_path=None, db_path=None)
    assert not any(isinstance(e, (JSONLExporter, SQLiteExporter)) for e in _exporters)

    # A real path still registers normally.
    _exporters.clear()
    instrument(console=False, jsonl_path=str(tmp_path / "traces.jsonl"))
    assert any(isinstance(e, JSONLExporter) for e in _exporters)
