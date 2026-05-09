import json
import os
from .span import Span


class JSONLExporter:
    def __init__(self, path: str = "traces.jsonl"):
        self.path = path

    def export(self, span: Span) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(span.to_dict()) + "\n")


class ConsoleExporter:
    def export(self, span: Span) -> None:
        duration = f"{span.duration_ms:.1f}ms" if span.duration_ms else "?"
        indent = "  " if span.parent_id else ""
        attrs = ""
        if "model" in span.attributes:
            attrs += f" model={span.attributes['model']}"
        if "tokens_total" in span.attributes:
            attrs += f" tokens={span.attributes['tokens_total']}"
        print(f"{indent}[{span.name}] {duration}{attrs}")


_exporters: list = []


def add_exporter(exporter) -> None:
    _exporters.append(exporter)


def export_span(span: Span) -> None:
    for exporter in _exporters:
        exporter.export(span)
