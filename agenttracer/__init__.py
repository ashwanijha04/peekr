from __future__ import annotations
from .exporters import add_exporter, JSONLExporter, ConsoleExporter
from .context import start_span, end_span, get_current_span
from .decorators import trace
from .patches.openai_patch import patch_openai
from .patches.anthropic_patch import patch_anthropic


def instrument(exporter=None, console: bool = True, jsonl_path: str | None = "traces.jsonl"):
    """
    Auto-instrument OpenAI and Anthropic SDKs.
    Call this once before any LLM calls.
    """
    if exporter:
        add_exporter(exporter)
    else:
        if console:
            add_exporter(ConsoleExporter())
        if jsonl_path:
            add_exporter(JSONLExporter(jsonl_path))

    patch_openai()
    patch_anthropic()


__all__ = ["instrument", "trace", "start_span", "end_span", "get_current_span", "JSONLExporter", "ConsoleExporter"]
