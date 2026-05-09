from __future__ import annotations
from .exporters import add_exporter, JSONLExporter, ConsoleExporter, SQLiteExporter
from .context import start_span, end_span, get_current_span
from .decorators import trace
from .patches.openai_patch import patch_openai
from .patches.anthropic_patch import patch_anthropic

_patched = False


def instrument(
    exporter=None,
    console: bool = True,
    storage: str = "jsonl",       # "jsonl" | "sqlite" | "both"
    jsonl_path: str = "traces.jsonl",
    db_path: str = "traces.db",
):
    """
    Auto-instrument OpenAI and Anthropic SDKs.
    Call once before any LLM calls.

    storage="jsonl"   → write to traces.jsonl (default)
    storage="sqlite"  → write to traces.db (multi-process safe, queryable)
    storage="both"    → write to both
    """
    global _patched

    if exporter:
        add_exporter(exporter)
    else:
        if console:
            add_exporter(ConsoleExporter())
        if storage in ("jsonl", "both"):
            add_exporter(JSONLExporter(jsonl_path))
        if storage in ("sqlite", "both"):
            add_exporter(SQLiteExporter(db_path))

    if not _patched:
        patch_openai()
        patch_anthropic()
        _patched = True


__all__ = [
    "instrument", "trace",
    "start_span", "end_span", "get_current_span",
    "JSONLExporter", "ConsoleExporter", "SQLiteExporter",
]
