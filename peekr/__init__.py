from __future__ import annotations
from .exporters import add_exporter, JSONLExporter, ConsoleExporter, SQLiteExporter
from .context import start_span, end_span, get_current_span
from .decorators import trace
from .session import session, set_grounding, get_grounding
from .feedback import feedback, export_feedback
from .experiment import experiment
from .alerts import alert
from . import eval as eval  # noqa: A001 — subpackage, not the builtin
from . import guardrails as guardrails  # subpackage
from .patches.openai_patch import patch_openai
from .patches.anthropic_patch import patch_anthropic
from .patches.bedrock_patch import patch_bedrock
from .patches.gemini_patch import patch_gemini

_patched = False


def instrument(
    exporter=None,
    console: bool = True,
    storage: str = "jsonl",       # "jsonl" | "sqlite" | "both"
    jsonl_path: str = "traces.jsonl",
    db_path: str = "traces.db",
    alerts: list = None,
    evaluators: list = None,
    guards: list = None,
):
    """
    Auto-instrument OpenAI, Anthropic, Bedrock, and Gemini SDKs.
    Call once before any LLM calls.

    storage="jsonl"     → write to traces.jsonl (default)
    storage="sqlite"    → write to traces.db (multi-process safe, queryable)
    storage="both"      → write to both

    alerts=[...]        → fire callbacks when thresholds are crossed
    evaluators=[...]    → score LLM outputs after each call
    guards=[...]        → run guardrails on every LLM call's input and output
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

    if alerts:
        from .alerts import AlertExporter
        add_exporter(AlertExporter(alerts))

    if evaluators:
        from .eval import EvalExporter
        add_exporter(EvalExporter(evaluators))

    if guards:
        from .guardrails import GuardrailExporter
        add_exporter(GuardrailExporter(guards))

    if not _patched:
        patch_openai()
        patch_anthropic()
        patch_bedrock()
        patch_gemini()
        _patched = True


__all__ = [
    # core
    "instrument", "trace", "session",
    "set_grounding", "get_grounding",
    "start_span", "end_span", "get_current_span",
    # exporters
    "JSONLExporter", "ConsoleExporter", "SQLiteExporter", "add_exporter",
    # features
    "feedback", "export_feedback",
    "experiment",
    "alert",
    "eval",
    "guardrails",
]
