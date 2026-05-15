from __future__ import annotations
from .exporters import add_exporter, JSONLExporter, ConsoleExporter, SQLiteExporter
from .context import start_span, end_span, get_current_span
from .decorators import trace
from .session import session
from .feedback import feedback, export_feedback
from .experiment import experiment
from .alerts import alert
from . import eval as eval  # noqa: A001 — subpackage, not the builtin
from .patches.openai_patch import patch_openai
from .patches.anthropic_patch import patch_anthropic
from .patches.bedrock_patch import patch_bedrock
from .patches.langchain_patch import patch_langchain
from .patches.llamaindex_patch import patch_llamaindex
from .patches.crewai_patch import patch_crewai

_patched = False


def instrument(
    exporter=None,
    console: bool = True,
    storage: str = "jsonl",       # "jsonl" | "sqlite" | "both"
    jsonl_path: str = "traces.jsonl",
    db_path: str = "traces.db",
    alerts: list = None,
    evaluators: list = None,
    evaluate_filter=None,
):
    """
    Auto-instrument LLM SDKs (OpenAI, Anthropic, Bedrock) and agent
    frameworks (LangChain, LlamaIndex, CrewAI). Call once before any
    LLM calls.

    storage="jsonl"     → write to traces.jsonl (default)
    storage="sqlite"    → write to traces.db (multi-process safe, queryable)
    storage="both"      → write to both

    alerts=[...]        → fire callbacks when thresholds are crossed
    evaluators=[...]    → score LLM outputs after each call
    """
    global _patched

    # Order matters. Exporters run in the order they're registered, on the
    # SAME span object. EvalExporter and AlertExporter mutate span.attributes
    # (eval_scores / alerts), so they MUST run before any storage exporter —
    # otherwise JSONL/SQLite serialize the span before the scores are written
    # and you get traces with empty `eval_scores`. (Reported by users running
    # peekr with a real RAG workload.)
    if exporter:
        add_exporter(exporter)
    else:
        # 1) Console first — purely a display tool, doesn't mutate.
        if console:
            add_exporter(ConsoleExporter())
        # 2) Mutators (eval + alerts) BEFORE persistence.
        if evaluators:
            from .eval import EvalExporter
            add_exporter(EvalExporter(evaluators, span_filter=evaluate_filter))
        if alerts:
            from .alerts import AlertExporter
            add_exporter(AlertExporter(alerts))
        # 3) Persistence last — what's written to disk now includes any
        #    attributes the mutators added (notably eval_scores).
        if storage in ("jsonl", "both"):
            add_exporter(JSONLExporter(jsonl_path))
        if storage in ("sqlite", "both"):
            add_exporter(SQLiteExporter(db_path))

    if not _patched:
        patch_openai()
        patch_anthropic()
        patch_bedrock()
        patch_langchain()
        patch_llamaindex()
        patch_crewai()
        _patched = True


__all__ = [
    # core
    "instrument", "trace", "session",
    "start_span", "end_span", "get_current_span",
    # exporters
    "JSONLExporter", "ConsoleExporter", "SQLiteExporter", "add_exporter",
    # features
    "feedback", "export_feedback",
    "experiment",
    "alert",
    "eval",
]
