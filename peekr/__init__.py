from __future__ import annotations
from .exporters import (
    add_exporter,
    clear_exporters,
    JSONLExporter,
    ConsoleExporter,
    SQLiteExporter,
    HTTPExporter,
)
from .otel import OTelExporter
from .context import (
    start_span,
    end_span,
    get_current_span,
    set_process_defaults,
)
from .decorators import trace
from .session import session
from .feedback import feedback, export_feedback
from .experiment import experiment
from .alerts import alert
from . import eval as eval  # noqa: A001 — subpackage, not the builtin
from . import guard
from . import prompts as prompts  # noqa: PLC0414 — public re-export
from .middleware import FastAPIMiddleware, PeekrASGIMiddleware  # noqa: F401
from .harnesses.rag import instrument_rag
from .evidence import record_evidence, clear_evidence, evidence  # noqa: F401
from .patches.openai_patch import patch_openai
from .patches.anthropic_patch import patch_anthropic
from .patches.bedrock_patch import patch_bedrock
from .patches.gemini_patch import patch_gemini
from .patches.langchain_patch import patch_langchain
from .patches.llamaindex_patch import patch_llamaindex
from .patches.crewai_patch import patch_crewai
from .patches.litellm_patch import patch_litellm
from .patches.pinecone_patch import patch_pinecone
from .patches.chroma_patch import patch_chroma
from .patches.qdrant_patch import patch_qdrant
from .budget import BudgetAlert, BudgetExceededError, budget_alert

_patched = False

# Sentinel: user explicitly passed compliance=[] (disabled).
# Distinct from compliance=None (auto-discover from dashboard).
_COMPLIANCE_DISABLED = object()


def instrument(
    exporter=None,
    console: bool = True,
    storage: str = "jsonl",  # "jsonl" | "sqlite" | "both"
    jsonl_path: str = "traces.jsonl",
    db_path: str = "traces.db",
    alerts: list = None,
    evaluators: list = None,
    evaluate_filter=None,
    async_eval: bool = True,
    guardrails: list = None,
    compliance=None,  # None = auto-discover from dashboard | ["HIPAA"] = explicit | [] = disable
    tenant_id: str | None = None,
    retention_class: str | None = None,
    sample_rate: float | None = None,
    keep_errors: bool | None = None,
    budget_alert=None,  # e.g. BudgetAlert(limit_usd=10.0, window="day")
):
    """
    Auto-instrument LLM SDKs (OpenAI, Anthropic, Bedrock) and agent
    frameworks (LangChain, LlamaIndex, CrewAI). Call once before any
    LLM calls.

    storage="jsonl"     → write to traces.jsonl (default)
    storage="sqlite"    → write to traces.db (multi-process safe, queryable)
    storage="both"      → write to both
    jsonl_path=None     → disable JSONL storage (likewise db_path=None for
                          SQLite). Passing exporter=... also suppresses the
                          default storage pipeline entirely.

    alerts=[...]        → fire callbacks when thresholds are crossed
    evaluators=[...]    → score LLM outputs after each call
    async_eval=False    → run evaluators synchronously in the export path
                          (default True = background thread, zero caller
                          latency; short-lived scripts should either set
                          False or call the EvalExporter's flush() before
                          exiting so scores reach storage)
    guardrails=[...]    → enforce rules on inputs/outputs; may raise GuardrailError

    Guardrail pipeline order (ensures PII-free storage and auditable blocks):
      1. _MutatingGuardrailExporter  — PIIRedact etc., run before eval + storage
      2. EvalExporter                — eval_scores attached to span
      3. AlertExporter               — threshold alerts
      4. Storage exporters           — JSONL / SQLite / HTTP (span now includes scores)
      5. _BlockingGuardrailExporter  — HallucinationBlock etc., raise after storage
                                       so violations are always persisted

    tenant_id           → process-wide default tenant (B2B customer org).
                          Overridden per-request by peekr.session(tenant_id=...).
                          Falls back to env PEEKR_TENANT_ID.
    retention_class     → process-wide default retention tier.
                          Recommended: "default" | "short" | "long" | "pii".
                          Falls back to env PEEKR_RETENTION_CLASS.

    sample_rate         → fraction of root traces to persist, 0.0–1.0
                          (default 1.0 = keep everything). Decision is made
                          at root-span creation and inherited by all children,
                          so traces are never partially captured.
                          Evaluators and alerts still see every span — only
                          storage exporters respect sampling.
    keep_errors         → if True (default), errored spans are always
                          persisted even when their trace was sampled out.
    budget_alert        → BudgetAlert(limit_usd=10.0) → warn at $8 (80%),
                          raise BudgetExceededError at $10 (per day by default).
                          e.g. budget_alert=BudgetAlert(10.0)
    """
    global _patched

    # Map compliance=[] (explicit empty list) to the disabled sentinel.
    # compliance=None stays None → auto-discover.
    if compliance is not None and isinstance(compliance, list) and len(compliance) == 0:
        compliance = _COMPLIANCE_DISABLED

    set_process_defaults(
        tenant_id=tenant_id,
        retention_class=retention_class,
        sample_rate=sample_rate,
        keep_errors=keep_errors,
    )

    # Exporter pipeline. Order is load-bearing — see docstring above.
    # Evaluators, guardrails, and alerts are PRE-STORAGE processors —
    # they run regardless of which storage backend is chosen.
    if not exporter:
        # 1) Console — display only, no mutations.
        if console:
            add_exporter(ConsoleExporter())

    # 2) Mutating guardrails FIRST — redact PII before eval sees the text.
    #    Also register any input-blocking guardrails for pre-call checks.
    if guardrails:
        from .guard import _MutatingGuardrailExporter
        from .context import register_input_guard

        add_exporter(_MutatingGuardrailExporter(guardrails))
        for g in guardrails:
            if getattr(g, "_input_guard", False):
                register_input_guard(g)

    # 3) Evaluators — scores written to span.attributes["eval_scores"].
    if evaluators:
        from .eval import EvalExporter

        add_exporter(
            EvalExporter(evaluators, span_filter=evaluate_filter, async_eval=async_eval)
        )

    # 4) Alerts.
    if alerts:
        from .alerts import AlertExporter

        add_exporter(AlertExporter(alerts))

    # 4c) Budget alert — warn at 80% of limit, raise BudgetExceededError at limit.
    if budget_alert:
        add_exporter(budget_alert)

    # 4b) Cloud compliance guardrails — fetched from Peekr Cloud, enforced locally.
    #
    # Three modes:
    #   compliance not passed (default)  → auto-discover: fetch all packs enabled
    #                                       in the dashboard for this project
    #   compliance=["HIPAA", "FDCPA"]    → enforce exactly those packs
    #   compliance=[]                    → explicitly disabled, skip entirely
    #
    # "Auto" is the recommended default — the dashboard becomes the single
    # source of truth and no code change is needed when you toggle a pack.
    _should_add_compliance = (
        exporter is not None
        and compliance is not _COMPLIANCE_DISABLED
        and getattr(exporter, "api_key", None)
    )
    if _should_add_compliance:
        from .guard._cloud_compliance import CloudComplianceGuard

        _api_key = getattr(exporter, "api_key", None)
        _endpoint = getattr(exporter, "endpoint", "https://peekr.starkspherelabs.com")
        # compliance=None → auto (packs=None means "all enabled in dashboard")
        # compliance=[...] → explicit list passed through
        _packs = compliance if compliance is not None else None
        _cg = CloudComplianceGuard(
            api_key=_api_key,
            endpoint=_endpoint,
            packs=_packs,
        )
        if True:  # keep indent for the prepend line below
            guardrails = [_cg] + list(guardrails or [])

    # 5) Storage — span is fully annotated (redacted + scored) by now.
    # A None path disables that backend; JSONLExporter(None)/SQLiteExporter(None)
    # would otherwise raise on every span export.
    if exporter:
        add_exporter(exporter)
    else:
        if storage in ("jsonl", "both") and jsonl_path is not None:
            add_exporter(JSONLExporter(jsonl_path))
        if storage in ("sqlite", "both") and db_path is not None:
            add_exporter(SQLiteExporter(db_path))

    # 6) Blocking guardrails LAST — raise after storage so violations persist.
    if guardrails:
        from .guard import _BlockingGuardrailExporter

        add_exporter(_BlockingGuardrailExporter(guardrails))

    if not _patched:
        patch_openai()
        patch_anthropic()
        patch_bedrock()
        patch_gemini()
        patch_langchain()
        patch_llamaindex()
        patch_crewai()
        patch_litellm()
        patch_pinecone()
        patch_chroma()
        patch_qdrant()
        _patched = True


__all__ = [
    # core
    "instrument",
    "trace",
    "session",
    "start_span",
    "end_span",
    "get_current_span",
    # exporters
    "JSONLExporter",
    "ConsoleExporter",
    "SQLiteExporter",
    "HTTPExporter",
    "add_exporter",
    "clear_exporters",
    "OTelExporter",
    # features
    "feedback",
    "export_feedback",
    "experiment",
    "alert",
    "eval",
    "guard",
    "guard.GuardrailError",
    # middleware
    "FastAPIMiddleware",
    "PeekrASGIMiddleware",
    # harnesses
    "instrument_rag",
    # evidence ("why did the AI say that?")
    "record_evidence",
    "clear_evidence",
    "evidence",
    # budget
    "BudgetAlert",
    "BudgetExceededError",
    "budget_alert",
]
