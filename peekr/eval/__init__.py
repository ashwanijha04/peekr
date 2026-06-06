from __future__ import annotations

import abc
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as _futures_wait
from contextvars import ContextVar
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from ..span import Span

# Guard against infinite recursion when an evaluator itself triggers an LLM span
_in_eval: ContextVar[bool] = ContextVar("_in_eval", default=False)

# Prefixes that identify LLM spans
_LLM_PREFIXES = ("openai.", "anthropic.", "bedrock.", "gemini.", "google.")


class BaseEvaluator(abc.ABC):
    """Abstract base class for all evaluators."""

    @abc.abstractmethod
    def evaluate(self, span: Span) -> float:
        """Score the span on a scale of 0.0 to 1.0."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__


class EvalExporter:
    """Exporter that runs evaluators on LLM spans and stores scores in span.attributes.

    By default evaluators run on a background thread so they add zero latency
    to the calling application. The span is exported immediately; scores are
    patched back to Peekr Cloud when the background job finishes.

    Parameters
    ----------
    evaluators
        The list of `BaseEvaluator` instances to run on each LLM span.
    span_filter
        Optional callable `(span) -> bool`. When set, evaluators only run on
        spans for which the filter returns truthy.
    async_eval
        If True (default), evaluations run on a background thread — zero
        latency to the caller. Set False to run synchronously, or call
        `flush()` before exit (useful in scripts that exit immediately
        after the LLM call).
    """

    def __init__(
        self,
        evaluators: list[BaseEvaluator],
        span_filter: Optional[Callable[["Span"], bool]] = None,
        async_eval: bool = True,
    ) -> None:
        self.evaluators = evaluators
        self.span_filter = span_filter
        self.async_eval = async_eval
        # Shared thread pool — evaluators are I/O bound (LLM calls)
        # so a larger pool is fine; 4 concurrent judge calls is plenty
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="peekr-eval")
        self._pending: set = set()  # in-flight background evals — see flush()
        self._is_eval_exporter = True  # lets storage exporters identify us

    def export(self, span: Span) -> None:
        attrs = span.attributes or {}

        if _in_eval.get():
            return
        if attrs.get("peekr.internal"):
            return

        output = attrs.get("output")
        has_output = isinstance(output, str) and bool(output.strip())
        is_llm = any(span.name.startswith(p) for p in _LLM_PREFIXES)
        if not (is_llm or has_output):
            return

        if self.span_filter is not None:
            try:
                if not self.span_filter(span):
                    return
            except Exception:
                return

        if self.async_eval:
            # Fire-and-forget: span is already exported by the storage exporter
            # (which runs before EvalExporter in the pipeline). The background
            # job scores the span and sends a PATCH to update it in Peekr Cloud.
            import copy

            span_copy = copy.copy(span)
            span_copy.attributes = dict(span.attributes or {})
            fut = self._pool.submit(self._run_eval_and_patch, span_copy)
            self._pending.add(fut)
            fut.add_done_callback(self._pending.discard)
        else:
            self._run_eval_sync(span)

    def flush(self, timeout: Optional[float] = None) -> None:
        """Block until queued background evaluations finish.

        Call before process exit in short-lived scripts (and in tests) so
        async scores reach storage. No-op when async_eval=False or when
        nothing is in flight.
        """
        pending = set(self._pending)
        if pending:
            _futures_wait(pending, timeout=timeout)

    def _run_eval_sync(self, span: Span) -> None:
        """Run all evaluators synchronously and write scores directly to span."""
        token = _in_eval.set(True)
        try:
            scores: dict[str, float] = {}
            errors: dict[str, str] = {}
            for evaluator in self.evaluators:
                try:
                    scores[evaluator.name] = evaluator.evaluate(span)
                except Exception as e:
                    errors[evaluator.name] = f"{type(e).__name__}: {e}"
            if scores:
                span.attributes.setdefault("eval_scores", {})
                span.attributes["eval_scores"].update(scores)
            if errors:
                span.attributes.setdefault("eval_errors", {})
                span.attributes["eval_errors"].update(errors)
        finally:
            _in_eval.reset(token)

    def _run_eval_and_patch(self, span: Span) -> None:
        """Background worker: run evaluators then patch scores back to the exporter."""
        token = _in_eval.set(True)
        try:
            scores: dict[str, float] = {}
            errors: dict[str, str] = {}
            for evaluator in self.evaluators:
                try:
                    scores[evaluator.name] = evaluator.evaluate(span)
                except Exception as e:
                    errors[evaluator.name] = f"{type(e).__name__}: {e}"

            if not scores and not errors:
                return

            # Patch scores back into the span and re-export to storage so
            # Peekr Cloud receives them. Only the storage exporters that
            # support PATCH (HTTPExporter) will update the existing span;
            # local exporters (JSONL/SQLite) will upsert on span_id.
            if scores:
                span.attributes.setdefault("eval_scores", {})
                span.attributes["eval_scores"].update(scores)
            if errors:
                span.attributes.setdefault("eval_errors", {})
                span.attributes["eval_errors"].update(errors)

            # Re-export span with scores now attached.
            # Use only storage exporters (JSONL, SQLite, HTTP) — skip
            # eval/alert exporters to avoid infinite recursion.
            # NB: ..exporters (package root), NOT .exporters — peekr.eval has
            # no exporters module, and the bare relative import made every
            # background eval die with ImportError inside its Future, silently
            # dropping the scores it had just computed.
            from ..exporters import _exporters

            for exporter in list(_exporters):
                if isinstance(exporter, EvalExporter):
                    continue
                if not getattr(exporter, "_is_storage", False):
                    continue
                try:
                    exporter.export(span)
                except Exception:
                    pass
        finally:
            _in_eval.reset(token)


# Import concrete evaluators so they're importable from peekr.eval
from .rubric import Rubric, NotEmpty, NoError  # noqa: E402, F401
from .hallucination import Hallucination  # noqa: E402, F401
from .citation import CitationAccuracy  # noqa: E402, F401

__all__ = [
    "BaseEvaluator",
    "EvalExporter",
    "_in_eval",
    "Rubric",
    "NotEmpty",
    "NoError",
    "Hallucination",
    "CitationAccuracy",
]
