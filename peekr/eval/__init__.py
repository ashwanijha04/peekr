from __future__ import annotations

import abc
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

    Parameters
    ----------
    evaluators
        The list of `BaseEvaluator` instances to run on each LLM span.
    span_filter
        Optional callable `(span) -> bool`. When set, evaluators only run on
        spans for which the filter returns truthy. Useful to scope the LLM-as-
        judge cost to user-facing calls only, e.g. by checking the endpoint:

            instrument(evaluators=[...],
                       evaluate_filter=lambda s: s.attributes.get("endpoint") == "/api/qa")
    """

    def __init__(
        self,
        evaluators: list[BaseEvaluator],
        span_filter: Optional[Callable[["Span"], bool]] = None,
    ) -> None:
        self.evaluators = evaluators
        self.span_filter = span_filter

    def export(self, span: Span) -> None:
        attrs = span.attributes or {}

        # Do not recurse into evaluation
        if _in_eval.get():
            return

        # Skip spans explicitly marked as internal (e.g. evaluator judge calls
        # tagged by the SDK patches via _in_eval). Belt-and-braces with the
        # ContextVar guard above.
        if attrs.get("peekr.internal"):
            return

        # Evaluate a span if it declares an LLM origin (legacy provider
        # prefixes — keeps evaluators like NotEmpty/NoError that must inspect a
        # span *regardless* of output working) OR if it produced a free-text
        # `output`. The output branch is what makes coverage provider-agnostic:
        # it picks up Gemini (`gemini.*`) and custom-named generation spans
        # (e.g. `rag.answer`) that the prefix list alone silently dropped —
        # which is why Gemini-primary workloads were going unscored.
        output = attrs.get("output")
        has_output = isinstance(output, str) and bool(output.strip())
        is_llm = any(span.name.startswith(p) for p in _LLM_PREFIXES)
        if not (is_llm or has_output):
            return

        # Per-span filter — lets users scope evaluation to a subset of spans
        # without re-implementing the exporter. See `span_filter` in
        # `EvalExporter.__init__`.
        if self.span_filter is not None:
            try:
                if not self.span_filter(span):
                    return
            except Exception:
                # A faulty filter should never break tracing.
                return

        token = _in_eval.set(True)
        try:
            scores: dict[str, float] = {}
            errors: dict[str, str] = {}
            for evaluator in self.evaluators:
                try:
                    score = evaluator.evaluate(span)
                    scores[evaluator.name] = score
                except Exception as e:
                    # Distinguish "evaluator crashed" from "evaluator returned 0.0".
                    # Storing 0.0 on crash made every span show as fully
                    # hallucinated when the judge LLM was unreachable (no API
                    # key, rate limit, network) — reported by users running
                    # peekr against an Anthropic-only workload while having
                    # `openai` installed as a transitive dep.
                    #
                    # Instead, record the error on a separate attribute so
                    # downstream tooling (dashboard, alerts, queries) can
                    # surface "judge unavailable" vs "score is genuinely 0.0".
                    errors[evaluator.name] = f"{type(e).__name__}: {e}"

            if scores:
                span.attributes.setdefault("eval_scores", {})
                span.attributes["eval_scores"].update(scores)
            if errors:
                span.attributes.setdefault("eval_errors", {})
                span.attributes["eval_errors"].update(errors)
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
