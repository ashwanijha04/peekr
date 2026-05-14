from __future__ import annotations

import abc
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..span import Span

# Guard against infinite recursion when an evaluator itself triggers an LLM span
_in_eval: ContextVar[bool] = ContextVar("_in_eval", default=False)

# Prefixes that identify LLM spans
_LLM_PREFIXES = ("openai.", "anthropic.", "bedrock.")


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
    """Exporter that runs evaluators on LLM spans and stores scores in span.attributes."""

    def __init__(self, evaluators: list[BaseEvaluator]) -> None:
        self.evaluators = evaluators

    def export(self, span: Span) -> None:
        # Only evaluate LLM spans
        if not any(span.name.startswith(p) for p in _LLM_PREFIXES):
            return

        # Do not recurse into evaluation
        if _in_eval.get():
            return

        token = _in_eval.set(True)
        try:
            scores: dict[str, float] = {}
            for evaluator in self.evaluators:
                try:
                    score = evaluator.evaluate(span)
                    scores[evaluator.name] = score
                except Exception:
                    # Don't let a failing evaluator crash the exporter
                    scores[evaluator.name] = 0.0

            if scores:
                span.attributes.setdefault("eval_scores", {})
                span.attributes["eval_scores"].update(scores)
        finally:
            _in_eval.reset(token)


# Import concrete evaluators so they're importable from peekr.eval
from .rubric import Rubric, NotEmpty, NoError  # noqa: E402, F401
from .hallucination import Faithfulness, AnswerRelevance, ContextRelevance  # noqa: E402, F401

__all__ = [
    "BaseEvaluator",
    "EvalExporter",
    "_in_eval",
    "Rubric",
    "NotEmpty",
    "NoError",
    "Faithfulness",
    "AnswerRelevance",
    "ContextRelevance",
]
