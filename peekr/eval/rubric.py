from __future__ import annotations

from typing import TYPE_CHECKING

from . import BaseEvaluator

if TYPE_CHECKING:
    from ..span import Span


class Rubric(BaseEvaluator):
    """Evaluator that uses an LLM to score output against a rubric criterion.

    Provider selection is delegated to ``peekr.eval._judge.call_judge`` so
    it picks whichever SDK has *credentials* (env var), not just whichever
    SDK is importable. See ``_judge.py`` for the full algorithm.

    Parameters
    ----------
    criteria
        Plain-English description of what "good" looks like. Goes verbatim
        into the prompt.
    judge_provider
        ``"auto"`` (default) picks based on env credentials. Pass
        ``"openai"`` or ``"anthropic"`` to force a specific provider.
    model
        Override the judge model.
    """

    def __init__(
        self,
        criteria: str,
        judge_provider: str = "auto",
        model: str | None = None,
    ) -> None:
        self.criteria = criteria
        self.judge_provider = judge_provider
        self.model = model

    @property
    def name(self) -> str:
        return f"Rubric({self.criteria[:30]})"

    def evaluate(self, span: "Span") -> float:
        from ._judge import call_judge  # local import to avoid cycles

        output = span.attributes.get("output", "")
        prompt = (
            f"Score this LLM response on a scale of 0.0 to 1.0 based on the criterion: "
            f"{self.criteria}. Response: {output}. Return only a float."
        )
        text = call_judge(
            prompt,
            max_tokens=10,
            model=self.model,
            provider=self.judge_provider,
            fallback="0.0",
        )
        return float(text.strip())


class NotEmpty(BaseEvaluator):
    """Evaluator that returns 1.0 if the span output is a non-empty string, else 0.0."""

    def evaluate(self, span: Span) -> float:
        output = span.attributes.get("output", "")
        return 1.0 if isinstance(output, str) and output.strip() else 0.0


class NoError(BaseEvaluator):
    """Evaluator that returns 1.0 if span.status == 'ok', else 0.0."""

    def evaluate(self, span: Span) -> float:
        return 1.0 if span.status == "ok" else 0.0
