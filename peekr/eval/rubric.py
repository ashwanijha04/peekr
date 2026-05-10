from __future__ import annotations

from typing import TYPE_CHECKING

from . import BaseEvaluator

if TYPE_CHECKING:
    from ..span import Span

# Optional LLM providers — imported at module level so tests can patch them.
try:
    import openai  # type: ignore
except ImportError:
    openai = None  # type: ignore

try:
    import anthropic  # type: ignore
except ImportError:
    anthropic = None  # type: ignore


class Rubric(BaseEvaluator):
    """Evaluator that uses an LLM to score output against a rubric criterion.

    Makes a real LLM call (openai if available, else anthropic).
    Prompt: score 0.0–1.0 based on the criterion; returns only a float.
    """

    def __init__(self, criteria: str) -> None:
        self.criteria = criteria

    @property
    def name(self) -> str:
        return f"Rubric({self.criteria[:30]})"

    def evaluate(self, span: Span) -> float:
        output = span.attributes.get("output", "")
        prompt = (
            f"Score this LLM response on a scale of 0.0 to 1.0 based on the criterion: "
            f"{self.criteria}. Response: {output}. Return only a float."
        )

        if openai is not None:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
            )
            text = response.choices[0].message.content or "0.0"
            return float(text.strip())

        if anthropic is not None:
            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text if response.content else "0.0"
            return float(text.strip())

        raise ImportError(
            "Rubric evaluator requires either 'openai' or 'anthropic' to be installed."
        )


class NotEmpty(BaseEvaluator):
    """Evaluator that returns 1.0 if the span output is a non-empty string, else 0.0."""

    def evaluate(self, span: Span) -> float:
        output = span.attributes.get("output", "")
        return 1.0 if isinstance(output, str) and output.strip() else 0.0


class NoError(BaseEvaluator):
    """Evaluator that returns 1.0 if span.status == 'ok', else 0.0."""

    def evaluate(self, span: Span) -> float:
        return 1.0 if span.status == "ok" else 0.0
