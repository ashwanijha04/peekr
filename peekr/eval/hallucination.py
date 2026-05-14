"""Hallucination detection evaluators.

Three complementary scores, modelled on the Ragas / TruLens RAG triad:

  Faithfulness     — is the answer grounded in the retrieved context?
                     (claim-level entailment via LLM-as-judge)
  AnswerRelevance  — does the answer actually address the question?
  ContextRelevance — was the retrieved context relevant to the question?

Each evaluator works in two ways:

1. As a registered evaluator on every LLM call::

       peekr.instrument(evaluators=[
           peekr.eval.Faithfulness(),
           peekr.eval.AnswerRelevance(),
       ])

   It reads ``context`` and ``query`` from the span's attributes (you set
   these via ``peekr.set_grounding(context=..., query=...)`` before the LLM
   call) or from a ``peekr.session(grounding=...)`` block.

2. Programmatically — call ``.score(answer, context=..., query=...)`` to
   score a single string outside the trace pipeline::

       score = Faithfulness().score(answer, context=docs)
"""
from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Optional

from . import BaseEvaluator

if TYPE_CHECKING:
    from ..span import Span

try:
    import openai  # type: ignore
except ImportError:
    openai = None  # type: ignore

try:
    import anthropic  # type: ignore
except ImportError:
    anthropic = None  # type: ignore


_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
_DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


def _call_judge(prompt: str, max_tokens: int = 200) -> str:
    """Send a prompt to whichever judge model is available. Returns raw text."""
    if openai is not None:
        try:
            response = openai.chat.completions.create(
                model=_DEFAULT_OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception:
            pass
    if anthropic is not None:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=_DEFAULT_ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.content:
            return getattr(response.content[0], "text", "").strip()
        return ""
    raise ImportError(
        "Hallucination evaluators need either 'openai' or 'anthropic' installed."
    )


_FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_score(text: str) -> float:
    """Pull the first float out of an LLM response and clamp to [0, 1]."""
    match = _FLOAT_RE.search(text)
    if not match:
        return 0.0
    value = float(match.group(0))
    if value > 1.0:
        value = value / 100.0 if value <= 100 else 1.0
    return max(0.0, min(1.0, value))


def _grounding_from_span(span: "Span", key: str) -> Optional[str]:
    """Pull grounding info (context / query) from span attributes or session state."""
    attrs = span.attributes
    value = attrs.get(key) or attrs.get(f"grounding.{key}")
    if value is None:
        try:
            from ..session import get_grounding
            value = get_grounding(key)
        except ImportError:
            value = None
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return "\n\n".join(str(v) for v in value)
    return str(value)


class Faithfulness(BaseEvaluator):
    """Score: are the answer's claims supported by the retrieved context?

    Uses an LLM judge that breaks the answer into claims and checks each
    against the context. Returns the fraction of claims that are supported,
    so an answer with one hallucinated bullet out of five gets 0.8.

    Requires ``context`` to be available on the span (or via
    ``peekr.session(grounding={"context": ...})``).
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self._custom_name = name

    @property
    def name(self) -> str:
        return self._custom_name or "Faithfulness"

    def evaluate(self, span: "Span") -> float:
        answer = span.attributes.get("output", "")
        context = _grounding_from_span(span, "context")
        if not answer or not context:
            return 0.0
        return self.score(answer, context=context)

    def score(self, answer: str, *, context: str) -> float:
        prompt = (
            "You are a fact-checking judge. Given a CONTEXT and an ANSWER, "
            "extract every factual claim made in the ANSWER, then decide whether "
            "each claim is directly supported by the CONTEXT.\n\n"
            "Reply with JSON only, of the form:\n"
            '{"claims":[{"claim":"…","supported":true|false}, …]}\n'
            "If the answer makes no factual claims (e.g. a clarifying question), "
            'reply: {"claims":[]}.\n\n'
            f"CONTEXT:\n{context}\n\n"
            f"ANSWER:\n{answer}\n"
        )
        text = _call_judge(prompt, max_tokens=600)
        try:
            data = json.loads(_strip_code_fence(text))
        except Exception:
            return _parse_score(text)
        if not isinstance(data, dict):
            return _parse_score(text)
        claims = data.get("claims") or []
        if not claims:
            return 1.0
        supported = sum(1 for c in claims if c.get("supported"))
        return supported / len(claims)


class AnswerRelevance(BaseEvaluator):
    """Score: does the answer address the original question?

    Cheap, single-shot LLM judge. Requires ``query`` to be set on the span
    or via the session grounding.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self._custom_name = name

    @property
    def name(self) -> str:
        return self._custom_name or "AnswerRelevance"

    def evaluate(self, span: "Span") -> float:
        answer = span.attributes.get("output", "")
        query = _grounding_from_span(span, "query")
        if not answer or not query:
            return 0.0
        return self.score(answer, query=query)

    def score(self, answer: str, *, query: str) -> float:
        prompt = (
            "Score how directly the ANSWER addresses the QUESTION on a 0.0 – 1.0 "
            "scale. 1.0 = the answer fully and specifically addresses the question. "
            "0.0 = the answer is off-topic or evasive. Reply with only the number.\n\n"
            f"QUESTION:\n{query}\n\nANSWER:\n{answer}\n"
        )
        return _parse_score(_call_judge(prompt, max_tokens=10))


class ContextRelevance(BaseEvaluator):
    """Score: is the retrieved context relevant to the question?

    Diagnoses RAG retrieval problems — a low score here with a high
    Faithfulness score means the retriever pulled tangential docs but the
    LLM did the right thing with what it had.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self._custom_name = name

    @property
    def name(self) -> str:
        return self._custom_name or "ContextRelevance"

    def evaluate(self, span: "Span") -> float:
        context = _grounding_from_span(span, "context")
        query = _grounding_from_span(span, "query")
        if not context or not query:
            return 0.0
        return self.score(context=context, query=query)

    def score(self, *, context: str, query: str) -> float:
        prompt = (
            "Score how relevant the CONTEXT is to the QUESTION on a 0.0 – 1.0 "
            "scale. 1.0 = context directly answers the question. 0.0 = context "
            "is unrelated. Reply with only the number.\n\n"
            f"QUESTION:\n{query}\n\nCONTEXT:\n{context}\n"
        )
        return _parse_score(_call_judge(prompt, max_tokens=10))


def _strip_code_fence(text: str) -> str:
    """Strip ```json ... ``` fences that judge models sometimes return."""
    stripped = text.strip()
    if stripped.startswith("```"):
        # remove opening fence + optional language tag
        stripped = re.sub(r"^```[A-Za-z0-9_-]*\n?", "", stripped)
        if stripped.endswith("```"):
            stripped = stripped[: -3]
    return stripped.strip()
