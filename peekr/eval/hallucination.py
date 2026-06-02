from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Callable

from . import BaseEvaluator

if TYPE_CHECKING:
    from ..span import Span

_FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?")
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_VALID_VERDICTS = ("supported", "contradicted", "unsupported")


def _parse_score(text: str) -> float:
    match = _FLOAT_RE.search(text or "")
    if not match:
        raise ValueError(f"Could not parse score from judge output: {text!r}")
    value = float(match.group(0))
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def _parse_claims(text: str) -> list[dict[str, str]]:
    """Pull a {"claims": [{"text": ..., "verdict": ...}, ...]} blob out of judge output."""
    if not text:
        raise ValueError("Empty judge output")
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        raise ValueError(f"No JSON object in judge output: {text!r}")
    blob = json.loads(match.group(0))
    raw = blob.get("claims", [])
    if not isinstance(raw, list):
        raise ValueError(f"'claims' is not a list: {raw!r}")
    cleaned: list[dict[str, str]] = []
    for c in raw:
        if not isinstance(c, dict):
            continue
        claim_text = str(c.get("text", "")).strip()
        verdict = str(c.get("verdict", "")).strip().lower()
        if not claim_text:
            continue
        if verdict not in _VALID_VERDICTS:
            verdict = "unsupported"
        cleaned.append({"text": claim_text, "verdict": verdict})
    return cleaned


_SIMPLE_PROMPT = (
    "You are a strict fact-checker. Given a CONTEXT and an OUTPUT, decide "
    "what fraction of the factual claims in OUTPUT are directly supported "
    "by CONTEXT. Score from 0.0 (no claim supported / fully hallucinated) "
    "to 1.0 (every claim supported). Unsupported claims include invented "
    "names, numbers, dates, citations, or facts not present in CONTEXT. "
    "Return ONLY the float — no words, no explanation.\n\n"
    "CONTEXT:\n{context}\n\nOUTPUT:\n{output}"
)

_DETAILED_PROMPT = (
    "You are a strict fact-checker following the RAGAS Faithfulness method.\n"
    "Step 1 — CLAIM DECOMPOSITION: split OUTPUT into atomic factual claims. "
    "Each claim should assert a single fact (one subject, one predicate, one "
    "object). Do NOT include opinions, questions, or restatements of CONTEXT.\n"
    "Step 2 — VERDICT: for each claim assign exactly one verdict:\n"
    "  - supported     — directly entailed by CONTEXT\n"
    "  - contradicted  — directly conflicts with CONTEXT\n"
    "  - unsupported   — neither entailed nor contradicted (CONTEXT is silent)\n"
    "Return ONLY a JSON object in this exact shape — no prose:\n"
    '  {{"claims": [{{"text": "<claim>", "verdict": "<verdict>"}}, ...]}}\n'
    "If OUTPUT has no factual claims, return {{\"claims\": []}}.\n\n"
    "CONTEXT:\n{context}\n\nOUTPUT:\n{output}"
)

_ATTRIBUTION_PROMPT = (
    "You are a strict fact-checker performing SOURCE ATTRIBUTION (RAGAS "
    "Faithfulness with grounding). CONTEXT is a NUMBERED list of evidence "
    "chunks the model was given.\n"
    "Step 1 — CLAIM DECOMPOSITION: split OUTPUT into atomic factual claims "
    "(one subject, one predicate, one object). Ignore opinions, questions, and "
    "restatements of the question.\n"
    "Step 2 — ATTRIBUTION: for EACH claim assign exactly one verdict and the "
    "responsible chunk numbers:\n"
    "  - supported    — directly entailed by one or more chunks; put them in 'support'\n"
    "  - contradicted — a chunk directly conflicts with the claim; put them in 'contradict'\n"
    "  - unsupported  — no chunk addresses the claim ('support' and 'contradict' both empty)\n"
    "Use ONLY chunk numbers that appear in CONTEXT. Return ONLY this JSON, no prose:\n"
    '  {{"claims": [{{"text": "<claim>", "verdict": "<verdict>", '
    '"support": [<int>, ...], "contradict": [<int>, ...]}}]}}\n'
    "If OUTPUT has no factual claims, return {{\"claims\": []}}.\n\n"
    "CONTEXT:\n{context}\n\nOUTPUT:\n{output}"
)


def _blame_for(verdict: str, support: list[int], contradict: list[int]) -> str:
    """Route blame for a claim to the layer an engineer should fix.

    - ``grounded``       — a chunk supports it; the answer is doing its job.
    - ``contradicted``   — a retrieved chunk disagrees with the claim: the
                           model contradicted its own source (a grounding /
                           generation failure, not a retrieval one).
    - ``not_in_context`` — nothing retrieved backs the claim: either the
                           answer-bearing source was never retrieved (a
                           retrieval miss) or the model fabricated it. We
                           cannot tell those apart from the trace alone, so we
                           label honestly rather than overclaim "retrieval miss".
    """
    if verdict == "contradicted" or contradict:
        return "contradicted"
    if verdict == "supported" and support:
        return "grounded"
    return "not_in_context"


def _parse_attributed_claims(text: str, n_chunks: int) -> list[dict]:
    """Parse the attribution judge output into claims with chunk references."""
    if not text:
        raise ValueError("Empty judge output")
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        raise ValueError(f"No JSON object in judge output: {text!r}")
    blob = json.loads(match.group(0))
    raw = blob.get("claims", [])
    if not isinstance(raw, list):
        raise ValueError(f"'claims' is not a list: {raw!r}")

    def _refs(value) -> list[int]:
        out: list[int] = []
        if isinstance(value, list):
            for v in value:
                try:
                    iv = int(v)
                except (TypeError, ValueError):
                    continue
                if 1 <= iv <= n_chunks and iv not in out:
                    out.append(iv)
        return out

    cleaned: list[dict] = []
    for c in raw:
        if not isinstance(c, dict):
            continue
        claim_text = str(c.get("text", "")).strip()
        if not claim_text:
            continue
        verdict = str(c.get("verdict", "")).strip().lower()
        if verdict not in _VALID_VERDICTS:
            verdict = "unsupported"
        support = _refs(c.get("support"))
        contradict = _refs(c.get("contradict"))
        # Reconcile verdict with the attribution so the blame tag is consistent
        # even when the judge labels loosely.
        if contradict and verdict != "contradicted":
            verdict = "contradicted"
        elif support and verdict == "unsupported":
            verdict = "supported"
        cleaned.append({
            "text": claim_text,
            "verdict": verdict,
            "support": support,
            "contradict": contradict,
            "blame": _blame_for(verdict, support, contradict),
        })
    return cleaned


class Hallucination(BaseEvaluator):
    """LLM-as-judge evaluator that scores how well an output is grounded in its context.

    Two modes:

    * ``detailed=False`` (default) — single-shot float score in [0, 1]. Cheap.
    * ``detailed=True`` — RAGAS-style faithfulness: the judge decomposes the
      output into atomic claims, assigns each a verdict
      (``supported`` / ``contradicted`` / ``unsupported``), and the score is
      ``supported_count / total_claims``. The full breakdown is written to
      ``span.attributes["hallucination_details"]`` so a dashboard can show
      *which* claims failed.

    Score interpretation in both modes:
        1.0 → every factual claim is supported by the context
        0.0 → no claim is supported (fully hallucinated / contradicted)

    Spans with no output or no context return 1.0 — there's nothing to judge.
    """

    def __init__(
        self,
        context_extractor: Callable[["Span"], str] | None = None,
        model: str | None = None,
        detailed: bool = False,
        judge_provider: str = "auto",
    ) -> None:
        """
        Parameters
        ----------
        context_extractor
            Optional callable that pulls the grounding context out of a span.
            Defaults to ``span.attributes["input"]``.
        model
            Override the judge model. Defaults to ``gpt-4o-mini`` (OpenAI) or
            ``claude-haiku-4-5-20251001`` (Anthropic).
        detailed
            ``True`` runs the RAGAS-style claim decomposition; default ``False``
            returns a single 0-1 score.
        judge_provider
            ``"auto"`` (default) picks whichever SDK has credentials in env.
            ``"openai"`` / ``"anthropic"`` forces a specific provider — useful
            when both SDKs are installed but only one is configured.
        """
        self.context_extractor = context_extractor
        self.model = model
        self.detailed = detailed
        self.judge_provider = judge_provider

    @property
    def name(self) -> str:
        return "Hallucination"

    def evaluate(self, span: "Span") -> float:
        output = span.attributes.get("output", "")
        if not isinstance(output, str) or not output.strip():
            return 1.0

        # Structured / tool-call outputs aren't free-text answers — the
        # LLM judge is not designed for them. ToolUseBlock(...) is the
        # Anthropic SDK's repr; raw JSON / Python literals show up when
        # users dump tool I/O. Treat these as not-evaluable (score 1.0)
        # rather than paying the judge to misgrade them as 0.0.
        from .citation import looks_like_tool_call  # local import to avoid cycle
        if looks_like_tool_call(output):
            span.attributes.setdefault(
                "hallucination_details",
                {
                    "claims": [],
                    "supported": 0,
                    "contradicted": 0,
                    "unsupported": 0,
                    "total": 0,
                    "score": 1.0,
                    "reason": "tool call, not a generation",
                },
            )
            return 1.0

        # Evidence-based attribution — if addressable chunks were recorded for
        # this request via peekr.record_evidence(), link each atomic claim to
        # the specific chunk(s) that support or contradict it. This is the
        # "why did the AI say that?" path; it supersedes the blob-context
        # faithfulness score because it produces the evidence edge, not just a
        # number.
        from ..evidence import get_evidence
        chunks = get_evidence()
        if chunks:
            return self._evaluate_with_attribution(chunks, output, span)

        if self.context_extractor is not None:
            context = self.context_extractor(span)
        else:
            context = span.attributes.get("input", "")

        if not isinstance(context, str) or not context.strip():
            return 1.0

        if self.detailed:
            return self._evaluate_detailed(context, output, span)
        return self._evaluate_simple(context, output)

    # ------------------------------------------------------------------
    # Simple mode — one float
    # ------------------------------------------------------------------

    def _evaluate_simple(self, context: str, output: str) -> float:
        prompt = _SIMPLE_PROMPT.format(context=context, output=output)
        text = self._judge(prompt, max_tokens=10, fallback="1.0")
        return _parse_score(text)

    # ------------------------------------------------------------------
    # Detailed mode — RAGAS-style claim decomposition + verdicts
    # ------------------------------------------------------------------

    def _evaluate_detailed(
        self,
        context: str,
        output: str,
        span: "Span",
    ) -> float:
        prompt = _DETAILED_PROMPT.format(context=context, output=output)
        text = self._judge(prompt, max_tokens=800, fallback='{"claims": []}')
        claims = _parse_claims(text)

        counts = {v: 0 for v in _VALID_VERDICTS}
        for c in claims:
            counts[c["verdict"]] += 1
        total = len(claims)
        score = (counts["supported"] / total) if total > 0 else 1.0

        span.attributes["hallucination_details"] = {
            "claims": claims,
            "supported": counts["supported"],
            "contradicted": counts["contradicted"],
            "unsupported": counts["unsupported"],
            "total": total,
            "score": score,
        }
        return score

    # ------------------------------------------------------------------
    # Attribution mode — claim → evidence-chunk edges ("why did it say that?")
    # ------------------------------------------------------------------

    def _evaluate_with_attribution(
        self,
        chunks: list[dict],
        output: str,
        span: "Span",
    ) -> float:
        from ..evidence import format_for_judge

        # Always persist the evidence chunks, even if the judge later fails —
        # the dashboard can show "what the model was given" regardless.
        span.attributes["evidence_chunks"] = chunks

        context = format_for_judge(chunks)
        prompt = _ATTRIBUTION_PROMPT.format(context=context, output=output)
        text = self._judge(prompt, max_tokens=1500, fallback='{"claims": []}')
        claims = _parse_attributed_claims(text, len(chunks))

        counts = {v: 0 for v in _VALID_VERDICTS}
        for c in claims:
            counts[c["verdict"]] += 1
        total = len(claims)
        score = (counts["supported"] / total) if total > 0 else 1.0

        span.attributes["hallucination_details"] = {
            "claims": claims,
            "supported": counts["supported"],
            "contradicted": counts["contradicted"],
            "unsupported": counts["unsupported"],
            "total": total,
            "score": score,
            "attributed": True,
        }
        return score

    # ------------------------------------------------------------------
    # Provider routing — delegated to _judge.call_judge so the selection
    # logic (env-key-aware, with explicit override) lives in one place.
    # See peekr/eval/_judge.py for the algorithm.
    # ------------------------------------------------------------------

    def _judge(self, prompt: str, max_tokens: int, fallback: str) -> str:
        from ._judge import call_judge
        return call_judge(
            prompt,
            max_tokens=max_tokens,
            model=self.model,
            provider=self.judge_provider,
            fallback=fallback,
        )


def _claim_summary(details: dict[str, Any]) -> str:
    """Human-readable one-liner for hallucination_details (used by the dashboard)."""
    total = details.get("total", 0)
    if not total:
        return "no factual claims"
    parts = []
    for v in _VALID_VERDICTS:
        n = details.get(v, 0)
        if n:
            parts.append(f"{n} {v}")
    return f"{total} claims · " + ", ".join(parts)
