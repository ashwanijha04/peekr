"""Citation / reference accuracy evaluator.

A complementary signal to the LLM-as-judge `Hallucination` evaluator: this
one runs *purely on string patterns* — zero LLM calls — and is therefore
free, fast, and deterministic. It catches the specific case where an LLM
invents URLs, arXiv IDs, paper titles, statute numbers, or "Author et al.
(YEAR)" citations that aren't present in the source context.

Score:
    1.0 → every citation pattern in the output appears in the context
    0.0 → no citation pattern in the output is grounded
    1.0 → no citation patterns at all (nothing to verify)

A low score is a strong "look here" signal: paired with the `Hallucination`
evaluator on the dashboard, it pinpoints the *kind* of hallucination
(invented citations) vs general fabrication.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Callable

from . import BaseEvaluator

if TYPE_CHECKING:
    from ..span import Span


# ---------------------------------------------------------------------------
# Pattern extractors
# ---------------------------------------------------------------------------
# Each pattern returns canonical strings used for context-membership lookup.
# We keep them deliberately loose: false-positive "I think I found a citation"
# is much better than false-negative "I missed an invented one" for a quality
# signal — we are flagging, not blocking.
# ---------------------------------------------------------------------------

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("url", re.compile(r"https?://[^\s)>\]]+")),
    ("arxiv", re.compile(r"\barXiv\s*:\s*\d{4}\.\d{4,5}\b", re.IGNORECASE)),
    ("doi", re.compile(r"\b10\.\d{4,9}/[\w\-./;:()]+\b")),
    # "Smith et al., 2018" or "Smith et al. (2018)"
    (
        "author_year",
        re.compile(
            r"\b[A-Z][a-z]+(?:-[A-Z][a-z]+)?\s+et\s+al\.?,?\s*\(?\s*(19|20)\d{2}\s*\)?"
        ),
    ),
    # Statute / section references — "Section 230", "§ 7", "Title VII".
    # No \b on the § alternative — § is non-word so a word boundary won't match it.
    ("section", re.compile(r"(?:\bSection\s*|§\s*)\d+[A-Za-z]?\b")),
    # Paper / book titles in quotes — REQUIRE a citation-shaped preamble like
    # "in 'X'", "titled 'X'", "from 'X'", "see 'X'", "cited in 'X'", "per 'X'".
    # Without this guard, any product name or claim string in quoted output
    # (e.g. tool-use payloads, packing lists) matches and produces noisy
    # "invented citation" flags. Reported by users running peekr on real RAG.
    (
        "quoted_title",
        re.compile(
            r"(?:\b(?:in|from|titled|named|called|see|cited\s+in|cited\s+by|per|according\s+to|read|wrote|published\s+in)\s+)"
            r"['\"“‘]([A-Z][^'\"”’]{8,120})['\"”’]",
            re.IGNORECASE,
        ),
    ),
)

# Outputs that *look like* structured tool calls or raw data dumps rather
# than free-text answers. Citation extraction (and LLM-judge hallucination
# scoring) is meaningless on these — we skip the evaluator instead of
# emitting confident-but-wrong scores.
_TOOL_CALL_PREFIXES: tuple[str, ...] = (
    "ToolUseBlock",
    "ToolResultBlock",
    "TextBlock(",
    "{",
    "[",
)


def looks_like_tool_call(output: str) -> bool:
    """Heuristic: True if `output` reads as a tool-call repr or JSON payload."""
    if not isinstance(output, str):
        return False
    stripped = output.lstrip()
    return stripped.startswith(_TOOL_CALL_PREFIXES)


def _normalize(s: str) -> str:
    """Loose normalization so 'Section  230' matches 'Section 230' and
    'arXiv: 1810.04805' matches 'arXiv:1810.04805' for dedup + membership checks."""
    s = re.sub(r"\s+", " ", s).strip().lower()
    # Strip whitespace immediately after a colon — keeps "arxiv:" and "doi:" canonical.
    s = re.sub(r":\s+", ":", s)
    return s


def extract_citations(text: str) -> list[dict[str, str]]:
    """Return [{'kind': ..., 'text': <raw match>}] for every citation pattern found."""
    if not text:
        return []
    out: list[dict[str, str]] = []
    for kind, pat in _PATTERNS:
        for m in pat.finditer(text):
            out.append({"kind": kind, "text": m.group(0).strip()})
    # Dedupe preserving order
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, str]] = []
    for c in out:
        key = (c["kind"], _normalize(c["text"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    return deduped


def _is_grounded(citation: dict[str, str], context_norm: str) -> bool:
    """A citation is grounded if its normalized text appears in the normalized context."""
    return _normalize(citation["text"]) in context_norm


class CitationAccuracy(BaseEvaluator):
    """Fraction of citation patterns in the output that appear in the context.

    Pure-Python, no LLM calls. Pair with `Hallucination` to distinguish
    "invented citations" from "wrong facts".

    Parameters
    ----------
    context_extractor : callable, optional
        Pulls the grounding context out of the span. Defaults to
        `span.attributes["input"]` — same convention as `Hallucination`.
    """

    def __init__(
        self,
        context_extractor: Callable[["Span"], str] | None = None,
    ) -> None:
        self.context_extractor = context_extractor

    @property
    def name(self) -> str:
        return "CitationAccuracy"

    def evaluate(self, span: "Span") -> float:
        output = span.attributes.get("output", "")
        if not isinstance(output, str) or not output.strip():
            return 1.0

        # Skip tool-call / structured outputs — citation extraction is built
        # for free-text prose and produces false positives on ToolUseBlock or
        # raw JSON payloads (each Capitalized quoted value would look like
        # an invented title).
        if looks_like_tool_call(output):
            return 1.0

        if self.context_extractor is not None:
            context = self.context_extractor(span)
        else:
            context = span.attributes.get("input", "")
        if not isinstance(context, str):
            context = ""

        citations = extract_citations(output)
        if not citations:
            # Nothing to verify — neutral 1.0 (consistent with Hallucination()).
            return 1.0

        context_norm = _normalize(context)
        grounded = [c for c in citations if _is_grounded(c, context_norm)]
        invented = [c for c in citations if not _is_grounded(c, context_norm)]

        # Record per-citation detail on the span so dashboards can drill in.
        span.attributes["citation_details"] = {
            "total": len(citations),
            "grounded": len(grounded),
            "invented": len(invented),
            "items": [
                {**c, "grounded": _is_grounded(c, context_norm)} for c in citations
            ],
        }
        return len(grounded) / len(citations)
