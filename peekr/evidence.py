"""Addressable evidence chunks — the substrate for "why did the AI say that?".

The RAG harness (``harnesses/rag.py``) stores retrieved docs as a single
truncated *blob* used only as grounding context for the faithfulness judge.
That answers "is the answer grounded?" but not "grounded in WHICH source?".

This module captures retrieval results as *individually addressable* chunks —
each with a stable id, a 1-based number ``n`` (the ``[N]`` the model cites),
an optional retrieval ``score`` and ``rank``, and source metadata. The
``Hallucination`` evaluator reads these and asks the judge to attribute each
atomic claim back to the specific chunk(s) that support or contradict it, so
the dashboard can show the evidence edge — not just a score.

Usage::

    import peekr

    hits = retriever.search(query)          # your ranked results
    peekr.record_evidence(hits)             # normalises + stores for this request
    answer = generate(query, hits)          # the LLM call (auto-instrumented)
    # On span export the Hallucination evaluator links each claim -> chunk.

``record_evidence`` accepts the common retrieval shapes (dicts with
``content``/``text``/``page_content``, LangChain Documents, LlamaIndex Nodes,
plain strings) and flexible score/id keys, so it works without a translation
layer in most pipelines.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

# Per-request addressable evidence. Set by record_evidence(); read by the
# Hallucination evaluator at span-export time. Follows the async task tree
# via ContextVar so it stays correct across await boundaries and is isolated
# between concurrent requests.
_evidence: ContextVar[list[dict]] = ContextVar("peekr_evidence", default=[])

# Keep stored chunk text bounded — enough to display + judge against, small
# enough to keep span payloads sane.
_TRUNCATE = 800


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _truncate(text: str) -> str:
    return text if len(text) <= _TRUNCATE else text[:_TRUNCATE] + "…"


def _chunk_text(item: Any) -> str:
    """Pull the text out of any common retrieval-result shape."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(
            item.get("text")
            or item.get("content")
            or item.get("page_content")
            or item.get("chunk")
            or item.get("body")
            or ""
        )
    # LangChain Document, LlamaIndex Node, custom object…
    return str(
        getattr(item, "page_content", None)
        or getattr(item, "text", None)
        or getattr(item, "content", None)
        or item
    )


def _get(item: Any, *keys: str) -> Any:
    """First present value across dict keys / object attributes."""
    for k in keys:
        if isinstance(item, dict):
            if item.get(k) is not None:
                return item[k]
        else:
            v = getattr(item, k, None)
            if v is not None:
                return v
    return None


def normalize_chunk(item: Any, n: int) -> dict:
    """Turn one retrieval result into a stable, addressable evidence chunk.

    ``n`` is the 1-based position — the number the model cites as ``[n]`` and
    the key the claim->chunk attribution refers to.
    """
    text = _truncate(_chunk_text(item).strip())
    return {
        "n": n,
        "id": str(_get(item, "id", "chunk_id", "doc_id", "uuid") or n),
        "text": text,
        "title": _get(item, "title", "name", "heading"),
        "source_kind": _get(item, "source_kind", "source", "kind", "type"),
        "source_url": _get(item, "source_url", "url", "uri", "link"),
        # Prefer a reranker score, then fusion/semantic scores.
        "score": _coerce_float(
            _get(item, "score", "rerank_score", "rrf_score", "relevance", "vec_similarity", "similarity")
        ),
        "rank": _get(item, "rank") or n,
    }


def record_evidence(chunks: Any) -> list[dict]:
    """Record the evidence the model was given, as addressable chunks.

    Accepts a list of retrieval results (dicts, Documents, Nodes, or strings)
    or a single item. Returns the normalised chunks (also stored for the
    current request so the Hallucination evaluator can attribute claims).
    """
    if chunks is None:
        normalized: list[dict] = []
    elif isinstance(chunks, (list, tuple)):
        normalized = [normalize_chunk(c, i) for i, c in enumerate(chunks, start=1)]
    else:
        normalized = [normalize_chunk(chunks, 1)]
    _evidence.set(normalized)
    return normalized


def get_evidence() -> list[dict]:
    """Return the evidence chunks recorded for the current request (or [])."""
    return _evidence.get()


def clear_evidence() -> None:
    """Drop recorded evidence (call at the end of a request to be tidy)."""
    _evidence.set([])


@contextmanager
def evidence(chunks: Any):
    """Scope evidence to a block, clearing it on exit::

        with peekr.evidence(hits):
            answer = generate(...)
    """
    token = _evidence.set(record_evidence(chunks))
    try:
        yield
    finally:
        _evidence.reset(token)


def format_for_judge(chunks: list[dict]) -> str:
    """Render chunks as a numbered list for the attribution judge prompt."""
    lines = []
    for c in chunks:
        head = f"[{c['n']}]"
        if c.get("title"):
            head += f" {c['title']} —"
        lines.append(f"{head} {c.get('text', '')}".strip())
    return "\n\n".join(lines)
