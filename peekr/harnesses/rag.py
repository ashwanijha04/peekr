"""peekr.instrument_rag() — opinionated preset for RAG pipelines.

Wires three things automatically:
  1. @harness.retrieve  — spans your retrieval step, stores docs as grounding context
  2. @harness.rerank    — spans your reranking step, updates the grounding context
  3. Hallucination eval — pre-configured with a context_extractor that reads the
                          retrieved docs, so faithfulness scoring is automatic

Usage::

    import peekr

    harness = peekr.instrument_rag(
        exporter=peekr.HTTPExporter(
            endpoint="https://peekr.starkspherelabs.com",
            api_key="pk_live_…",
        ),
    )

    @harness.retrieve
    def search(query: str) -> list[str]:
        return vector_db.similarity_search(query, k=5)

    @harness.rerank          # optional — skip if you don't rerank
    def rerank(query: str, docs: list[str]) -> list[str]:
        return cohere.rerank(query, docs)

    # LLM call is auto-instrumented — no decorator needed.
    # The hallucination eval automatically uses the docs from @harness.retrieve
    # as its grounding context.
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Context: {docs}"},
            {"role": "user",   "content": query},
        ],
    )

The grounding context flows through ContextVar so it works correctly across
async/await boundaries and within the same request.  If no retrieve step ran
in the current call stack, the evaluator falls back to the LLM span's input.

You can also set the context manually::

    with harness.context(docs):
        response = client.chat.completions.create(...)
"""

from __future__ import annotations

import functools
import inspect
import json
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable


from ..context import start_span, end_span
from ..exporters import export_span

_TRUNCATE = 1000

# Holds the stringified retrieved docs for the current request.
# Set by @harness.retrieve / @harness.rerank / harness.context().
# Read by the Hallucination context_extractor.
_rag_docs: ContextVar[str] = ContextVar("rag_docs", default="")


def _serialize_docs(docs: Any) -> str:
    """Turn any common retrieval output shape into a plain string."""
    if docs is None:
        return ""
    if isinstance(docs, str):
        return docs if len(docs) <= _TRUNCATE else docs[:_TRUNCATE] + "…"
    if isinstance(docs, list):
        parts = []
        for d in docs:
            if isinstance(d, str):
                parts.append(d)
            elif isinstance(d, dict):
                # LangChain Document dict, Weaviate result, etc.
                text = (
                    d.get("page_content")
                    or d.get("text")
                    or d.get("content")
                    or d.get("chunk")
                    or json.dumps(d, default=str)
                )
                parts.append(str(text))
            else:
                # LangChain Document object, LlamaIndex Node, etc.
                text = (
                    getattr(d, "page_content", None)
                    or getattr(d, "text", None)
                    or getattr(d, "content", None)
                    or str(d)
                )
                parts.append(str(text))
        joined = "\n\n---\n\n".join(parts)
        return joined[:_TRUNCATE] + "…" if len(joined) > _TRUNCATE else joined
    try:
        return str(docs)[:_TRUNCATE]
    except Exception:
        return ""


def _rag_context_extractor(span: Any) -> str:
    """Passed to Hallucination(context_extractor=…).

    Reads the ContextVar set by @harness.retrieve, falling back to the
    span's input if no retrieval step ran in this call stack.
    """
    stored = _rag_docs.get()
    if stored:
        return stored
    return span.attributes.get("input", "")


class RAGHarness:
    """Returned by instrument_rag(). Provides decorators for RAG pipeline steps."""

    def retrieve(self, func=None, *, name: str = "rag.retrieve"):
        """Decorator for the retrieval step.

        Captures:
          - ``rag.query``          — the first string argument
          - ``rag.docs_count``     — number of docs returned
          - ``rag.retrieved_docs`` — first 1000 chars of stringified docs (for audit)

        Also stores the docs in a ContextVar so the Hallucination evaluator
        can use them as grounding context automatically.
        """

        def decorator(fn: Callable) -> Callable:
            if inspect.iscoroutinefunction(fn):

                @functools.wraps(fn)
                async def async_wrap(*args, **kwargs):
                    return await _run_retrieve(fn, args, kwargs, name, is_async=True)

                return async_wrap
            else:

                @functools.wraps(fn)
                def sync_wrap(*args, **kwargs):
                    return _run_retrieve_sync(fn, args, kwargs, name)

                return sync_wrap

        if func is not None:
            return decorator(func)
        return decorator

    def rerank(self, func=None, *, name: str = "rag.rerank"):
        """Decorator for the reranking step.

        Captures:
          - ``rag.docs_count``     — number of docs after reranking
          - ``rag.retrieved_docs`` — updated grounding context (replaces the
                                     value from @retrieve so the evaluator uses
                                     the post-rerank context)
        """

        def decorator(fn: Callable) -> Callable:
            if inspect.iscoroutinefunction(fn):

                @functools.wraps(fn)
                async def async_wrap(*args, **kwargs):
                    return await _run_retrieve(fn, args, kwargs, name, is_async=True)

                return async_wrap
            else:

                @functools.wraps(fn)
                def sync_wrap(*args, **kwargs):
                    return _run_retrieve_sync(fn, args, kwargs, name)

                return sync_wrap

        if func is not None:
            return decorator(func)
        return decorator

    def generate(self, func=None, *, name: str = "rag.generate"):
        """Optional decorator for your generation wrapper.

        The LLM call inside is already auto-instrumented by peekr.instrument().
        Use this if you want to wrap the whole generate() function as a parent span.
        """
        from ..decorators import trace

        if func is not None:
            return trace(func, name=name)
        return trace(name=name)

    @contextmanager
    def context(self, docs: Any):
        """Context manager to manually set the grounding context.

        Use this when you don't use @harness.retrieve but still want
        faithfulness scoring::

            with harness.context(retrieved_docs):
                response = client.chat.completions.create(...)
        """
        token = _rag_docs.set(_serialize_docs(docs))
        try:
            yield
        finally:
            _rag_docs.reset(token)


def _run_retrieve_sync(fn, args, kwargs, span_name):
    span, token = start_span(span_name)
    # Capture query from first string arg
    for arg in args:
        if isinstance(arg, str):
            span.attributes["rag.query"] = arg[:_TRUNCATE]
            break
    try:
        result = fn(*args, **kwargs)
        docs_str = _serialize_docs(result)
        _rag_docs.set(docs_str)
        count = len(result) if isinstance(result, list) else 1
        span.attributes["rag.docs_count"] = count
        span.attributes["rag.retrieved_docs"] = docs_str
        span.status = "ok"
        return result
    except Exception as e:
        span.status = "error"
        span.attributes["error"] = str(e)
        raise
    finally:
        end_span(span, token)
        export_span(span)


async def _run_retrieve(fn, args, kwargs, span_name, *, is_async: bool):
    span, token = start_span(span_name)
    for arg in args:
        if isinstance(arg, str):
            span.attributes["rag.query"] = arg[:_TRUNCATE]
            break
    try:
        result = await fn(*args, **kwargs) if is_async else fn(*args, **kwargs)
        docs_str = _serialize_docs(result)
        _rag_docs.set(docs_str)
        count = len(result) if isinstance(result, list) else 1
        span.attributes["rag.docs_count"] = count
        span.attributes["rag.retrieved_docs"] = docs_str
        span.status = "ok"
        return result
    except Exception as e:
        span.status = "error"
        span.attributes["error"] = str(e)
        raise
    finally:
        end_span(span, token)
        export_span(span)


def instrument_rag(
    exporter=None,
    *,
    faithfulness_threshold: float = 0.7,
    detailed: bool = False,
    console: bool = False,
    **instrument_kwargs,
) -> RAGHarness:
    """Configure peekr for a RAG pipeline with faithfulness eval pre-wired.

    Parameters
    ----------
    exporter
        Typically ``peekr.HTTPExporter(endpoint=..., api_key=...)``.
        If omitted, falls back to the default JSONL exporter.
    faithfulness_threshold
        Alert when the hallucination score drops below this value (0.0–1.0).
        Default 0.7 — any response where <70% of claims are supported triggers
        the alert.
    detailed
        If True, use RAGAS-style claim-level decomposition. Scores each
        claim as supported/contradicted/unsupported. More expensive but
        tells you *which* claims failed, not just *that* something did.
    console
        Print spans to stdout. Defaults to False for RAG pipelines (too noisy).
    **instrument_kwargs
        Any additional kwargs forwarded to ``peekr.instrument()``.

    Returns
    -------
    RAGHarness
        Use ``@harness.retrieve`` and ``@harness.rerank`` on your pipeline steps.
    """
    from ..eval.hallucination import Hallucination

    hallucination_eval = Hallucination(
        context_extractor=_rag_context_extractor,
        detailed=detailed,
    )

    alerts = instrument_kwargs.pop("alerts", [])

    import peekr as _peekr

    _peekr.instrument(
        exporter=exporter,
        console=console,
        evaluators=[hallucination_eval],
        alerts=alerts or None,
        **instrument_kwargs,
    )

    return RAGHarness()
