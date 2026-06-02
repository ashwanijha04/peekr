from __future__ import annotations
import time

from ..context import start_span, end_span
from ..exporters import export_span

_TRUNCATE = 1000


def _get_index_name(self) -> str:
    for attr in ("_name", "name", "_index_name", "index_name"):
        val = getattr(self, attr, None)
        if isinstance(val, str) and val:
            return val
    return "unknown"


def _make_query_patch(original):
    def patched(self, *args, **kwargs):
        span, token = start_span("pinecone.query")
        span.attributes["index_name"] = _get_index_name(self)
        span.attributes["top_k"] = kwargs.get("top_k", None)
        span.attributes["namespace"] = kwargs.get("namespace", "")
        t0 = time.time()
        try:
            response = original(self, *args, **kwargs)
            latency_ms = (time.time() - t0) * 1000
            span.attributes["latency_ms"] = round(latency_ms, 2)
            try:
                matches = response.matches if hasattr(response, "matches") else response.get("matches", [])
                span.attributes["num_results"] = len(matches) if matches else 0
            except Exception:
                span.attributes["num_results"] = 0
            span.status = "ok"
            return response
        except Exception as e:
            span.status = "error"
            span.attributes["error"] = str(e)
            raise
        finally:
            end_span(span, token)
            export_span(span)
    return patched


def _make_async_query_patch(original):
    async def patched(self, *args, **kwargs):
        span, token = start_span("pinecone.query")
        span.attributes["index_name"] = _get_index_name(self)
        span.attributes["top_k"] = kwargs.get("top_k", None)
        span.attributes["namespace"] = kwargs.get("namespace", "")
        t0 = time.time()
        try:
            response = await original(self, *args, **kwargs)
            latency_ms = (time.time() - t0) * 1000
            span.attributes["latency_ms"] = round(latency_ms, 2)
            try:
                matches = response.matches if hasattr(response, "matches") else response.get("matches", [])
                span.attributes["num_results"] = len(matches) if matches else 0
            except Exception:
                span.attributes["num_results"] = 0
            span.status = "ok"
            return response
        except Exception as e:
            span.status = "error"
            span.attributes["error"] = str(e)
            raise
        finally:
            end_span(span, token)
            export_span(span)
    return patched


def _make_upsert_patch(original):
    def patched(self, *args, **kwargs):
        span, token = start_span("pinecone.upsert")
        span.attributes["index_name"] = _get_index_name(self)
        vectors = kwargs.get("vectors", args[0] if args else [])
        span.attributes["vector_count"] = len(vectors) if vectors else 0
        try:
            response = original(self, *args, **kwargs)
            span.status = "ok"
            return response
        except Exception as e:
            span.status = "error"
            span.attributes["error"] = str(e)
            raise
        finally:
            end_span(span, token)
            export_span(span)
    return patched


def _make_async_upsert_patch(original):
    async def patched(self, *args, **kwargs):
        span, token = start_span("pinecone.upsert")
        span.attributes["index_name"] = _get_index_name(self)
        vectors = kwargs.get("vectors", args[0] if args else [])
        span.attributes["vector_count"] = len(vectors) if vectors else 0
        try:
            response = await original(self, *args, **kwargs)
            span.status = "ok"
            return response
        except Exception as e:
            span.status = "error"
            span.attributes["error"] = str(e)
            raise
        finally:
            end_span(span, token)
            export_span(span)
    return patched


def patch_pinecone():
    try:
        import pinecone
    except ImportError:
        return

    # ── v2 API: pinecone.Index ────────────────────────────────────────────────
    try:
        Index = pinecone.Index
        if not getattr(Index.query, "_peekr_patched", False):
            orig = Index.query
            Index.query = _make_query_patch(orig)
            Index.query._peekr_patched = True
    except Exception:
        pass

    try:
        Index = pinecone.Index
        if not getattr(Index.upsert, "_peekr_patched", False):
            orig = Index.upsert
            Index.upsert = _make_upsert_patch(orig)
            Index.upsert._peekr_patched = True
    except Exception:
        pass

    # ── v3 API: pinecone.Pinecone().Index (GRPCIndex / Index from pinecone.data) ──
    try:
        from pinecone.data import Index as DataIndex
        if not getattr(DataIndex.query, "_peekr_patched", False):
            orig = DataIndex.query
            DataIndex.query = _make_query_patch(orig)
            DataIndex.query._peekr_patched = True
    except Exception:
        pass

    try:
        from pinecone.data import Index as DataIndex
        if not getattr(DataIndex.upsert, "_peekr_patched", False):
            orig = DataIndex.upsert
            DataIndex.upsert = _make_upsert_patch(orig)
            DataIndex.upsert._peekr_patched = True
    except Exception:
        pass

    # ── async variant (pinecone.data.index.Index async methods if present) ────
    try:
        from pinecone.data import Index as DataIndex
        query_async = getattr(DataIndex, "query_async", None)
        if query_async and not getattr(query_async, "_peekr_patched", False):
            DataIndex.query_async = _make_async_query_patch(query_async)
            DataIndex.query_async._peekr_patched = True
    except Exception:
        pass

    try:
        from pinecone.data import Index as DataIndex
        upsert_async = getattr(DataIndex, "upsert_async", None)
        if upsert_async and not getattr(upsert_async, "_peekr_patched", False):
            DataIndex.upsert_async = _make_async_upsert_patch(upsert_async)
            DataIndex.upsert_async._peekr_patched = True
    except Exception:
        pass
