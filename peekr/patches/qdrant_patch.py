from __future__ import annotations

from ..context import start_span, end_span
from ..exporters import export_span

_TRUNCATE = 1000


def _make_search_patch(original):
    def patched(self, *args, **kwargs):
        span, token = start_span("qdrant.search")
        collection = str(args[0]) if args else str(kwargs.get("collection_name", ""))
        span.attributes["collection"] = collection
        span.attributes["limit"] = kwargs.get("limit", 10)
        try:
            response = original(self, *args, **kwargs)
            span.attributes["num_results"] = len(response) if response else 0
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


def _make_async_search_patch(original):
    async def patched(self, *args, **kwargs):
        span, token = start_span("qdrant.search")
        collection = str(args[0]) if args else str(kwargs.get("collection_name", ""))
        span.attributes["collection"] = collection
        span.attributes["limit"] = kwargs.get("limit", 10)
        try:
            response = await original(self, *args, **kwargs)
            span.attributes["num_results"] = len(response) if response else 0
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
        span, token = start_span("qdrant.upsert")
        collection = str(args[0]) if args else str(kwargs.get("collection_name", ""))
        span.attributes["collection"] = collection
        points = kwargs.get("points", [])
        span.attributes["point_count"] = len(points) if points else 0
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
        span, token = start_span("qdrant.upsert")
        collection = str(args[0]) if args else str(kwargs.get("collection_name", ""))
        span.attributes["collection"] = collection
        points = kwargs.get("points", [])
        span.attributes["point_count"] = len(points) if points else 0
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


def patch_qdrant():
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        return

    # ── sync QdrantClient ─────────────────────────────────────────────────────
    try:
        if not getattr(QdrantClient.search, "_peekr_patched", False):
            orig = QdrantClient.search
            QdrantClient.search = _make_search_patch(orig)
            QdrantClient.search._peekr_patched = True
    except Exception:
        pass

    try:
        if not getattr(QdrantClient.upsert, "_peekr_patched", False):
            orig = QdrantClient.upsert
            QdrantClient.upsert = _make_upsert_patch(orig)
            QdrantClient.upsert._peekr_patched = True
    except Exception:
        pass

    # ── async QdrantClient (AsyncQdrantClient) ────────────────────────────────
    try:
        from qdrant_client import AsyncQdrantClient
        if not getattr(AsyncQdrantClient.search, "_peekr_patched", False):
            orig = AsyncQdrantClient.search
            AsyncQdrantClient.search = _make_async_search_patch(orig)
            AsyncQdrantClient.search._peekr_patched = True
    except Exception:
        pass

    try:
        from qdrant_client import AsyncQdrantClient
        if not getattr(AsyncQdrantClient.upsert, "_peekr_patched", False):
            orig = AsyncQdrantClient.upsert
            AsyncQdrantClient.upsert = _make_async_upsert_patch(orig)
            AsyncQdrantClient.upsert._peekr_patched = True
    except Exception:
        pass
