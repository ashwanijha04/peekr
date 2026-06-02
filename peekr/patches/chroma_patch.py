from __future__ import annotations

from ..context import start_span, end_span
from ..exporters import export_span

_TRUNCATE = 1000


def _make_query_patch(original):
    def patched(self, *args, **kwargs):
        span, token = start_span("chroma.query")
        span.attributes["collection"] = getattr(self, "name", "unknown")
        span.attributes["n_results"] = kwargs.get("n_results", 10)
        try:
            result = original(self, *args, **kwargs)
            try:
                ids = result.get("ids") if result else None
                span.attributes["num_returned"] = len(ids[0]) if ids and ids else 0
            except Exception:
                span.attributes["num_returned"] = 0
            span.status = "ok"
            return result
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
        span, token = start_span("chroma.query")
        span.attributes["collection"] = getattr(self, "name", "unknown")
        span.attributes["n_results"] = kwargs.get("n_results", 10)
        try:
            result = await original(self, *args, **kwargs)
            try:
                ids = result.get("ids") if result else None
                span.attributes["num_returned"] = len(ids[0]) if ids and ids else 0
            except Exception:
                span.attributes["num_returned"] = 0
            span.status = "ok"
            return result
        except Exception as e:
            span.status = "error"
            span.attributes["error"] = str(e)
            raise
        finally:
            end_span(span, token)
            export_span(span)
    return patched


def _make_add_patch(original):
    def patched(self, *args, **kwargs):
        span, token = start_span("chroma.add")
        span.attributes["collection"] = getattr(self, "name", "unknown")
        documents = kwargs.get("documents", args[0] if args else [])
        span.attributes["doc_count"] = len(documents) if documents else 0
        try:
            result = original(self, *args, **kwargs)
            span.status = "ok"
            return result
        except Exception as e:
            span.status = "error"
            span.attributes["error"] = str(e)
            raise
        finally:
            end_span(span, token)
            export_span(span)
    return patched


def _make_async_add_patch(original):
    async def patched(self, *args, **kwargs):
        span, token = start_span("chroma.add")
        span.attributes["collection"] = getattr(self, "name", "unknown")
        documents = kwargs.get("documents", args[0] if args else [])
        span.attributes["doc_count"] = len(documents) if documents else 0
        try:
            result = await original(self, *args, **kwargs)
            span.status = "ok"
            return result
        except Exception as e:
            span.status = "error"
            span.attributes["error"] = str(e)
            raise
        finally:
            end_span(span, token)
            export_span(span)
    return patched


def patch_chroma():
    try:
        import chromadb
    except ImportError:
        return

    # ── sync Collection ───────────────────────────────────────────────────────
    try:
        Collection = chromadb.Collection
        if not getattr(Collection.query, "_peekr_patched", False):
            orig = Collection.query
            Collection.query = _make_query_patch(orig)
            Collection.query._peekr_patched = True
    except Exception:
        pass

    try:
        Collection = chromadb.Collection
        if not getattr(Collection.add, "_peekr_patched", False):
            orig = Collection.add
            Collection.add = _make_add_patch(orig)
            Collection.add._peekr_patched = True
    except Exception:
        pass

    # ── also patch via chromadb.api.models.Collection if present ─────────────
    try:
        from chromadb.api.models.Collection import Collection as ModelCollection
        if not getattr(ModelCollection.query, "_peekr_patched", False):
            orig = ModelCollection.query
            ModelCollection.query = _make_query_patch(orig)
            ModelCollection.query._peekr_patched = True
    except Exception:
        pass

    try:
        from chromadb.api.models.Collection import Collection as ModelCollection
        if not getattr(ModelCollection.add, "_peekr_patched", False):
            orig = ModelCollection.add
            ModelCollection.add = _make_add_patch(orig)
            ModelCollection.add._peekr_patched = True
    except Exception:
        pass

    # ── async variant (AsyncCollection if present) ────────────────────────────
    try:
        from chromadb.api.async_api import AsyncCollection
        if not getattr(AsyncCollection.query, "_peekr_patched", False):
            orig = AsyncCollection.query
            AsyncCollection.query = _make_async_query_patch(orig)
            AsyncCollection.query._peekr_patched = True
    except Exception:
        pass

    try:
        from chromadb.api.async_api import AsyncCollection
        if not getattr(AsyncCollection.add, "_peekr_patched", False):
            orig = AsyncCollection.add
            AsyncCollection.add = _make_async_add_patch(orig)
            AsyncCollection.add._peekr_patched = True
    except Exception:
        pass
