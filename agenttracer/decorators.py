from __future__ import annotations
import asyncio
import functools
from typing import Callable

from .context import start_span, end_span
from .exporters import export_span


def trace(_func=None, *, name: str | None = None):
    """
    Decorator to trace any function as a span.

    Usage:
        @trace
        def my_tool(query): ...

        @trace(name="tool.search")
        def search(query): ...
    """
    def decorator(func: Callable) -> Callable:
        span_name = name or f"{func.__module__}.{func.__qualname__}"

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                span, token = start_span(span_name)
                try:
                    result = await func(*args, **kwargs)
                    span.status = "ok"
                    return result
                except Exception as e:
                    span.status = "error"
                    span.attributes["error"] = str(e)
                    raise
                finally:
                    end_span(span, token)
                    export_span(span)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                span, token = start_span(span_name)
                try:
                    result = func(*args, **kwargs)
                    span.status = "ok"
                    return result
                except Exception as e:
                    span.status = "error"
                    span.attributes["error"] = str(e)
                    raise
                finally:
                    end_span(span, token)
                    export_span(span)
            return sync_wrapper

    # Handles both @trace and @trace(name="...")
    if _func is not None:
        return decorator(_func)
    return decorator
