from __future__ import annotations
import os
import uuid
from contextvars import ContextVar
from typing import Optional

from .span import Span

_current_span: ContextVar[Optional[Span]] = ContextVar("current_span", default=None)
_current_trace_id: ContextVar[Optional[str]] = ContextVar("current_trace_id", default=None)

# Process-wide defaults set by instrument(). Lower priority than session() context.
_default_tenant_id: Optional[str] = None
_default_retention_class: Optional[str] = None


def set_process_defaults(
    tenant_id: Optional[str] = None,
    retention_class: Optional[str] = None,
) -> None:
    """Called by instrument() to set process-wide identity/retention fallbacks."""
    global _default_tenant_id, _default_retention_class
    if tenant_id is not None:
        _default_tenant_id = tenant_id
    if retention_class is not None:
        _default_retention_class = retention_class


def get_current_span() -> Optional[Span]:
    return _current_span.get()


def get_or_create_trace_id() -> str:
    trace_id = _current_trace_id.get()
    if trace_id is None:
        trace_id = uuid.uuid4().hex
        _current_trace_id.set(trace_id)
    return trace_id


def _resolve_tenant_id() -> Optional[str]:
    """Resolution order: session() > instrument default > env > None."""
    from .session import get_tenant_id
    return (
        get_tenant_id()
        or _default_tenant_id
        or os.environ.get("PEEKR_TENANT_ID")
    )


def _resolve_retention_class() -> Optional[str]:
    from .session import get_retention_class
    return (
        get_retention_class()
        or _default_retention_class
        or os.environ.get("PEEKR_RETENTION_CLASS")
    )


def start_span(name: str) -> tuple[Span, object]:
    trace_id = get_or_create_trace_id()
    parent = get_current_span()
    span = Span(
        name=name,
        trace_id=trace_id,
        parent_id=parent.span_id if parent else None,
        tenant_id=_resolve_tenant_id(),
        retention_class=_resolve_retention_class(),
    )
    # Attach session metadata if a session is active
    try:
        from .session import get_session_id, get_user_id
        sid = get_session_id()
        uid = get_user_id()
        if sid:
            span.attributes["session_id"] = sid
        if uid:
            span.attributes["user_id"] = uid
    except ImportError:
        pass
    span_token = _current_span.set(span)
    return span, span_token


def end_span(span: Span, span_token: object) -> None:
    span.finish()
    _current_span.reset(span_token)
