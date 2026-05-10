from __future__ import annotations
import uuid
from contextvars import ContextVar
from typing import Optional

from .span import Span

_current_span: ContextVar[Optional[Span]] = ContextVar("current_span", default=None)
_current_trace_id: ContextVar[Optional[str]] = ContextVar("current_trace_id", default=None)


def get_current_span() -> Optional[Span]:
    return _current_span.get()


def get_or_create_trace_id() -> str:
    trace_id = _current_trace_id.get()
    if trace_id is None:
        trace_id = uuid.uuid4().hex
        _current_trace_id.set(trace_id)
    return trace_id


def start_span(name: str) -> tuple[Span, object]:
    trace_id = get_or_create_trace_id()
    parent = get_current_span()
    span = Span(
        name=name,
        trace_id=trace_id,
        parent_id=parent.span_id if parent else None,
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
