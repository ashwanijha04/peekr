from __future__ import annotations
import os
import random
import uuid
from contextvars import ContextVar
from typing import TYPE_CHECKING, Optional

from .span import Span

if TYPE_CHECKING:
    pass  # Span already imported above


# ── GuardrailError ────────────────────────────────────────────────────────────
# Defined here (not in guard/) so patches can import it without any risk of
# circular imports: patches → context is already the established direction.

class GuardrailError(Exception):
    """Raised by a guardrail to abort an LLM call or block its response.

    When raised by an input guardrail (pre-call), the SDK method never fires.
    When raised by a blocking guardrail (post-call), the response is discarded.
    In both cases the span is still exported so violations are auditable.
    """

    def __init__(
        self,
        message: str,
        guardrail_name: str = "",
        span: "Span | None" = None,
    ) -> None:
        super().__init__(message)
        self.guardrail_name = guardrail_name
        self.span = span


# ── Input guard registry ──────────────────────────────────────────────────────
# Guards registered here run BEFORE the LLM API call.  Raising GuardrailError
# aborts the call entirely — the SDK method is never invoked.

_input_guards: list = []


def register_input_guard(guard) -> None:
    """Register a guardrail to run before every LLM call."""
    _input_guards.append(guard)


def clear_input_guards() -> None:
    """Remove all registered input guards (used in tests)."""
    _input_guards.clear()


def _run_input_guards(span: "Span") -> None:
    """Called by patches after input is captured, before the API call.

    Sets ``span.status = "blocked"`` and records the violation on the span
    before re-raising so the finally block can export an auditable record even
    though the API call was aborted.
    """
    first_error: "GuardrailError | None" = None
    for guard in _input_guards:
        try:
            guard.run(span)
        except GuardrailError as exc:
            if first_error is None:
                first_error = exc
        except Exception:
            pass  # infra failure — never abort on it
    if first_error is not None:
        span.status = "blocked"
        raise first_error

_current_span: ContextVar[Optional[Span]] = ContextVar("current_span", default=None)
_current_trace_id: ContextVar[Optional[str]] = ContextVar("current_trace_id", default=None)

# Sampling decision propagates from the root span through children via
# ContextVar (it follows the async task tree). A value of None means
# "not yet decided" — the next root span will decide.
_trace_sample_keep: ContextVar[Optional[bool]] = ContextVar("trace_sample_keep", default=None)

# Process-wide defaults set by instrument(). Lower priority than session() context.
_default_tenant_id: Optional[str] = None
_default_retention_class: Optional[str] = None
_sample_rate: float = 1.0       # fraction of root traces to keep
_keep_errors: bool = True       # always persist errored spans, even when trace dropped


def set_process_defaults(
    tenant_id: Optional[str] = None,
    retention_class: Optional[str] = None,
    sample_rate: Optional[float] = None,
    keep_errors: Optional[bool] = None,
) -> None:
    """Called by instrument() to set process-wide defaults."""
    global _default_tenant_id, _default_retention_class, _sample_rate, _keep_errors
    if tenant_id is not None:
        _default_tenant_id = tenant_id
    if retention_class is not None:
        _default_retention_class = retention_class
    if sample_rate is not None:
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError("sample_rate must be between 0.0 and 1.0")
        _sample_rate = sample_rate
    if keep_errors is not None:
        _keep_errors = keep_errors


def should_persist(span: Span) -> bool:
    """Storage exporters call this to decide whether to write a span.

    Returns True if (a) the trace was sampled in, or (b) keep_errors is on
    and this particular span is an error. Returns False to drop.
    """
    keep = _trace_sample_keep.get()
    if keep is None or keep:
        return True
    return _keep_errors and span.status == "error"


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
    # Root span: make the sampling decision once for the whole trace.
    # Children inherit through the ContextVar.
    if parent is None and _trace_sample_keep.get() is None:
        _trace_sample_keep.set(random.random() < _sample_rate)
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
