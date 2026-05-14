from __future__ import annotations

import uuid
from contextvars import ContextVar
from typing import Optional


_session_id: ContextVar[Optional[str]] = ContextVar("session_id", default=None)
_user_id: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
_grounding: ContextVar[Optional[dict]] = ContextVar("grounding", default=None)


def get_session_id() -> Optional[str]:
    return _session_id.get()


def get_user_id() -> Optional[str]:
    return _user_id.get()


def get_grounding(key: Optional[str] = None):
    """Return the active grounding payload (context / query / etc.).

    Set via ``peekr.session(grounding={...})`` or ``peekr.set_grounding(...)``.
    """
    payload = _grounding.get()
    if payload is None:
        return None
    if key is None:
        return payload
    return payload.get(key)


def set_grounding(**kwargs) -> None:
    """Attach grounding (context, query, …) to the current trace.

    All values are merged into a single dict, so calling ``set_grounding``
    multiple times accumulates fields::

        peekr.set_grounding(query=user_q, context=retrieved_docs)
    """
    existing = _grounding.get() or {}
    payload = {**existing, **kwargs}
    _grounding.set(payload)
    # Also attach to current span attributes for trace-level visibility.
    try:
        from .context import get_current_span
        span = get_current_span()
        if span is not None:
            for key, value in kwargs.items():
                span.attributes[f"grounding.{key}"] = value
    except ImportError:
        pass


class session:
    """Context manager that attaches session_id and user_id to all spans created within it.

    Uses ContextVar so it works correctly across async/await boundaries.

    Usage::

        with peekr.session(user_id="user_123", session_id="sess_abc"):
            run_agent()
        # all spans inside have attributes["session_id"] and attributes["user_id"] set
    """

    def __init__(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        grounding: Optional[dict] = None,
    ) -> None:
        self._user_id = user_id
        self._session_id = session_id if session_id is not None else uuid.uuid4().hex
        self._grounding = grounding
        self._session_token: object = None
        self._user_token: object = None
        self._grounding_token: object = None

    def __enter__(self) -> "session":
        self._session_token = _session_id.set(self._session_id)
        self._user_token = _user_id.set(self._user_id)
        self._grounding_token = _grounding.set(self._grounding)
        return self

    def __exit__(self, *_: object) -> None:
        _session_id.reset(self._session_token)
        _user_id.reset(self._user_token)
        _grounding.reset(self._grounding_token)

    async def __aenter__(self) -> "session":
        return self.__enter__()

    async def __aexit__(self, *args: object) -> None:
        self.__exit__(*args)
