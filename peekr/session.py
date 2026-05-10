from __future__ import annotations

import uuid
from contextvars import ContextVar
from typing import Optional


_session_id: ContextVar[Optional[str]] = ContextVar("session_id", default=None)
_user_id: ContextVar[Optional[str]] = ContextVar("user_id", default=None)


def get_session_id() -> Optional[str]:
    return _session_id.get()


def get_user_id() -> Optional[str]:
    return _user_id.get()


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
    ) -> None:
        self._user_id = user_id
        self._session_id = session_id if session_id is not None else uuid.uuid4().hex
        self._session_token: object = None
        self._user_token: object = None

    def __enter__(self) -> "session":
        self._session_token = _session_id.set(self._session_id)
        self._user_token = _user_id.set(self._user_id)
        return self

    def __exit__(self, *_: object) -> None:
        _session_id.reset(self._session_token)
        _user_id.reset(self._user_token)

    async def __aenter__(self) -> "session":
        return self.__enter__()

    async def __aexit__(self, *args: object) -> None:
        self.__exit__(*args)
