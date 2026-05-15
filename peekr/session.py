from __future__ import annotations

import uuid
from contextvars import ContextVar
from typing import Optional


_session_id: ContextVar[Optional[str]] = ContextVar("session_id", default=None)
_user_id: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
_tenant_id: ContextVar[Optional[str]] = ContextVar("tenant_id", default=None)
_retention_class: ContextVar[Optional[str]] = ContextVar("retention_class", default=None)


def get_session_id() -> Optional[str]:
    return _session_id.get()


def get_user_id() -> Optional[str]:
    return _user_id.get()


def get_tenant_id() -> Optional[str]:
    return _tenant_id.get()


def get_retention_class() -> Optional[str]:
    return _retention_class.get()


class session:
    """Context manager that attaches identity + retention to all spans created within it.

    Uses ContextVars so it works correctly across async/await boundaries.

    - user_id          : the end-user interacting with the agent (B2C identifier)
    - tenant_id        : the customer org hosting the agent (B2B identifier).
                         Distinct from user_id; required for multi-tenant routing.
    - session_id       : conversation/session correlation id
    - retention_class  : storage tier hint. Recommended values:
                         "default" (~30d), "short" (~7d, debug), "long" (audit/compliance),
                         "pii" (special handling). OSS stores it; hosted enforces it.

    Usage::

        with peekr.session(tenant_id="acme", user_id="user_123",
                           retention_class="long"):
            run_agent()
    """

    def __init__(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        retention_class: Optional[str] = None,
    ) -> None:
        self._user_id = user_id
        self._session_id = session_id if session_id is not None else uuid.uuid4().hex
        self._tenant_id = tenant_id
        self._retention_class = retention_class
        self._session_token: object = None
        self._user_token: object = None
        self._tenant_token: object = None
        self._retention_token: object = None

    def __enter__(self) -> "session":
        self._session_token = _session_id.set(self._session_id)
        self._user_token = _user_id.set(self._user_id)
        self._tenant_token = _tenant_id.set(self._tenant_id)
        self._retention_token = _retention_class.set(self._retention_class)
        return self

    def __exit__(self, *_: object) -> None:
        _session_id.reset(self._session_token)
        _user_id.reset(self._user_token)
        _tenant_id.reset(self._tenant_token)
        _retention_class.reset(self._retention_token)

    async def __aenter__(self) -> "session":
        return self.__enter__()

    async def __aexit__(self, *args: object) -> None:
        self.__exit__(*args)
