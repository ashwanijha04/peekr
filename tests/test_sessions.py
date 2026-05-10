from __future__ import annotations

import asyncio

import pytest

from peekr.session import session, get_session_id, get_user_id


# ---------------------------------------------------------------------------
# Basic context-manager behaviour
# ---------------------------------------------------------------------------

def test_session_sets_ids():
    with session(user_id="u1", session_id="s1"):
        assert get_session_id() == "s1"
        assert get_user_id() == "u1"


def test_session_clears_ids_on_exit():
    with session(user_id="u1", session_id="s1"):
        pass
    assert get_session_id() is None
    assert get_user_id() is None


def test_session_default_user_id_is_none():
    with session(session_id="s1"):
        assert get_user_id() is None
        assert get_session_id() == "s1"


def test_session_auto_generates_session_id():
    with session(user_id="u1") as sess:
        sid = get_session_id()
    assert sid is not None
    assert len(sid) == 32  # uuid4().hex


def test_session_auto_session_id_unique():
    sids = set()
    for _ in range(10):
        with session() as s:
            sids.add(get_session_id())
    assert len(sids) == 10


def test_session_ids_not_visible_outside():
    assert get_session_id() is None
    assert get_user_id() is None
    with session(user_id="u2", session_id="s2"):
        assert get_session_id() == "s2"
    assert get_session_id() is None
    assert get_user_id() is None


# ---------------------------------------------------------------------------
# Nesting
# ---------------------------------------------------------------------------

def test_session_nesting_restores_outer():
    with session(user_id="outer-user", session_id="outer-sess"):
        assert get_session_id() == "outer-sess"
        with session(user_id="inner-user", session_id="inner-sess"):
            assert get_session_id() == "inner-sess"
            assert get_user_id() == "inner-user"
        # outer is restored
        assert get_session_id() == "outer-sess"
        assert get_user_id() == "outer-user"


# ---------------------------------------------------------------------------
# Exception safety
# ---------------------------------------------------------------------------

def test_session_clears_on_exception():
    try:
        with session(user_id="u3", session_id="s3"):
            raise ValueError("boom")
    except ValueError:
        pass
    assert get_session_id() is None
    assert get_user_id() is None


# ---------------------------------------------------------------------------
# Async support
# ---------------------------------------------------------------------------

def test_async_session_propagates():
    async def run():
        async with session(user_id="async-user", session_id="async-sess"):
            assert get_session_id() == "async-sess"
            assert get_user_id() == "async-user"
        assert get_session_id() is None
        assert get_user_id() is None

    asyncio.run(run())


def test_async_session_isolated_tasks():
    """Each task that enters its own session should see its own IDs."""

    results: dict[str, str] = {}

    async def task(uid: str, sid: str):
        async with session(user_id=uid, session_id=sid):
            await asyncio.sleep(0)  # yield to event loop
            results[uid] = get_session_id()

    async def run():
        await asyncio.gather(
            task("u-a", "s-a"),
            task("u-b", "s-b"),
        )

    asyncio.run(run())
    assert results["u-a"] == "s-a"
    assert results["u-b"] == "s-b"


# ---------------------------------------------------------------------------
# Integration: spans created inside a session pick up session attributes
# (requires context.py to read session vars — tested at unit level here)
# ---------------------------------------------------------------------------

def test_get_session_id_outside_session_returns_none():
    assert get_session_id() is None


def test_get_user_id_outside_session_returns_none():
    assert get_user_id() is None


def test_session_returns_self():
    with session(session_id="s99") as s:
        assert isinstance(s, session)
