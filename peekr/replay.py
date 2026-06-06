from __future__ import annotations

"""
Trace replay for peekr.

Re-runs every LLM call found in a stored trace by extracting the original
inputs and invoking the real SDK again.  Because peekr is already patched,
each re-invocation creates a new span that is exported normally, giving you
a second trace you can compare against the original.

Programmatic API
----------------
    from peekr.replay import replay_trace

    new_trace_id = replay_trace(
        trace_id="a3f2b1c0",
        db_path="traces.db",   # or jsonl_path="traces.jsonl"
    )

CLI
---
    peekr replay a3f2b1c0
    peekr replay a3f2b1c0 --db traces.db
    peekr replay a3f2b1c0 --jsonl traces.jsonl
"""
import json  # noqa: E402
import sqlite3  # noqa: E402
import uuid  # noqa: E402
from typing import Optional  # noqa: E402


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


def load_trace(
    trace_id: str,
    db_path: Optional[str] = None,
    jsonl_path: Optional[str] = None,
) -> list[dict]:
    """
    Load all spans for *trace_id* from SQLite or JSONL storage.

    At least one of *db_path* or *jsonl_path* must be provided.  If both are
    given, SQLite takes priority.
    """
    if db_path is not None:
        return _load_from_sqlite(trace_id, db_path)
    if jsonl_path is not None:
        return _load_from_jsonl(trace_id, jsonl_path)
    raise ValueError("Provide db_path or jsonl_path")


def _load_from_sqlite(trace_id: str, db_path: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_time",
        (trace_id,),
    ).fetchall()
    conn.close()
    spans = []
    for r in rows:
        s = dict(r)
        s["attributes"] = json.loads(s.get("attributes") or "{}")
        spans.append(s)
    return spans


def _load_from_jsonl(trace_id: str, jsonl_path: str) -> list[dict]:
    spans = []
    with open(jsonl_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            if s.get("trace_id") == trace_id:
                # attributes may be a dict already (JSONL) or a JSON string
                if isinstance(s.get("attributes"), str):
                    s["attributes"] = json.loads(s["attributes"])
                spans.append(s)
    return spans


# ---------------------------------------------------------------------------
# LLM re-invocation helpers
# ---------------------------------------------------------------------------

_LLM_PREFIXES = ("openai.", "anthropic.", "bedrock.")


def _is_llm_span(span: dict) -> bool:
    name = span.get("name", "")
    return any(name.startswith(p) for p in _LLM_PREFIXES)


def _replay_openai(span: dict) -> None:
    import sys  # noqa: PLC0415

    openai = sys.modules.get("openai")
    if openai is None:
        import openai as _openai  # noqa: PLC0415

        openai = _openai
    attrs = span.get("attributes", {})
    model = attrs.get("model", "gpt-4o")
    raw_input = attrs.get("input")
    if not raw_input:
        return
    messages = json.loads(raw_input)
    openai.chat.completions.create(model=model, messages=messages)


def _replay_anthropic(span: dict) -> None:
    import sys  # noqa: PLC0415

    anthropic = sys.modules.get("anthropic")
    if anthropic is None:
        import anthropic as _anthropic  # noqa: PLC0415

        anthropic = _anthropic
    attrs = span.get("attributes", {})
    model = attrs.get("model", "claude-3-5-haiku-20241022")
    raw_input = attrs.get("input")
    if not raw_input:
        return
    messages = json.loads(raw_input)
    client = anthropic.Anthropic()
    client.messages.create(model=model, max_tokens=1024, messages=messages)


def _replay_bedrock(span: dict) -> None:
    import sys  # noqa: PLC0415

    boto3 = sys.modules.get("boto3")
    if boto3 is None:
        import boto3 as _boto3  # noqa: PLC0415

        boto3 = _boto3
    attrs = span.get("attributes", {})
    model = attrs.get("model", "amazon.titan-text-lite-v1")
    raw_input = attrs.get("input")
    if not raw_input:
        return
    messages = json.loads(raw_input)
    client = boto3.client("bedrock-runtime")
    client.converse(modelId=model, messages=messages)


def _replay_span(span: dict) -> None:
    name = span.get("name", "")
    if name.startswith("openai."):
        _replay_openai(span)
    elif name.startswith("anthropic."):
        _replay_anthropic(span)
    elif name.startswith("bedrock."):
        _replay_bedrock(span)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def replay_trace(
    trace_id: str,
    db_path: Optional[str] = None,
    jsonl_path: Optional[str] = None,
) -> str:
    """
    Re-run every LLM call found in *trace_id* and return the new trace_id.

    The new trace_id is set before the first replay call so that all
    re-invoked spans share it.  Because peekr patches the SDKs at
    instrument() time, each call is automatically traced and exported.

    Parameters
    ----------
    trace_id:
        The original trace to replay.
    db_path:
        Path to a SQLite database (``traces.db``).
    jsonl_path:
        Path to a JSONL file (``traces.jsonl``).  Used only when *db_path*
        is not provided.

    Returns
    -------
    str
        The new trace_id shared by all replayed spans.
    """
    # Default storage path
    if db_path is None and jsonl_path is None:
        import os

        if os.path.exists("traces.db"):
            db_path = "traces.db"
        else:
            jsonl_path = "traces.jsonl"

    spans = load_trace(trace_id, db_path=db_path, jsonl_path=jsonl_path)
    llm_spans = [s for s in spans if _is_llm_span(s)]

    if not llm_spans:
        raise ValueError(f"No LLM spans found in trace {trace_id!r}")

    # Stamp a new trace_id into the context so all replayed calls share it
    from .context import _current_trace_id  # noqa: PLC0415

    new_trace_id = uuid.uuid4().hex
    token = _current_trace_id.set(new_trace_id)
    try:
        for span in llm_spans:
            _replay_span(span)
    finally:
        _current_trace_id.reset(token)

    return new_trace_id
