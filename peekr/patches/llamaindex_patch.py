from __future__ import annotations
import json
from threading import Lock

from ..context import _current_span, get_or_create_trace_id
from ..exporters import export_span
from ..span import Span

_TRUNCATE = 1000
_PATCHED = False


def _truncate(s: str) -> str:
    return s[:_TRUNCATE] + "…" if len(s) > _TRUNCATE else s


def _serialize(value) -> str:
    try:
        s = json.dumps(value, default=str)
    except Exception:
        s = str(value)
    return _truncate(s)


def _start_framework_span(name, parent_span_id):
    trace_id = get_or_create_trace_id()
    if parent_span_id is None:
        current = _current_span.get()
        if current is not None:
            parent_span_id = current.span_id
    span = Span(name=name, trace_id=trace_id, parent_id=parent_span_id)
    try:
        from ..session import get_session_id, get_user_id

        sid = get_session_id()
        uid = get_user_id()
        if sid:
            span.attributes["session_id"] = sid
        if uid:
            span.attributes["user_id"] = uid
    except ImportError:
        pass
    token = _current_span.set(span)
    return span, token


def _finish_framework_span(span, token):
    span.finish()
    export_span(span)
    if token is not None:
        try:
            _current_span.reset(token)
        except (LookupError, ValueError):
            pass


def _event_type_str(event_type) -> str:
    """Return a clean lowercase event-type string for span naming.

    LlamaIndex passes either a CBEventType enum or a string. Enum values are
    SHOUTY ('LLM', 'RETRIEVE', ...); we normalize to dotted lowercase.
    """
    if event_type is None:
        return "event"
    val = getattr(event_type, "value", event_type)
    return str(val).lower()


class PeekrLlamaIndexHandler:
    """
    LlamaIndex BaseCallbackHandler that emits peekr spans for LLM calls,
    retrievers, agent steps, query / synthesize / function-call events.

    LlamaIndex's CallbackManager fires `on_event_start(event_type, payload,
    event_id, parent_id)` and `on_event_end(event_type, payload, event_id)`
    for every traced event. We mirror those into peekr spans, keyed by
    event_id, parented via parent_id.
    """

    def __init__(self, event_starts_to_ignore=None, event_ends_to_ignore=None):
        # LlamaIndex BaseCallbackHandler.__init__ expects these
        self.event_starts_to_ignore = list(event_starts_to_ignore or [])
        self.event_ends_to_ignore = list(event_ends_to_ignore or [])
        self._events: dict = {}
        self._lock = Lock()

    # ── Trace lifecycle (top-level query / agent run) ────────────────────────
    def start_trace(self, trace_id=None):
        # LlamaIndex calls this at the start of a logical trace. We rely on
        # peekr's own trace_id ContextVar, so this is a no-op — events still
        # nest correctly under whatever peekr span is current.
        return

    def end_trace(self, trace_id=None, trace_map=None):
        return

    # ── Event hooks ──────────────────────────────────────────────────────────
    def on_event_start(
        self, event_type, payload=None, event_id="", parent_id="", **kwargs
    ):
        type_str = _event_type_str(event_type)
        with self._lock:
            parent_entry = self._events.get(parent_id) if parent_id else None
        parent_span_id = parent_entry[0].span_id if parent_entry else None

        span_name = f"llamaindex.{type_str}"
        attrs = {}

        payload = payload or {}
        if type_str == "llm":
            model = (
                payload.get("serialized", {}).get("model")
                if isinstance(payload.get("serialized"), dict)
                else None
            )
            attrs["model"] = model or payload.get("model") or "unknown"
            if "messages" in payload:
                attrs["input"] = _serialize(payload["messages"])
            elif "prompt" in payload:
                attrs["input"] = _serialize(payload["prompt"])
        elif type_str == "retrieve":
            if "query_str" in payload:
                attrs["input"] = _truncate(str(payload["query_str"]))
            elif "query" in payload:
                attrs["input"] = _serialize(payload["query"])
        elif type_str == "query":
            if "query_str" in payload:
                attrs["input"] = _truncate(str(payload["query_str"]))
        elif type_str == "agent_step":
            if "messages" in payload:
                attrs["input"] = _serialize(payload["messages"])
        elif type_str == "function_call":
            if "function_call" in payload:
                attrs["tool"] = _truncate(str(payload["function_call"]))
            if "tool" in payload:
                attrs["tool"] = _truncate(str(payload["tool"]))

        span, token = _start_framework_span(span_name, parent_span_id)
        if attrs:
            span.attributes.update(attrs)
        with self._lock:
            self._events[event_id] = (span, token)
        return event_id

    def on_event_end(self, event_type, payload=None, event_id="", **kwargs):
        with self._lock:
            entry = self._events.pop(event_id, None)
        if not entry:
            return
        span, token = entry
        type_str = _event_type_str(event_type)
        payload = payload or {}

        attrs = {}
        try:
            if type_str == "llm":
                response = payload.get("response")
                if response is not None:
                    raw = getattr(response, "raw", None)
                    usage = None
                    if raw is not None:
                        usage = getattr(raw, "usage", None)
                        if usage is None and isinstance(raw, dict):
                            usage = raw.get("usage")
                    if usage is not None:
                        inp = getattr(usage, "prompt_tokens", None)
                        if inp is None and isinstance(usage, dict):
                            inp = usage.get(
                                "prompt_tokens", usage.get("input_tokens", 0)
                            )
                        out = getattr(usage, "completion_tokens", None)
                        if out is None and isinstance(usage, dict):
                            out = usage.get(
                                "completion_tokens", usage.get("output_tokens", 0)
                            )
                        tot = getattr(usage, "total_tokens", None)
                        if tot is None and isinstance(usage, dict):
                            tot = usage.get("total_tokens", (inp or 0) + (out or 0))
                        if inp is not None:
                            attrs["tokens_input"] = inp
                        if out is not None:
                            attrs["tokens_output"] = out
                        if tot is not None:
                            attrs["tokens_total"] = tot
                    text = getattr(response, "text", None) or getattr(
                        response, "content", None
                    )
                    if text is None:
                        msg = getattr(response, "message", None)
                        text = (
                            getattr(msg, "content", None) if msg is not None else None
                        )
                    if text is not None:
                        attrs["output"] = _truncate(str(text))
                if "response" not in payload and "completion" in payload:
                    attrs["output"] = _truncate(str(payload["completion"]))
            elif type_str == "retrieve":
                nodes = payload.get("nodes")
                if nodes is not None:
                    attrs["output"] = _serialize(nodes)
                    try:
                        attrs["document_count"] = len(nodes)
                    except Exception:
                        pass
            elif type_str == "query":
                response = payload.get("response")
                if response is not None:
                    attrs["output"] = _truncate(str(response))
            elif type_str == "function_call":
                if "function_call_response" in payload:
                    attrs["output"] = _truncate(str(payload["function_call_response"]))
        except Exception:
            pass

        if attrs:
            span.attributes.update(attrs)

        if isinstance(payload, dict) and payload.get("exception") is not None:
            span.status = "error"
            span.attributes["error"] = str(payload["exception"])

        _finish_framework_span(span, token)


def patch_llamaindex():
    """
    Install peekr's global callback handler on LlamaIndex.

    Sets `llama_index.core.Settings.callback_manager` so every query engine,
    retriever, and agent constructed downstream emits its events through
    our handler. Falls back to the legacy `llama_index.callbacks` namespace
    when the new layout is unavailable.
    """
    global _PATCHED
    if _PATCHED:
        return

    BaseCallbackHandler = None
    CallbackManager = None
    Settings = None
    _ = None

    try:
        from llama_index.core.callbacks.base_handler import BaseCallbackHandler as _BCH
        from llama_index.core.callbacks import CallbackManager as _CM
        from llama_index.core import Settings as _Settings

        BaseCallbackHandler = _BCH
        CallbackManager = _CM
        Settings = _Settings
    except ImportError:
        try:
            from llama_index.callbacks.base import BaseCallbackHandler as _BCH
            from llama_index.callbacks import CallbackManager as _CM

            BaseCallbackHandler = _BCH
            CallbackManager = _CM
        except ImportError:
            return

    handler_cls = type(
        "PeekrLlamaIndexHandlerRegistered",
        (PeekrLlamaIndexHandler, BaseCallbackHandler),
        {"_peekr_patched": True},
    )
    handler = handler_cls()

    try:
        if Settings is not None:
            existing = getattr(Settings, "callback_manager", None)
            handlers = list(getattr(existing, "handlers", []) or [])
            handlers.append(handler)
            Settings.callback_manager = CallbackManager(handlers)
        else:
            # Legacy path — best effort
            from llama_index import set_global_handler as _sgh  # type: ignore

            _sgh("simple")  # initialise global handler infra
    except Exception:
        pass

    _PATCHED = True
