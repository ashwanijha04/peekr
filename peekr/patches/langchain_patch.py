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


def _extract_name(serialized, override, default):
    if override:
        return override
    if isinstance(serialized, dict):
        ident = serialized.get("id")
        if isinstance(ident, list) and ident:
            return ident[-1]
        if serialized.get("name"):
            return serialized["name"]
    return default


class PeekrLangChainHandler:
    """
    LangChain BaseCallbackHandler that emits peekr spans for chains, tools,
    retrievers, agents and LLM calls. Spans are keyed by LangChain's run_id
    so callbacks that fire out of order (async, parallel) still finalize the
    right span; parent/child links come from LangChain's parent_run_id and
    fall back to peekr's current span otherwise.
    """

    # BaseCallbackHandler interface flags
    raise_error = False
    run_inline = True
    ignore_llm = False
    ignore_chain = False
    ignore_agent = False
    ignore_retriever = False
    ignore_chat_model = False
    ignore_custom_event = False

    def __init__(self):
        self._runs: dict = {}
        self._lock = Lock()

    def _begin(self, run_id, parent_run_id, name, attrs=None):
        with self._lock:
            parent = self._runs.get(parent_run_id) if parent_run_id else None
        parent_span_id = parent[0].span_id if parent else None
        span, token = _start_framework_span(name, parent_span_id)
        if attrs:
            span.attributes.update(attrs)
        with self._lock:
            self._runs[run_id] = (span, token)
        return span

    def _finish(self, run_id, attrs=None, status="ok"):
        with self._lock:
            entry = self._runs.pop(run_id, None)
        if not entry:
            return None
        span, token = entry
        if attrs:
            span.attributes.update(attrs)
        span.status = status
        _finish_framework_span(span, token)
        return span

    # ── Chain ────────────────────────────────────────────────────────────────
    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        name = _extract_name(serialized, kwargs.get("name"), "run")
        self._begin(run_id, parent_run_id, f"langchain.chain.{name}",
                    {"input": _serialize(inputs)})

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self._finish(run_id, {"output": _serialize(outputs)})

    def on_chain_error(self, error, *, run_id, **kwargs):
        self._finish(run_id, {"error": str(error)}, status="error")

    # ── Tool ─────────────────────────────────────────────────────────────────
    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs):
        name = _extract_name(serialized, kwargs.get("name"), "call")
        self._begin(run_id, parent_run_id, f"langchain.tool.{name}",
                    {"input": _truncate(str(input_str))})

    def on_tool_end(self, output, *, run_id, **kwargs):
        self._finish(run_id, {"output": _serialize(output)})

    def on_tool_error(self, error, *, run_id, **kwargs):
        self._finish(run_id, {"error": str(error)}, status="error")

    # ── Retriever ────────────────────────────────────────────────────────────
    def on_retriever_start(self, serialized, query, *, run_id, parent_run_id=None, **kwargs):
        name = _extract_name(serialized, kwargs.get("name"), "query")
        self._begin(run_id, parent_run_id, f"langchain.retriever.{name}",
                    {"input": _serialize(query)})

    def on_retriever_end(self, documents, *, run_id, **kwargs):
        attrs = {"output": _serialize(documents)}
        try:
            attrs["document_count"] = len(documents)
        except Exception:
            pass
        self._finish(run_id, attrs)

    def on_retriever_error(self, error, *, run_id, **kwargs):
        self._finish(run_id, {"error": str(error)}, status="error")

    # ── Agent ────────────────────────────────────────────────────────────────
    def on_agent_action(self, action, *, run_id, parent_run_id=None, **kwargs):
        tool = getattr(action, "tool", "unknown")
        tool_input = getattr(action, "tool_input", None)
        self._begin(run_id, parent_run_id, "langchain.agent.action", {
            "tool": tool,
            "input": _serialize(tool_input),
        })
        # AgentAction is a point-in-time event in LangChain — no matching end.
        self._finish(run_id)

    def on_agent_finish(self, finish, *, run_id, parent_run_id=None, **kwargs):
        ret = getattr(finish, "return_values", None)
        self._begin(run_id, parent_run_id, "langchain.agent.finish", {
            "output": _serialize(ret),
        })
        self._finish(run_id)

    # ── LLM / Chat model ─────────────────────────────────────────────────────
    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs):
        name = _extract_name(serialized, kwargs.get("name"), "completion")
        invocation = kwargs.get("invocation_params") or {}
        attrs = {
            "input": _serialize(prompts),
            "model": invocation.get("model") or invocation.get("model_name") or "unknown",
        }
        self._begin(run_id, parent_run_id, f"langchain.llm.{name}", attrs)

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
        name = _extract_name(serialized, kwargs.get("name"), "chat")
        invocation = kwargs.get("invocation_params") or {}
        attrs = {
            "input": _serialize(messages),
            "model": invocation.get("model") or invocation.get("model_name") or "unknown",
        }
        self._begin(run_id, parent_run_id, f"langchain.chat.{name}", attrs)

    def on_llm_end(self, response, *, run_id, **kwargs):
        attrs = {}
        try:
            llm_output = getattr(response, "llm_output", None) or {}
            if isinstance(llm_output, dict):
                usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
                if usage:
                    inp = usage.get("prompt_tokens", usage.get("input_tokens", 0))
                    out = usage.get("completion_tokens", usage.get("output_tokens", 0))
                    tot = usage.get("total_tokens", (inp or 0) + (out or 0))
                    attrs["tokens_input"] = inp
                    attrs["tokens_output"] = out
                    attrs["tokens_total"] = tot
            gens = getattr(response, "generations", None)
            if gens:
                first = gens[0][0] if gens[0] else None
                if first is not None:
                    text = getattr(first, "text", None)
                    if text is None:
                        msg = getattr(first, "message", None)
                        text = getattr(msg, "content", None) if msg is not None else None
                    if text is not None:
                        attrs["output"] = _truncate(str(text))
        except Exception:
            pass
        self._finish(run_id, attrs)

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._finish(run_id, {"error": str(error)}, status="error")


def patch_langchain():
    """
    Install a global peekr callback handler on LangChain.

    Uses langchain_core's `register_configure_hook` so every CallbackManager
    constructed downstream picks up our handler — chain / tool / retriever /
    agent / LLM events all flow through `PeekrLangChainHandler` and emit
    peekr spans without any user code changes.
    """
    global _PATCHED
    if _PATCHED:
        return

    BaseCallbackHandler = None
    register_configure_hook = None
    try:
        from langchain_core.callbacks.base import BaseCallbackHandler as _BCH
        BaseCallbackHandler = _BCH
    except ImportError:
        try:
            from langchain.callbacks.base import BaseCallbackHandler as _BCH
            BaseCallbackHandler = _BCH
        except ImportError:
            return

    try:
        from langchain_core.tracers.context import register_configure_hook as _rch
        register_configure_hook = _rch
    except ImportError:
        try:
            from langchain_core.callbacks.manager import register_configure_hook as _rch
            register_configure_hook = _rch
        except ImportError:
            register_configure_hook = None

    handler_cls = type(
        "PeekrLangChainHandlerRegistered",
        (PeekrLangChainHandler, BaseCallbackHandler),
        {"_peekr_patched": True},
    )
    handler = handler_cls()

    if register_configure_hook is not None:
        from contextvars import ContextVar
        cv: ContextVar = ContextVar("peekr_langchain_handler", default=None)
        cv.set(handler)
        try:
            register_configure_hook(cv, True)
        except TypeError:
            try:
                register_configure_hook(cv, True, handler_cls)
            except Exception:
                pass

    _PATCHED = True
