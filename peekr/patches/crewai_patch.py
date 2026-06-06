from __future__ import annotations
import json

from ..context import start_span, end_span
from ..exporters import export_span

_TRUNCATE = 1000


def _truncate(s: str) -> str:
    return s[:_TRUNCATE] + "…" if len(s) > _TRUNCATE else s


def _serialize(value) -> str:
    try:
        s = json.dumps(value, default=str)
    except Exception:
        s = str(value)
    return _truncate(s)


def _wrap(method, span_name, attrs_fn):
    """Wrap a CrewAI bound-ish method as a peekr span.

    `attrs_fn(self, args, kwargs)` returns a dict of starting attributes
    (e.g. agent role, task description). The method is called with the
    receiver as the first positional arg, just like a normal class method.
    """
    if getattr(method, "_peekr_patched", False):
        return method

    def wrapper(self, *args, **kwargs):
        span, token = start_span(span_name)
        try:
            attrs = attrs_fn(self, args, kwargs) or {}
            if attrs:
                span.attributes.update(attrs)
        except Exception:
            pass
        try:
            result = method(self, *args, **kwargs)
            span.status = "ok"
            try:
                if result is not None:
                    span.attributes["output"] = _truncate(str(result))
            except Exception:
                pass
            return result
        except Exception as e:
            span.status = "error"
            span.attributes["error"] = str(e)
            raise
        finally:
            end_span(span, token)
            export_span(span)

    wrapper._peekr_patched = True
    wrapper.__wrapped__ = method
    return wrapper


def _crew_attrs(self, args, kwargs):
    out = {}
    agents = getattr(self, "agents", None)
    if agents is not None:
        try:
            out["agent_count"] = len(agents)
        except Exception:
            pass
    tasks = getattr(self, "tasks", None)
    if tasks is not None:
        try:
            out["task_count"] = len(tasks)
        except Exception:
            pass
    process = getattr(self, "process", None)
    if process is not None:
        out["process"] = str(getattr(process, "value", process))
    if kwargs.get("inputs") is not None:
        out["input"] = _serialize(kwargs["inputs"])
    return out


def _agent_attrs(self, args, kwargs):
    out = {}
    role = getattr(self, "role", None)
    if role is not None:
        out["agent"] = _truncate(str(role))
    # CrewAI passes the task either positionally or as `task=...`
    task = kwargs.get("task")
    if task is None and args:
        task = args[0]
    if task is not None:
        desc = getattr(task, "description", None) or str(task)
        out["task"] = _truncate(str(desc))
    return out


def _task_attrs(self, args, kwargs):
    out = {}
    desc = getattr(self, "description", None)
    if desc is not None:
        out["task"] = _truncate(str(desc))
    agent = getattr(self, "agent", None)
    role = getattr(agent, "role", None) if agent is not None else None
    if role is not None:
        out["agent"] = _truncate(str(role))
    return out


def patch_crewai():
    """
    Monkey-patch the CrewAI execution surface so a crew kickoff produces a
    nested span tree:

        crewai.crew.kickoff
          └── crewai.task.execute
                └── crewai.agent.execute_task
                      └── openai.chat.completions  ← from openai_patch

    Idempotent — methods are marked with `_peekr_patched=True` after wrapping.
    """
    try:
        from crewai import Crew, Agent, Task
    except ImportError:
        return

    if not getattr(Crew.kickoff, "_peekr_patched", False):
        Crew.kickoff = _wrap(Crew.kickoff, "crewai.crew.kickoff", _crew_attrs)

    kickoff_async = getattr(Crew, "kickoff_async", None)
    if kickoff_async is not None and not getattr(
        kickoff_async, "_peekr_patched", False
    ):
        Crew.kickoff_async = _wrap(kickoff_async, "crewai.crew.kickoff", _crew_attrs)

    if not getattr(Agent.execute_task, "_peekr_patched", False):
        Agent.execute_task = _wrap(
            Agent.execute_task, "crewai.agent.execute_task", _agent_attrs
        )

    for method_name in ("execute_sync", "execute", "_execute_core"):
        method = getattr(Task, method_name, None)
        if method is None or getattr(method, "_peekr_patched", False):
            continue
        setattr(Task, method_name, _wrap(method, "crewai.task.execute", _task_attrs))
