from __future__ import annotations

import abc
import asyncio
import functools
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

from ..context import get_current_span, start_span, end_span
from ..exporters import export_span
from ..span import Span


VALID_ACTIONS = {"warn", "redact", "block"}


class GuardrailViolation(Exception):
    """Raised when a guardrail with action='block' fails."""

    def __init__(self, name: str, findings: list[dict], text: str):
        self.name = name
        self.findings = findings
        self.text = text
        super().__init__(f"Guardrail '{name}' blocked: {findings}")


@dataclass
class GuardrailResult:
    """Outcome of a single guardrail check."""

    name: str
    passed: bool
    findings: list[dict] = field(default_factory=list)
    redacted: Optional[str] = None
    action: str = "warn"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "action": self.action,
            "findings": self.findings,
            "redacted": self.redacted is not None,
        }


class Guardrail(abc.ABC):
    """Base class for all guardrails.

    Subclasses implement ``_check(text) -> GuardrailResult`` and the framework
    handles action enforcement (warn / redact / block) and span annotation.
    """

    def __init__(self, action: str = "warn") -> None:
        if action not in VALID_ACTIONS:
            raise ValueError(
                f"action must be one of {sorted(VALID_ACTIONS)}, got {action!r}"
            )
        self.action = action

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abc.abstractmethod
    def _check(self, text: str) -> GuardrailResult:
        """Inspect ``text`` and return a GuardrailResult."""
        ...

    def check(self, text: str) -> GuardrailResult:
        """Public entry point — returns a result without enforcing the action."""
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        result = self._check(text)
        result.action = self.action
        return result


def _annotate_span(span: Optional[Span], result: GuardrailResult) -> None:
    if span is None:
        return
    bucket = span.attributes.setdefault("guardrails", {})
    bucket[result.name] = result.to_dict()
    if not result.passed:
        span.attributes.setdefault("guardrail_violations", []).append(result.name)


def run_guards(
    text: str,
    guards: Iterable[Guardrail],
    *,
    span: Optional[Span] = None,
    direction: str = "input",
) -> str:
    """Run a sequence of guardrails over ``text`` and apply their actions.

    Returns the (potentially redacted) text. Raises ``GuardrailViolation`` if
    any guardrail with ``action='block'`` fails. All results are annotated
    onto ``span`` if provided (defaults to the current span).
    """
    target_span = span if span is not None else get_current_span()
    current_text = text if isinstance(text, str) else "" if text is None else str(text)

    for guardrail in guards:
        try:
            result = guardrail.check(current_text)
        except Exception as exc:
            # A faulty guardrail must never break the user's pipeline.
            if target_span is not None:
                target_span.attributes.setdefault("guardrail_errors", []).append(
                    {"name": guardrail.name, "error": str(exc), "direction": direction}
                )
            continue

        result_dict = result.to_dict()
        result_dict["direction"] = direction
        if target_span is not None:
            bucket = target_span.attributes.setdefault("guardrails", {})
            bucket.setdefault(direction, {})[result.name] = result_dict
            if not result.passed:
                target_span.attributes.setdefault("guardrail_violations", []).append(
                    f"{direction}.{result.name}"
                )

        if result.passed:
            continue

        if result.action == "block":
            raise GuardrailViolation(result.name, result.findings, current_text)
        if result.action == "redact" and result.redacted is not None:
            current_text = result.redacted

    return current_text


class GuardrailExporter:
    """Exporter that runs guardrails on LLM span input/output.

    Registered automatically by ``peekr.instrument(guards=[...])``. Findings
    are stored on the span's ``guardrails`` attribute; ``block`` actions
    raise after the span is exported so users see what happened.
    """

    _LLM_PREFIXES = ("openai.", "anthropic.", "bedrock.", "gemini.", "google.")

    def __init__(
        self,
        guards: list[Guardrail],
        *,
        on_input: bool = True,
        on_output: bool = True,
    ) -> None:
        self.guards = guards
        self.on_input = on_input
        self.on_output = on_output

    def export(self, span: Span) -> None:
        if not any(span.name.startswith(p) for p in self._LLM_PREFIXES):
            return
        attrs = span.attributes
        if self.on_input and isinstance(attrs.get("input"), str):
            self._run(span, attrs["input"], "input")
        if self.on_output and isinstance(attrs.get("output"), str):
            self._run(span, attrs["output"], "output")

    def _run(self, span: Span, text: str, direction: str) -> None:
        for guardrail in self.guards:
            try:
                result = guardrail.check(text)
            except Exception as exc:
                span.attributes.setdefault("guardrail_errors", []).append(
                    {"name": guardrail.name, "error": str(exc), "direction": direction}
                )
                continue
            bucket = span.attributes.setdefault("guardrails", {})
            bucket.setdefault(direction, {})[result.name] = result.to_dict()
            if not result.passed:
                span.attributes.setdefault("guardrail_violations", []).append(
                    f"{direction}.{result.name}"
                )


def guard(
    _func: Optional[Callable] = None,
    *,
    input: Optional[list[Guardrail]] = None,
    output: Optional[list[Guardrail]] = None,
    name: Optional[str] = None,
) -> Callable:
    """Decorator: run guardrails on a function's first string argument (input)
    and on its return value (output). Findings are recorded on a span named
    ``guardrail.<func>``.

    Example::

        @guard(input=[PromptInjection()], output=[PII(action="redact")])
        def chat(user_message: str) -> str: ...
    """
    input_guards = list(input or [])
    output_guards = list(output or [])

    def decorator(func: Callable) -> Callable:
        span_name = name or f"guardrail.{func.__qualname__}"

        def _first_text_arg(args: tuple, kwargs: dict) -> Optional[str]:
            if args and isinstance(args[0], str):
                return args[0]
            for value in kwargs.values():
                if isinstance(value, str):
                    return value
            return None

        def _replace_first_text_arg(args: tuple, kwargs: dict, new_text: str):
            if args and isinstance(args[0], str):
                return (new_text,) + args[1:], kwargs
            new_kwargs = dict(kwargs)
            for key, value in kwargs.items():
                if isinstance(value, str):
                    new_kwargs[key] = new_text
                    break
            return args, new_kwargs

        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                span, token = start_span(span_name)
                try:
                    text = _first_text_arg(args, kwargs)
                    if text is not None and input_guards:
                        sanitised = run_guards(text, input_guards, span=span, direction="input")
                        if sanitised != text:
                            args, kwargs = _replace_first_text_arg(args, kwargs, sanitised)
                    result = await func(*args, **kwargs)
                    if isinstance(result, str) and output_guards:
                        result = run_guards(result, output_guards, span=span, direction="output")
                    span.status = "ok"
                    return result
                except GuardrailViolation as exc:
                    span.status = "error"
                    span.attributes["error"] = str(exc)
                    raise
                except Exception as exc:
                    span.status = "error"
                    span.attributes["error"] = str(exc)
                    raise
                finally:
                    end_span(span, token)
                    export_span(span)

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            span, token = start_span(span_name)
            try:
                text = _first_text_arg(args, kwargs)
                if text is not None and input_guards:
                    sanitised = run_guards(text, input_guards, span=span, direction="input")
                    if sanitised != text:
                        args, kwargs = _replace_first_text_arg(args, kwargs, sanitised)
                result = func(*args, **kwargs)
                if isinstance(result, str) and output_guards:
                    result = run_guards(result, output_guards, span=span, direction="output")
                span.status = "ok"
                return result
            except GuardrailViolation as exc:
                span.status = "error"
                span.attributes["error"] = str(exc)
                raise
            except Exception as exc:
                span.status = "error"
                span.attributes["error"] = str(exc)
                raise
            finally:
                end_span(span, token)
                export_span(span)

        return sync_wrapper

    if _func is not None:
        return decorator(_func)
    return decorator
