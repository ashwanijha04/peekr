"""Peekr Guardrails — composable input/output checks for LLM agents.

Three usage patterns:

    # 1. As a decorator on your own functions
    from peekr.guardrails import guard, PII, PromptInjection
    @guard(input=[PromptInjection()], output=[PII(action="redact")])
    def chat(user_message: str) -> str: ...

    # 2. Globally on every LLM call via instrument()
    import peekr
    from peekr.guardrails import PII, PromptInjection
    peekr.instrument(guards=[PromptInjection(action="block"), PII(action="redact")])

    # 3. Programmatically
    result = PII().check("Email me at foo@example.com")
    if not result.passed:
        print(result.findings, result.redacted)

Available checks: PII, Secrets, PromptInjection, Toxicity, Regex, MaxLength,
JSONSchema, AllowList, DenyList, NoURLs, NoCode.

Each check has an `action`: "warn" (default), "redact", or "block". `block`
raises ``GuardrailViolation``; `redact` returns sanitised text; `warn` records
the finding on the span without changing the value.
"""
from __future__ import annotations

from .base import (
    Guardrail,
    GuardrailResult,
    GuardrailViolation,
    GuardrailExporter,
    guard,
    run_guards,
)
from .checks import (
    PII,
    Secrets,
    PromptInjection,
    Toxicity,
    Regex,
    MaxLength,
    JSONSchema,
    AllowList,
    DenyList,
    NoURLs,
    NoCode,
)

__all__ = [
    "Guardrail",
    "GuardrailResult",
    "GuardrailViolation",
    "GuardrailExporter",
    "guard",
    "run_guards",
    "PII",
    "Secrets",
    "PromptInjection",
    "Toxicity",
    "Regex",
    "MaxLength",
    "JSONSchema",
    "AllowList",
    "DenyList",
    "NoURLs",
    "NoCode",
]
