"""Concrete guardrail implementations.

All checks are dependency-free (pure stdlib) so they work in any Python 3.9+
environment. They are intentionally conservative: heuristic detectors flag
the obvious cases and stay out of the way otherwise. For production-grade
detection, plug in your own ``Guardrail`` subclass — the base class is small.
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterable, Optional, Pattern

from .base import Guardrail, GuardrailResult


# ─────────────────────────────────────────────────────────────────────────────
# PII
# ─────────────────────────────────────────────────────────────────────────────
_PII_PATTERNS: dict[str, Pattern[str]] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(
        r"(?<!\d)(?:\+?\d{1,2}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\d)"
    ),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


def _luhn_ok(digits: str) -> bool:
    nums = [int(c) for c in digits if c.isdigit()]
    if len(nums) < 13 or len(nums) > 19:
        return False
    total = 0
    parity = len(nums) % 2
    for i, n in enumerate(nums):
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


class PII(Guardrail):
    """Detects common PII: emails, phone numbers, SSNs, credit cards, IPv4.

    Credit-card matches are validated with the Luhn checksum to reduce
    false positives from arbitrary 13–19 digit numbers.

    Parameters
    ----------
    types:
        Restrict to a subset, e.g. ``PII(types=["email", "ssn"])``.
    action:
        ``"warn"`` records findings; ``"redact"`` replaces matches with
        ``[REDACTED_<TYPE>]``; ``"block"`` raises ``GuardrailViolation``.
    """

    def __init__(
        self,
        types: Optional[Iterable[str]] = None,
        action: str = "warn",
    ) -> None:
        super().__init__(action=action)
        if types is not None:
            unknown = set(types) - set(_PII_PATTERNS)
            if unknown:
                raise ValueError(f"Unknown PII types: {sorted(unknown)}")
            self.types = list(types)
        else:
            self.types = list(_PII_PATTERNS)

    def _check(self, text: str) -> GuardrailResult:
        findings: list[dict] = []
        redacted = text
        for kind in self.types:
            pattern = _PII_PATTERNS[kind]
            for match in pattern.finditer(text):
                value = match.group(0)
                if kind == "credit_card" and not _luhn_ok(value):
                    continue
                findings.append({"type": kind, "match": value, "span": match.span()})
                redacted = redacted.replace(value, f"[REDACTED_{kind.upper()}]")
        return GuardrailResult(
            name="PII",
            passed=not findings,
            findings=findings,
            redacted=redacted if findings else None,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Secrets
# ─────────────────────────────────────────────────────────────────────────────
_SECRET_PATTERNS: dict[str, Pattern[str]] = {
    "openai_key": re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{20,}"),
    "anthropic_key": re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    "slack_token": re.compile(r"xox[abprs]-[A-Za-z0-9-]{10,}"),
    "stripe_key": re.compile(r"\b(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{20,}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"),
}


class Secrets(Guardrail):
    """Detects common API keys, tokens, and private keys in text.

    Matches:
        OpenAI, Anthropic, AWS, GitHub, Google API, Slack, Stripe,
        JWTs, and PEM private-key headers.
    """

    def _check(self, text: str) -> GuardrailResult:
        findings: list[dict] = []
        redacted = text
        for kind, pattern in _SECRET_PATTERNS.items():
            for match in pattern.finditer(text):
                value = match.group(0)
                findings.append({"type": kind, "match": value[:8] + "…"})
                redacted = redacted.replace(value, f"[REDACTED_{kind.upper()}]")
        return GuardrailResult(
            name="Secrets",
            passed=not findings,
            findings=findings,
            redacted=redacted if findings else None,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Prompt injection
# ─────────────────────────────────────────────────────────────────────────────
# Heuristic patterns drawn from published prompt-injection attacks. Not a
# replacement for a dedicated model — a fast first line of defence that
# catches the most common variants and runs in microseconds.
_INJECTION_PATTERNS: list[tuple[str, Pattern[str]]] = [
    ("ignore_previous", re.compile(
        r"\b(?:ignore|disregard|forget|override)\b[\s\S]{0,40}?\b"
        r"(?:previous|prior|earlier|above|all)\b[\s\S]{0,40}?\b"
        r"(?:instructions?|prompts?|rules?|system|directives?)\b",
        re.IGNORECASE,
    )),
    ("role_override", re.compile(
        r"\byou\s+are\s+now\b|\bact\s+as\b|\bpretend\s+to\s+be\b|"
        r"\bfrom\s+now\s+on\b[\s\S]{0,40}?\b(?:you|assistant|system)\b",
        re.IGNORECASE,
    )),
    ("system_prompt_leak", re.compile(
        r"\b(?:print|reveal|repeat|show|output|tell\s+me|what\s+(?:is|are|were))\b"
        r"[\s\S]{0,40}?\b(?:system\s+prompt|initial\s+(?:prompt|instructions?)|"
        r"hidden\s+instructions?|your\s+instructions?)\b",
        re.IGNORECASE,
    )),
    ("developer_mode", re.compile(
        r"\b(?:developer|dan|jailbreak|god|sudo|admin)\s+mode\b",
        re.IGNORECASE,
    )),
    ("delimiter_break", re.compile(
        r"</?(?:system|instructions?|user|assistant)>|"
        r"\[\[?(?:system|instructions?|admin)\]?\]|"
        r"```\s*system",
        re.IGNORECASE,
    )),
    ("unicode_hidden", re.compile(r"[​-‏ - ﻿]")),
]


class PromptInjection(Guardrail):
    """Heuristic prompt-injection detector.

    Flags the common attack families:
      - "ignore previous instructions"
      - role overrides ("you are now X", "act as Y")
      - system-prompt leak attempts
      - developer/jailbreak mode claims
      - delimiter / fake-tag injection
      - zero-width and bidirectional Unicode characters

    Fast and dependency-free; pair with an LLM-as-judge check for adversarial
    settings.
    """

    def _check(self, text: str) -> GuardrailResult:
        findings: list[dict] = []
        for kind, pattern in _INJECTION_PATTERNS:
            for match in pattern.finditer(text):
                findings.append({"type": kind, "match": match.group(0)[:80]})
        return GuardrailResult(
            name="PromptInjection",
            passed=not findings,
            findings=findings,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Toxicity (heuristic)
# ─────────────────────────────────────────────────────────────────────────────
_TOXIC_WORDS = {
    # Slurs/insults excluded by design — too many false positives without context.
    # This list focuses on the most explicit profanity classes; extend via
    # `Toxicity(extra_words=[...])` for project-specific vocab.
    "fuck", "shit", "asshole", "bitch", "bastard", "cunt", "dick",
    "motherfucker", "damn", "piss",
}


class Toxicity(Guardrail):
    """Heuristic profanity / toxicity check using a configurable wordlist.

    For real toxicity detection use a model (Detoxify, Perspective, OpenAI
    moderation). This check is a fast first-pass that catches obvious cases
    and is easy to extend per-project.

    Parameters
    ----------
    extra_words:
        Additional terms to flag (case-insensitive, whole-word match).
    threshold:
        Minimum match count to fail (default 1).
    """

    def __init__(
        self,
        extra_words: Optional[Iterable[str]] = None,
        threshold: int = 1,
        action: str = "warn",
    ) -> None:
        super().__init__(action=action)
        self.words = set(w.lower() for w in _TOXIC_WORDS)
        if extra_words:
            self.words.update(w.lower() for w in extra_words)
        self.threshold = threshold
        self._pattern = re.compile(
            r"\b(" + "|".join(re.escape(w) for w in self.words) + r")\b",
            re.IGNORECASE,
        )

    def _check(self, text: str) -> GuardrailResult:
        matches = self._pattern.findall(text)
        findings = [{"type": "profanity", "match": m.lower()} for m in matches]
        passed = len(matches) < self.threshold
        redacted = self._pattern.sub("[REDACTED]", text) if matches else None
        return GuardrailResult(
            name="Toxicity",
            passed=passed,
            findings=findings,
            redacted=redacted,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Regex / patterns
# ─────────────────────────────────────────────────────────────────────────────
class Regex(Guardrail):
    """Fail when a regex pattern matches the text.

    Use for custom deny patterns — e.g. internal product names, customer
    identifiers, etc.

    Parameters
    ----------
    pattern:
        Regex string or compiled ``re.Pattern``.
    flags:
        Regex flags (ignored when ``pattern`` is precompiled).
    """

    def __init__(self, pattern, flags: int = 0, action: str = "warn") -> None:
        super().__init__(action=action)
        if isinstance(pattern, str):
            self._regex = re.compile(pattern, flags)
        else:
            self._regex = pattern
        self._pattern_str = self._regex.pattern

    @property
    def name(self) -> str:
        return f"Regex({self._pattern_str[:30]})"

    def _check(self, text: str) -> GuardrailResult:
        matches = list(self._regex.finditer(text))
        findings = [
            {"type": "regex", "match": m.group(0)[:80], "span": m.span()}
            for m in matches
        ]
        redacted = self._regex.sub("[REDACTED]", text) if matches else None
        return GuardrailResult(
            name=self.name,
            passed=not matches,
            findings=findings,
            redacted=redacted,
        )


class MaxLength(Guardrail):
    """Fail when text exceeds ``limit`` characters.

    Useful as an output guardrail to keep responses bounded, or as an input
    guardrail to reject pathological prompts before they hit the LLM.
    """

    def __init__(self, limit: int, action: str = "warn") -> None:
        super().__init__(action=action)
        if limit <= 0:
            raise ValueError("limit must be positive")
        self.limit = limit

    def _check(self, text: str) -> GuardrailResult:
        if len(text) <= self.limit:
            return GuardrailResult(name="MaxLength", passed=True)
        return GuardrailResult(
            name="MaxLength",
            passed=False,
            findings=[{"type": "length", "actual": len(text), "limit": self.limit}],
            redacted=text[: self.limit],
        )


class AllowList(Guardrail):
    """Fail when text contains any term *outside* the allow list.

    More commonly useful as a topical filter — e.g. AllowList(["billing",
    "refund", "subscription"]) to keep a support agent on-topic. Matches
    if at least one allow-list term appears.
    """

    def __init__(self, terms: Iterable[str], action: str = "warn") -> None:
        super().__init__(action=action)
        self.terms = [t.lower() for t in terms]
        if not self.terms:
            raise ValueError("AllowList requires at least one term")

    def _check(self, text: str) -> GuardrailResult:
        lower = text.lower()
        if any(t in lower for t in self.terms):
            return GuardrailResult(name="AllowList", passed=True)
        return GuardrailResult(
            name="AllowList",
            passed=False,
            findings=[{"type": "off_topic", "allowed": self.terms}],
        )


class DenyList(Guardrail):
    """Fail when text contains any term from the deny list (case-insensitive)."""

    def __init__(self, terms: Iterable[str], action: str = "warn") -> None:
        super().__init__(action=action)
        self.terms = [t.lower() for t in terms]
        if not self.terms:
            raise ValueError("DenyList requires at least one term")

    def _check(self, text: str) -> GuardrailResult:
        lower = text.lower()
        hits = [t for t in self.terms if t in lower]
        if not hits:
            return GuardrailResult(name="DenyList", passed=True)
        return GuardrailResult(
            name="DenyList",
            passed=False,
            findings=[{"type": "denied", "match": h} for h in hits],
        )


class JSONSchema(Guardrail):
    """Validate that text parses as JSON and (optionally) matches a schema.

    The schema is a minimal subset of JSON-Schema:
      - "type": "object" | "array" | "string" | "number" | "boolean" | "null"
      - "required": [keys...]   (object only)
      - "properties": {key: schema}  (recursive, object only)
      - "items": schema  (array only — same type for every element)

    This keeps the check zero-dependency. For richer validation, plug in
    ``jsonschema`` via a custom Guardrail subclass.
    """

    def __init__(
        self,
        schema: Optional[dict[str, Any]] = None,
        action: str = "warn",
    ) -> None:
        super().__init__(action=action)
        self.schema = schema

    def _check(self, text: str) -> GuardrailResult:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            return GuardrailResult(
                name="JSONSchema",
                passed=False,
                findings=[{"type": "parse_error", "error": str(exc)}],
            )
        if self.schema is None:
            return GuardrailResult(name="JSONSchema", passed=True)
        errors: list[dict] = []
        _validate(data, self.schema, "$", errors)
        return GuardrailResult(
            name="JSONSchema",
            passed=not errors,
            findings=errors,
        )


_JSON_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "null": type(None),
}


def _validate(value: Any, schema: dict, path: str, errors: list[dict]) -> None:
    expected_type = schema.get("type")
    if expected_type and not isinstance(value, _JSON_TYPES.get(expected_type, object)):
        # Bool is a subclass of int — explicitly exclude when type is "number"/"integer".
        if expected_type in ("number", "integer") and isinstance(value, bool):
            errors.append({"path": path, "type": "wrong_type", "expected": expected_type})
            return
        errors.append({"path": path, "type": "wrong_type", "expected": expected_type})
        return
    if expected_type == "object" and isinstance(value, dict):
        for req in schema.get("required", []):
            if req not in value:
                errors.append({"path": f"{path}.{req}", "type": "missing"})
        for key, sub_schema in (schema.get("properties") or {}).items():
            if key in value:
                _validate(value[key], sub_schema, f"{path}.{key}", errors)
    elif expected_type == "array" and isinstance(value, list):
        sub_schema = schema.get("items")
        if sub_schema:
            for i, item in enumerate(value):
                _validate(item, sub_schema, f"{path}[{i}]", errors)


class NoURLs(Guardrail):
    """Flag any http:// or https:// URL in the text."""

    _URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)

    def _check(self, text: str) -> GuardrailResult:
        matches = list(self._URL_RE.finditer(text))
        findings = [{"type": "url", "match": m.group(0)} for m in matches]
        redacted = self._URL_RE.sub("[REDACTED_URL]", text) if matches else None
        return GuardrailResult(
            name="NoURLs",
            passed=not matches,
            findings=findings,
            redacted=redacted,
        )


class NoCode(Guardrail):
    """Flag fenced code blocks or obvious code-shaped content.

    Useful to keep customer-facing assistants from emitting code when they
    shouldn't, or to reject inputs containing executable snippets.
    """

    _PATTERNS = [
        re.compile(r"```[\s\S]+?```"),
        re.compile(r"\bimport\s+[A-Za-z_]\w*"),
        re.compile(r"\bdef\s+[A-Za-z_]\w*\s*\("),
        re.compile(r"<script[\s>]"),
        re.compile(r"SELECT\s+.+\s+FROM\s+", re.IGNORECASE),
    ]

    def _check(self, text: str) -> GuardrailResult:
        findings: list[dict] = []
        for pattern in self._PATTERNS:
            for match in pattern.finditer(text):
                findings.append({"type": "code", "match": match.group(0)[:80]})
        return GuardrailResult(
            name="NoCode",
            passed=not findings,
            findings=findings,
        )
