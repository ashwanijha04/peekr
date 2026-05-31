"""Guardrails — synchronous, in-path enforcement rules for LLM calls.

Two categories run at different points in the exporter pipeline:

  Mutating guardrails (run BEFORE eval + storage)
    PIIRedact   — strip personal data from span attributes before persistence
    Blocklist   — redact or warn on forbidden keywords/patterns (action != "raise")

  Blocking guardrails (run AFTER storage)
    HallucinationBlock — raise GuardrailError when hallucination score < threshold
    Blocklist          — raise GuardrailError on forbidden terms (action == "raise")

Usage::

    import peekr

    peekr.instrument(
        guardrails=[
            peekr.guard.PIIRedact(),
            peekr.guard.Blocklist(terms=["confidential"], action="raise"),
            peekr.guard.Blocklist(patterns=peekr.guard.Blocklist.COMMON_SECRETS, action="redact"),
            peekr.guard.HallucinationBlock(threshold=0.4),
        ]
    )

    # Alongside evaluators — HallucinationBlock reuses the existing score,
    # so the judge LLM is only called once:
    peekr.instrument(
        evaluators=[peekr.eval.Hallucination(detailed=True)],
        guardrails=[peekr.guard.HallucinationBlock(threshold=0.4)],
    )
"""
from __future__ import annotations

import abc
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..span import Span

# GuardrailError lives in context.py so patches can import it without any
# circular-import risk.  Re-exported here for the public peekr.guard API.
from ..context import GuardrailError  # noqa: F401  (re-export)


# ── Base class ────────────────────────────────────────────────────────────────

class BaseGuardrail(abc.ABC):
    """Abstract base for all guardrails.

    Subclasses set ``_blocks = True`` to be placed in the post-storage
    (blocking) exporter; ``False`` (default) places them pre-storage.
    """

    _blocks: bool = False

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abc.abstractmethod
    def run(self, span: "Span") -> None:
        """Inspect / mutate span. Raise ``GuardrailError`` to block the call."""
        ...


# ── PII patterns ──────────────────────────────────────────────────────────────

_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email":       re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone":       re.compile(r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ssn":         re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"),
    "ip_address":  re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


class PIIRedact(BaseGuardrail):
    """Redact PII from span attributes before they reach any storage exporter.

    Scans ``span.attributes["input"]`` and ``span.attributes["output"]`` (or
    whichever ``fields`` you specify) and replaces matches with tokens like
    ``[EMAIL]``, ``[PHONE]``, etc.

    This does **not** prevent PII from reaching the LLM — that requires
    prompt-level filtering before the SDK call. What ``PIIRedact`` guarantees
    is that your observability data (traces.jsonl, SQLite, Peekr Cloud) never
    stores raw emails, phone numbers, SSNs, credit-card numbers, or IPs.

    Parameters
    ----------
    fields
        Span attributes to scan. Default: ``("input", "output")``.
    categories
        Subset of PII categories to detect. ``None`` = all.
        Options: ``"email"``, ``"phone"``, ``"ssn"``, ``"credit_card"``,
        ``"ip_address"``.

    Examples
    --------
    ::

        # Redact everything from both input and output
        peekr.instrument(guardrails=[peekr.guard.PIIRedact()])

        # Only redact emails and phone numbers, only from output
        peekr.instrument(guardrails=[peekr.guard.PIIRedact(
            fields=("output",),
            categories=("email", "phone"),
        )])
    """

    _blocks = False

    def __init__(
        self,
        fields: tuple[str, ...] = ("input", "output"),
        categories: tuple[str, ...] | None = None,
    ) -> None:
        self.fields = fields
        if categories is not None:
            unknown = set(categories) - set(_PII_PATTERNS)
            if unknown:
                raise ValueError(f"Unknown PII categories: {unknown}. "
                                 f"Valid: {set(_PII_PATTERNS)}")
            self._patterns = {k: v for k, v in _PII_PATTERNS.items() if k in categories}
        else:
            self._patterns = _PII_PATTERNS

    def run(self, span: "Span") -> None:
        redacted_fields: list[str] = []
        redacted_cats: set[str] = set()

        for field in self.fields:
            value = span.attributes.get(field)
            if not isinstance(value, str):
                continue
            original = value
            for cat, pattern in self._patterns.items():
                replaced, n = pattern.subn(f"[{cat.upper()}]", value)
                if n:
                    value = replaced
                    redacted_cats.add(cat)
            if value != original:
                span.attributes[field] = value
                redacted_fields.append(field)

        if redacted_fields:
            span.attributes.setdefault("guardrail_warnings", [])
            span.attributes["guardrail_warnings"].append(
                f"PIIRedact: [{', '.join(sorted(redacted_cats))}] "
                f"removed from [{', '.join(redacted_fields)}]"
            )


# ── Blocklist ─────────────────────────────────────────────────────────────────

class Blocklist(BaseGuardrail):
    """Block, redact, or warn when forbidden terms or patterns appear in spans.

    Three actions — which also determines the pipeline phase:

    * ``"raise"``  — raises ``GuardrailError`` (post-storage, blocking).
                     The LLM call has already been made; this prevents the
                     response from reaching application code and records the
                     violation. To stop secrets from reaching the LLM at all,
                     filter your prompts before constructing them.
    * ``"redact"`` — replaces matches with ``[BLOCKED]`` before storage
                     (pre-storage, mutating). Useful for sanitising API keys
                     or internal codenames from stored traces.
    * ``"warn"``   — records to ``guardrail_warnings`` without modifying
                     the span (pre-storage, mutating). Good for auditing
                     without disrupting the pipeline.

    Parameters
    ----------
    terms
        Exact strings to match (case-insensitive by default).
    patterns
        Regular-expression strings. Use ``Blocklist.COMMON_SECRETS`` for a
        pre-built set that catches OpenAI / Anthropic / GitHub key formats,
        Bearer tokens, and private-key headers.
    action
        ``"raise"`` | ``"redact"`` | ``"warn"``. Default ``"raise"``.
    fields
        Span attributes to scan. Default: ``("input", "output")``.
    case_sensitive
        When ``True``, term matching is case-sensitive. Default ``False``.

    Examples
    --------
    ::

        # Raise if "confidential" appears anywhere in input or output
        Blocklist(terms=["confidential", "internal only"], action="raise")

        # Redact common API key patterns from stored traces
        Blocklist(patterns=Blocklist.COMMON_SECRETS, action="redact")

        # Only scan inputs, case-sensitive
        Blocklist(terms=["SECRET"], fields=("input",), case_sensitive=True)
    """

    # Pre-built patterns for common API key / secret formats
    COMMON_SECRETS: list[str] = [
        r"sk-[A-Za-z0-9\-]{20,}",                   # OpenAI API keys (sk-... and sk-proj-...)
        r"sk-ant-api\d{2}-[A-Za-z0-9\-]{93,}",     # Anthropic API keys
        r"AIza[A-Za-z0-9\-_]{35}",                  # Google / Gemini API keys
        r"ghp_[A-Za-z0-9]{36}",                     # GitHub personal access tokens
        r"ghs_[A-Za-z0-9]{36}",                     # GitHub Actions tokens
        r"xoxb-[0-9]+-[A-Za-z0-9\-]+",             # Slack bot tokens
        r"-----BEGIN [A-Z ]+ PRIVATE KEY-----",      # PEM private keys
        r"(?i)bearer\s+[A-Za-z0-9\-._~+/]{20,}",   # Generic Bearer tokens
    ]

    def __init__(
        self,
        terms: list[str] | None = None,
        patterns: list[str] | None = None,
        action: str = "raise",
        fields: tuple[str, ...] = ("input", "output"),
        case_sensitive: bool = False,
    ) -> None:
        if action not in ("raise", "redact", "warn"):
            raise ValueError(f"action must be 'raise', 'redact', or 'warn'; got {action!r}")
        if not terms and not patterns:
            raise ValueError("Blocklist requires at least one of: terms, patterns")

        self.action = action
        self.fields = fields
        self.case_sensitive = case_sensitive

        # _blocks is instance-level: "raise" → post-storage; others → pre-storage
        self._blocks = (action == "raise")
        # _input_guard: True when this guard scans inputs and should run
        # PRE-CALL (before the LLM API is invoked) rather than post-storage.
        # Pre-call guards are registered via register_input_guard() and are
        # excluded from _BlockingGuardrailExporter to avoid double-firing.
        self._input_guard = (action == "raise" and "input" in fields)

        flags = 0 if case_sensitive else re.IGNORECASE
        compiled: list[re.Pattern[str]] = []
        for term in (terms or []):
            compiled.append(re.compile(re.escape(term), flags))
        for pat in (patterns or []):
            compiled.append(re.compile(pat, flags))
        self._patterns = compiled

    def run(self, span: "Span") -> None:
        for field in self.fields:
            value = span.attributes.get(field)
            if not isinstance(value, str):
                continue

            matches = [p for p in self._patterns if p.search(value)]
            if not matches:
                continue

            match_count = len(matches)

            if self.action == "redact":
                for pat in matches:
                    value = pat.sub("[BLOCKED]", value)
                span.attributes[field] = value
                span.attributes.setdefault("guardrail_warnings", [])
                span.attributes["guardrail_warnings"].append(
                    f"Blocklist: redacted {match_count} pattern(s) from {field!r}"
                )

            elif self.action == "warn":
                span.attributes.setdefault("guardrail_warnings", [])
                span.attributes["guardrail_warnings"].append(
                    f"Blocklist: {match_count} blocked pattern(s) detected in {field!r}"
                )

            elif self.action == "raise":
                span.attributes.setdefault("guardrail_violations", [])
                span.attributes["guardrail_violations"].append(
                    f"Blocklist: blocked pattern in {field!r}"
                )
                raise GuardrailError(
                    f"Blocked: forbidden term detected in {field!r}",
                    guardrail_name="Blocklist",
                    span=span,
                )


# ── LLM span prefixes (kept in sync with eval/__init__.py) ───────────────────

_LLM_PREFIXES = ("openai.", "anthropic.", "bedrock.", "gemini.")


class HallucinationBlock(BaseGuardrail):
    """Raise ``GuardrailError`` when a response's hallucination score is too low.

    Runs **after** storage exporters so the violation is always persisted before
    the error propagates to the caller — giving you a full audit trail of every
    blocked response.

    If ``EvalExporter`` already ran ``Hallucination`` (i.e. you also set
    ``evaluators=[peekr.eval.Hallucination(...)]``), the existing score is
    reused and no second judge call is made.

    Parameters
    ----------
    threshold
        Minimum acceptable score (0.0–1.0). Responses that score *below* this
        value raise ``GuardrailError``. Default ``0.5``.
    detailed
        When running standalone (no EvalExporter), use claim-level RAGAS
        decomposition. More accurate, costs one extra LLM call per span.

    Examples
    --------
    ::

        # Block any response that's less than 40% grounded
        peekr.instrument(guardrails=[peekr.guard.HallucinationBlock(threshold=0.4)])

        # Shared eval — EvalExporter scores it, HallucinationBlock enforces it
        peekr.instrument(
            evaluators=[peekr.eval.Hallucination(detailed=True)],
            guardrails=[peekr.guard.HallucinationBlock(threshold=0.4)],
        )
    """

    _blocks = True

    def __init__(self, threshold: float = 0.5, detailed: bool = False) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0.0, 1.0], got {threshold!r}")
        self.threshold = threshold
        self.detailed = detailed
        self._evaluator = None  # lazy — avoids import cycle at module load time

    def _get_evaluator(self):
        if self._evaluator is None:
            from ..eval.hallucination import Hallucination
            self._evaluator = Hallucination(detailed=self.detailed)
        return self._evaluator

    def run(self, span: "Span") -> None:
        if not any(span.name.startswith(p) for p in _LLM_PREFIXES):
            return

        # Reuse score from EvalExporter when available — avoids a second judge call
        existing = (span.attributes.get("eval_scores") or {}).get("Hallucination")
        if existing is not None:
            score = float(existing)
        else:
            try:
                score = self._get_evaluator().evaluate(span)
                # Store so it shows up in the dashboard even without EvalExporter
                span.attributes.setdefault("eval_scores", {})
                span.attributes["eval_scores"]["Hallucination"] = score
            except Exception as exc:
                # Infra failure (no API key, rate limit, network) — warn, never block
                span.attributes.setdefault("guardrail_warnings", [])
                span.attributes["guardrail_warnings"].append(
                    f"HallucinationBlock: evaluator failed "
                    f"({type(exc).__name__}: {exc}) — not blocking"
                )
                return

        if score < self.threshold:
            msg = (
                f"Response blocked: hallucination score {score:.3f} "
                f"is below threshold {self.threshold:.3f}"
            )
            span.attributes.setdefault("guardrail_violations", [])
            span.attributes["guardrail_violations"].append(
                f"HallucinationBlock(score={score:.3f} < threshold={self.threshold})"
            )
            raise GuardrailError(msg, guardrail_name="HallucinationBlock", span=span)

        # Passed — record for auditability
        span.attributes.setdefault("guardrail_warnings", [])
        span.attributes["guardrail_warnings"].append(
            f"HallucinationBlock: passed (score={score:.3f}, threshold={self.threshold})"
        )


# ── Internal exporters — used by instrument(), not part of public API ─────────

class _MutatingGuardrailExporter:
    """Runs non-blocking guardrails that modify span attributes before storage.

    Placed before EvalExporter + storage so eval scores are computed on
    already-redacted content and nothing with PII is ever written to disk.
    """

    def __init__(self, guardrails: list[BaseGuardrail]) -> None:
        self.guardrails = [g for g in guardrails if not g._blocks]

    def export(self, span: "Span") -> None:
        for guard in self.guardrails:
            try:
                guard.run(span)
            except GuardrailError:
                # A non-blocking guardrail raising is a programming error;
                # degrade gracefully rather than breaking tracing.
                span.attributes.setdefault("guardrail_warnings", [])
                span.attributes["guardrail_warnings"].append(
                    f"{guard.name}: unexpected GuardrailError in mutating guardrail — ignored"
                )
            except Exception:
                pass  # guardrail failure must never break tracing


class _BlockingGuardrailExporter:
    """Runs blocking guardrails AFTER storage exporters.

    All guards run even if one raises — so every violation is recorded on the
    span. Only the first GuardrailError is re-raised after the loop.

    Because the SDK patches call ``export_span`` inside a ``finally`` block,
    raising here cancels the LLM response return value — the caller receives
    ``GuardrailError`` instead of the model output.

    Input guards (``_input_guard=True``) are excluded: they already fired
    pre-call via ``_run_input_guards`` and must not double-fire here.
    """

    def __init__(self, guardrails: list[BaseGuardrail]) -> None:
        self.guardrails = [
            g for g in guardrails
            if g._blocks and not getattr(g, "_input_guard", False)
        ]

    def export(self, span: "Span") -> None:
        first_error: GuardrailError | None = None
        for guard in self.guardrails:
            try:
                guard.run(span)
            except GuardrailError as exc:
                if first_error is None:
                    first_error = exc
            except Exception:
                pass  # infra failure — never block on it
        if first_error is not None:
            raise first_error


__all__ = [
    "GuardrailError",
    "BaseGuardrail",
    "PIIRedact",
    "Blocklist",
    "HallucinationBlock",
    "_MutatingGuardrailExporter",
    "_BlockingGuardrailExporter",
]
