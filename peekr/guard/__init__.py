"""Guardrails — synchronous, in-path enforcement rules for LLM calls.

Two categories run at different points in the exporter pipeline:

  Mutating guardrails (run BEFORE eval + storage)
    PIIRedact   — strip personal data from span attributes before persistence

  Blocking guardrails (run AFTER storage)
    HallucinationBlock — raise GuardrailError when hallucination score < threshold

Usage::

    import peekr

    peekr.instrument(
        guardrails=[
            peekr.guard.PIIRedact(),
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


# ── Public exception ──────────────────────────────────────────────────────────

class GuardrailError(Exception):
    """Raised by a blocking guardrail when the LLM response violates a rule.

    Because the patches call ``export_span`` inside a ``finally`` block, this
    exception propagates to the caller of the LLM SDK — cancelling the return
    value so the bad response is never consumed by application code.
    """

    def __init__(
        self,
        message: str,
        guardrail_name: str = "",
        span: "Span | None" = None,
    ) -> None:
        super().__init__(message)
        self.guardrail_name = guardrail_name
        self.span = span


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
    """

    def __init__(self, guardrails: list[BaseGuardrail]) -> None:
        self.guardrails = [g for g in guardrails if g._blocks]

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
    "HallucinationBlock",
    "_MutatingGuardrailExporter",
    "_BlockingGuardrailExporter",
]
