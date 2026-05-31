"""Tests for peekr.guard — PIIRedact and HallucinationBlock."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from peekr.span import Span
from peekr.guard import (
    GuardrailError,
    PIIRedact,
    HallucinationBlock,
    _MutatingGuardrailExporter,
    _BlockingGuardrailExporter,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _llm_span(input_text="", output_text="", eval_scores=None) -> Span:
    span = Span(name="openai.chat.completions", trace_id="trace-test")
    span.attributes["input"] = input_text
    span.attributes["output"] = output_text
    if eval_scores:
        span.attributes["eval_scores"] = eval_scores
    return span


def _tool_span() -> Span:
    return Span(name="tool.search_web", trace_id="trace-test")


# ── PIIRedact ─────────────────────────────────────────────────────────────────

class TestPIIRedact:

    def test_redacts_email_from_output(self):
        guard = PIIRedact()
        span = _llm_span(output_text="Contact us at alice@example.com for help.")
        guard.run(span)
        assert "alice@example.com" not in span.attributes["output"]
        assert "[EMAIL]" in span.attributes["output"]

    def test_redacts_phone_from_output(self):
        guard = PIIRedact()
        span = _llm_span(output_text="Call 555-867-5309 to book.")
        guard.run(span)
        assert "867-5309" not in span.attributes["output"]
        assert "[PHONE]" in span.attributes["output"]

    def test_redacts_ssn(self):
        guard = PIIRedact()
        span = _llm_span(output_text="SSN is 123-45-6789.")
        guard.run(span)
        assert "123-45-6789" not in span.attributes["output"]
        assert "[SSN]" in span.attributes["output"]

    def test_redacts_credit_card(self):
        guard = PIIRedact()
        span = _llm_span(output_text="Card: 4111 1111 1111 1111.")
        guard.run(span)
        assert "4111" not in span.attributes["output"]
        assert "[CREDIT_CARD]" in span.attributes["output"]

    def test_redacts_from_input_field(self):
        guard = PIIRedact(fields=("input",))
        span = _llm_span(input_text="My email is bob@test.com", output_text="bob@test.com")
        guard.run(span)
        assert "[EMAIL]" in span.attributes["input"]
        assert "bob@test.com" in span.attributes["output"]  # output not touched

    def test_category_filter(self):
        guard = PIIRedact(categories=("email",))
        span = _llm_span(output_text="Email: a@b.com Phone: 555-123-4567")
        guard.run(span)
        assert "[EMAIL]" in span.attributes["output"]
        assert "555-123-4567" in span.attributes["output"]  # phone not redacted

    def test_no_pii_no_warning(self):
        guard = PIIRedact()
        span = _llm_span(output_text="The answer is 42.")
        guard.run(span)
        assert "guardrail_warnings" not in span.attributes

    def test_warning_recorded_on_redaction(self):
        guard = PIIRedact()
        span = _llm_span(output_text="Email: x@y.com")
        guard.run(span)
        assert "guardrail_warnings" in span.attributes
        assert any("PIIRedact" in w for w in span.attributes["guardrail_warnings"])

    def test_skips_non_string_attributes(self):
        guard = PIIRedact()
        span = _llm_span()
        span.attributes["output"] = 42  # not a string
        guard.run(span)  # should not raise

    def test_invalid_category_raises(self):
        with pytest.raises(ValueError, match="Unknown PII categories"):
            PIIRedact(categories=("banana",))

    def test_does_not_block(self):
        assert PIIRedact._blocks is False

    def test_tool_span_processed(self):
        # PIIRedact doesn't filter by span name — it scans whatever fields exist
        guard = PIIRedact()
        span = _tool_span()
        span.attributes["input"] = "user email: a@b.com"
        guard.run(span)
        assert "[EMAIL]" in span.attributes["input"]


# ── HallucinationBlock ────────────────────────────────────────────────────────

class TestHallucinationBlock:

    def test_raises_when_score_below_threshold(self):
        guard = HallucinationBlock(threshold=0.5)
        span = _llm_span(eval_scores={"Hallucination": 0.3})
        with pytest.raises(GuardrailError) as exc_info:
            guard.run(span)
        assert "0.300" in str(exc_info.value)
        assert exc_info.value.guardrail_name == "HallucinationBlock"

    def test_passes_when_score_at_threshold(self):
        guard = HallucinationBlock(threshold=0.5)
        span = _llm_span(eval_scores={"Hallucination": 0.5})
        guard.run(span)  # must not raise

    def test_passes_when_score_above_threshold(self):
        guard = HallucinationBlock(threshold=0.5)
        span = _llm_span(eval_scores={"Hallucination": 0.9})
        guard.run(span)  # must not raise

    def test_violation_recorded_on_span(self):
        guard = HallucinationBlock(threshold=0.5)
        span = _llm_span(eval_scores={"Hallucination": 0.2})
        with pytest.raises(GuardrailError):
            guard.run(span)
        assert "guardrail_violations" in span.attributes
        assert any("HallucinationBlock" in v for v in span.attributes["guardrail_violations"])

    def test_reuses_existing_eval_score(self):
        guard = HallucinationBlock(threshold=0.5)
        span = _llm_span(eval_scores={"Hallucination": 0.9})
        # _get_evaluator should never be called since score already exists
        with patch.object(guard, "_get_evaluator") as mock_eval:
            guard.run(span)
            mock_eval.assert_not_called()

    def test_runs_evaluator_when_no_existing_score(self):
        guard = HallucinationBlock(threshold=0.5)
        span = _llm_span(
            input_text="The sky is blue.",
            output_text="The sky is blue.",
        )
        mock_eval = MagicMock()
        mock_eval.evaluate.return_value = 0.95
        guard._evaluator = mock_eval
        guard.run(span)  # should not raise
        mock_eval.evaluate.assert_called_once_with(span)

    def test_eval_score_stored_on_span_when_running_standalone(self):
        guard = HallucinationBlock(threshold=0.5)
        span = _llm_span(input_text="ctx", output_text="answer")
        mock_eval = MagicMock()
        mock_eval.evaluate.return_value = 0.8
        guard._evaluator = mock_eval
        guard.run(span)
        assert span.attributes["eval_scores"]["Hallucination"] == 0.8

    def test_warn_not_block_on_evaluator_failure(self):
        guard = HallucinationBlock(threshold=0.5)
        span = _llm_span(input_text="x", output_text="y")
        mock_eval = MagicMock()
        mock_eval.evaluate.side_effect = RuntimeError("no API key")
        guard._evaluator = mock_eval
        guard.run(span)  # must not raise
        assert "guardrail_warnings" in span.attributes
        assert any("evaluator failed" in w for w in span.attributes["guardrail_warnings"])

    def test_skips_non_llm_spans(self):
        guard = HallucinationBlock(threshold=0.5)
        span = _tool_span()
        guard.run(span)  # must not raise

    def test_invalid_threshold_raises(self):
        with pytest.raises(ValueError):
            HallucinationBlock(threshold=1.5)
        with pytest.raises(ValueError):
            HallucinationBlock(threshold=-0.1)

    def test_blocks_flag(self):
        assert HallucinationBlock._blocks is True


# ── Exporters ─────────────────────────────────────────────────────────────────

class TestMutatingGuardrailExporter:

    def test_runs_non_blocking_guardrails(self):
        guard = PIIRedact()
        exp = _MutatingGuardrailExporter([guard])
        span = _llm_span(output_text="Email: x@y.com")
        exp.export(span)
        assert "[EMAIL]" in span.attributes["output"]

    def test_excludes_blocking_guardrails(self):
        block_guard = HallucinationBlock(threshold=0.5)
        exp = _MutatingGuardrailExporter([block_guard])
        assert len(exp.guardrails) == 0

    def test_guardrail_exception_does_not_propagate(self):
        bad_guard = MagicMock(spec=PIIRedact)
        bad_guard._blocks = False
        bad_guard.name = "BadGuard"
        bad_guard.run.side_effect = RuntimeError("crash")
        exp = _MutatingGuardrailExporter([bad_guard])
        span = _llm_span()
        exp.export(span)  # must not raise


class TestBlockingGuardrailExporter:

    def test_raises_guardrail_error(self):
        guard = HallucinationBlock(threshold=0.5)
        exp = _BlockingGuardrailExporter([guard])
        span = _llm_span(eval_scores={"Hallucination": 0.1})
        with pytest.raises(GuardrailError):
            exp.export(span)

    def test_excludes_mutating_guardrails(self):
        pii = PIIRedact()
        exp = _BlockingGuardrailExporter([pii])
        assert len(exp.guardrails) == 0

    def test_all_guards_run_before_raising(self):
        # Two blocking guards — both should record violations before first error propagates
        g1 = MagicMock(spec=HallucinationBlock)
        g1._blocks = True
        g1.name = "Guard1"
        g1.run.side_effect = GuardrailError("first", guardrail_name="Guard1")

        g2 = MagicMock(spec=HallucinationBlock)
        g2._blocks = True
        g2.name = "Guard2"
        g2.run.side_effect = GuardrailError("second", guardrail_name="Guard2")

        exp = _BlockingGuardrailExporter([g1, g2])
        span = _llm_span()
        with pytest.raises(GuardrailError) as exc_info:
            exp.export(span)
        # First error propagates
        assert "first" in str(exc_info.value)
        # But second guard still ran
        g2.run.assert_called_once()

    def test_infra_exception_does_not_block(self):
        bad_guard = MagicMock(spec=HallucinationBlock)
        bad_guard._blocks = True
        bad_guard.name = "BadGuard"
        bad_guard.run.side_effect = RuntimeError("crash")
        exp = _BlockingGuardrailExporter([bad_guard])
        span = _llm_span()
        exp.export(span)  # RuntimeError swallowed, must not propagate


# ── Integration: both guardrail types together ────────────────────────────────

class TestGuardrailIntegration:

    def test_pii_redacted_before_hallucination_check(self):
        # PIIRedact runs first; HallucinationBlock sees the redacted text
        pii = PIIRedact()
        block = HallucinationBlock(threshold=0.5)

        mut_exp = _MutatingGuardrailExporter([pii, block])
        blk_exp = _BlockingGuardrailExporter([pii, block])

        span = _llm_span(
            output_text="Contact alice@example.com for help.",
            eval_scores={"Hallucination": 0.9},
        )

        mut_exp.export(span)
        assert "[EMAIL]" in span.attributes["output"]

        blk_exp.export(span)  # score is 0.9, should pass

    def test_end_to_end_block_after_pii_redact(self):
        pii = PIIRedact()
        block = HallucinationBlock(threshold=0.5)

        mut_exp = _MutatingGuardrailExporter([pii, block])
        blk_exp = _BlockingGuardrailExporter([pii, block])

        span = _llm_span(
            output_text="Email: z@z.com. The Eiffel Tower was built in 1923.",
            eval_scores={"Hallucination": 0.1},
        )

        mut_exp.export(span)
        assert "[EMAIL]" in span.attributes["output"]  # PII redacted

        with pytest.raises(GuardrailError):
            blk_exp.export(span)  # hallucination blocked

        # Violation is recorded on the span for audit
        assert "guardrail_violations" in span.attributes
