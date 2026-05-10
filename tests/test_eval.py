from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from peekr.span import Span
from peekr.eval import BaseEvaluator, EvalExporter, _in_eval
from peekr.eval.rubric import NotEmpty, NoError, Rubric


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_llm_span(name="openai.chat.completions", status="ok", output="hello", input_text="hi"):
    s = Span(name=name, trace_id="trace-1")
    s.attributes["input"] = input_text
    s.attributes["output"] = output
    s.status = status
    s.finish()
    return s


def make_non_llm_span():
    s = Span(name="my.custom.span", trace_id="trace-2")
    s.finish()
    return s


# ---------------------------------------------------------------------------
# NotEmpty
# ---------------------------------------------------------------------------

class TestNotEmpty:
    def test_non_empty_string_returns_one(self):
        span = make_llm_span(output="some response")
        assert NotEmpty().evaluate(span) == 1.0

    def test_empty_string_returns_zero(self):
        span = make_llm_span(output="")
        assert NotEmpty().evaluate(span) == 0.0

    def test_whitespace_only_returns_zero(self):
        span = make_llm_span(output="   ")
        assert NotEmpty().evaluate(span) == 0.0

    def test_missing_output_returns_zero(self):
        span = Span(name="openai.chat.completions", trace_id="t1")
        span.finish()
        assert NotEmpty().evaluate(span) == 0.0

    def test_non_string_output_returns_zero(self):
        span = make_llm_span(output="")
        span.attributes["output"] = 42  # non-string
        assert NotEmpty().evaluate(span) == 0.0

    def test_name(self):
        assert NotEmpty().name == "NotEmpty"


# ---------------------------------------------------------------------------
# NoError
# ---------------------------------------------------------------------------

class TestNoError:
    def test_ok_status_returns_one(self):
        span = make_llm_span(status="ok")
        assert NoError().evaluate(span) == 1.0

    def test_error_status_returns_zero(self):
        span = make_llm_span(status="error")
        assert NoError().evaluate(span) == 0.0

    def test_custom_status_returns_zero(self):
        span = make_llm_span(status="timeout")
        assert NoError().evaluate(span) == 0.0

    def test_name(self):
        assert NoError().name == "NoError"


# ---------------------------------------------------------------------------
# EvalExporter
# ---------------------------------------------------------------------------

class _ConstantEvaluator(BaseEvaluator):
    """Evaluator that always returns a fixed score — no LLM calls."""

    def __init__(self, score: float = 0.75, name_override: str = "ConstantEval"):
        self._score = score
        self._name = name_override

    @property
    def name(self) -> str:
        return self._name

    def evaluate(self, span: Span) -> float:
        return self._score


class TestEvalExporter:
    def test_scores_added_to_llm_span(self):
        evaluator = _ConstantEvaluator(score=0.9)
        exporter = EvalExporter(evaluators=[evaluator])
        span = make_llm_span()
        exporter.export(span)
        assert "eval_scores" in span.attributes
        assert span.attributes["eval_scores"]["ConstantEval"] == pytest.approx(0.9)

    def test_non_llm_span_is_skipped(self):
        evaluator = _ConstantEvaluator()
        exporter = EvalExporter(evaluators=[evaluator])
        span = make_non_llm_span()
        exporter.export(span)
        assert "eval_scores" not in span.attributes

    def test_anthropic_span_is_evaluated(self):
        evaluator = _ConstantEvaluator(score=1.0)
        exporter = EvalExporter(evaluators=[evaluator])
        span = make_llm_span(name="anthropic.messages")
        exporter.export(span)
        assert span.attributes["eval_scores"]["ConstantEval"] == pytest.approx(1.0)

    def test_bedrock_span_is_evaluated(self):
        evaluator = _ConstantEvaluator(score=0.5)
        exporter = EvalExporter(evaluators=[evaluator])
        span = make_llm_span(name="bedrock.invoke_model")
        exporter.export(span)
        assert span.attributes["eval_scores"]["ConstantEval"] == pytest.approx(0.5)

    def test_multiple_evaluators_all_run(self):
        e1 = _ConstantEvaluator(score=0.8, name_override="Eval1")
        e2 = _ConstantEvaluator(score=0.4, name_override="Eval2")
        exporter = EvalExporter(evaluators=[e1, e2])
        span = make_llm_span()
        exporter.export(span)
        scores = span.attributes["eval_scores"]
        assert scores["Eval1"] == pytest.approx(0.8)
        assert scores["Eval2"] == pytest.approx(0.4)

    def test_failing_evaluator_gets_zero(self):
        class BrokenEvaluator(BaseEvaluator):
            @property
            def name(self):
                return "BrokenEval"

            def evaluate(self, span):
                raise RuntimeError("boom")

        exporter = EvalExporter(evaluators=[BrokenEvaluator()])
        span = make_llm_span()
        exporter.export(span)
        assert span.attributes["eval_scores"]["BrokenEval"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _in_eval guard against recursion
# ---------------------------------------------------------------------------

class TestInEvalGuard:
    def test_evaluator_not_called_when_in_eval(self):
        call_count = 0

        class CountingEvaluator(BaseEvaluator):
            @property
            def name(self):
                return "Counting"

            def evaluate(self, span: Span) -> float:
                nonlocal call_count
                call_count += 1
                return 1.0

        exporter = EvalExporter(evaluators=[CountingEvaluator()])
        span = make_llm_span()

        # Simulate being inside an eval call already
        token = _in_eval.set(True)
        try:
            exporter.export(span)
        finally:
            _in_eval.reset(token)

        assert call_count == 0
        assert "eval_scores" not in span.attributes

    def test_in_eval_resets_after_export(self):
        """_in_eval must be False after a normal EvalExporter.export() call."""
        exporter = EvalExporter(evaluators=[_ConstantEvaluator()])
        span = make_llm_span()
        exporter.export(span)
        assert _in_eval.get() is False

    def test_in_eval_resets_even_on_exception(self):
        """_in_eval must be reset even if an unexpected error escapes the evaluator loop."""

        class AlwaysRaises(BaseEvaluator):
            @property
            def name(self):
                return "AlwaysRaises"

            def evaluate(self, span: Span) -> float:
                raise RuntimeError("unexpected")

        exporter = EvalExporter(evaluators=[AlwaysRaises()])
        span = make_llm_span()
        # Should not raise (errors are caught per-evaluator)
        exporter.export(span)
        assert _in_eval.get() is False


# ---------------------------------------------------------------------------
# Rubric (mocked — no real API calls)
# ---------------------------------------------------------------------------

class TestRubricMocked:
    def test_rubric_calls_openai_and_returns_float(self):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "0.85"

        with patch("peekr.eval.rubric.openai") as mock_openai:
            mock_openai.chat.completions.create.return_value = mock_response
            rubric = Rubric("Be concise and factually accurate")
            span = make_llm_span(output="The capital of France is Paris.")
            score = rubric.evaluate(span)

        assert score == pytest.approx(0.85)
        mock_openai.chat.completions.create.assert_called_once()

    def test_rubric_prompt_contains_criteria_and_output(self):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "0.9"
        criteria = "Be concise"
        output_text = "Short answer."

        with patch("peekr.eval.rubric.openai") as mock_openai:
            mock_openai.chat.completions.create.return_value = mock_response
            rubric = Rubric(criteria)
            span = make_llm_span(output=output_text)
            rubric.evaluate(span)

        call_kwargs = mock_openai.chat.completions.create.call_args
        messages = call_kwargs[1]["messages"] if call_kwargs[1] else call_kwargs[0][1]
        prompt_text = messages[0]["content"]
        assert criteria in prompt_text
        assert output_text in prompt_text

    def test_rubric_name_truncates_criteria(self):
        rubric = Rubric("Be very concise and factually accurate please")
        # name should start with "Rubric(" and truncate at 30 chars of criteria
        assert rubric.name.startswith("Rubric(")
        assert len(rubric.name) <= len("Rubric(") + 30 + 1  # +1 for closing paren

    def test_rubric_falls_back_to_anthropic_when_no_openai(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="0.7")]

        # Patch openai to None at the module level so Rubric skips it,
        # and provide a mock anthropic client.
        with patch("peekr.eval.rubric.openai", None):
            with patch("peekr.eval.rubric.anthropic") as mock_anthropic:
                mock_client = MagicMock()
                mock_anthropic.Anthropic.return_value = mock_client
                mock_client.messages.create.return_value = mock_response
                rubric = Rubric("Be factual")
                span = make_llm_span(output="Paris is the capital of France.")
                score = rubric.evaluate(span)

        assert score == pytest.approx(0.7)

    def test_rubric_raises_import_error_when_no_llm_available(self):
        with patch("peekr.eval.rubric.openai", None), patch("peekr.eval.rubric.anthropic", None):
            rubric = Rubric("Be concise")
            span = make_llm_span()
            with pytest.raises(ImportError):
                rubric.evaluate(span)
