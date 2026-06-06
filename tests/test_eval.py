from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from peekr.span import Span
from peekr.eval import BaseEvaluator, EvalExporter, _in_eval
from peekr.eval.rubric import NotEmpty, NoError, Rubric
from peekr.eval.hallucination import Hallucination, _parse_score, _parse_claims


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
        exporter = EvalExporter(async_eval=False, evaluators=[evaluator])
        span = make_llm_span()
        exporter.export(span)
        assert "eval_scores" in span.attributes
        assert span.attributes["eval_scores"]["ConstantEval"] == pytest.approx(0.9)

    def test_non_llm_span_is_skipped(self):
        evaluator = _ConstantEvaluator()
        exporter = EvalExporter(async_eval=False, evaluators=[evaluator])
        span = make_non_llm_span()
        exporter.export(span)
        assert "eval_scores" not in span.attributes

    def test_anthropic_span_is_evaluated(self):
        evaluator = _ConstantEvaluator(score=1.0)
        exporter = EvalExporter(async_eval=False, evaluators=[evaluator])
        span = make_llm_span(name="anthropic.messages")
        exporter.export(span)
        assert span.attributes["eval_scores"]["ConstantEval"] == pytest.approx(1.0)

    def test_bedrock_span_is_evaluated(self):
        evaluator = _ConstantEvaluator(score=0.5)
        exporter = EvalExporter(async_eval=False, evaluators=[evaluator])
        span = make_llm_span(name="bedrock.invoke_model")
        exporter.export(span)
        assert span.attributes["eval_scores"]["ConstantEval"] == pytest.approx(0.5)

    def test_multiple_evaluators_all_run(self):
        e1 = _ConstantEvaluator(score=0.8, name_override="Eval1")
        e2 = _ConstantEvaluator(score=0.4, name_override="Eval2")
        exporter = EvalExporter(async_eval=False, evaluators=[e1, e2])
        span = make_llm_span()
        exporter.export(span)
        scores = span.attributes["eval_scores"]
        assert scores["Eval1"] == pytest.approx(0.8)
        assert scores["Eval2"] == pytest.approx(0.4)

    def test_failing_evaluator_recorded_as_eval_error_not_silent_zero(self):
        """Behaviour change: previously a crashing evaluator was stored as
        a 0.0 score. That made "judge unavailable" indistinguishable from
        "judge graded 0.0 = fully hallucinated" on the dashboard. The
        exporter now records the exception in `eval_errors` and omits the
        score entirely so consumers know it's missing, not zero."""
        class BrokenEvaluator(BaseEvaluator):
            @property
            def name(self):
                return "BrokenEval"

            def evaluate(self, span):
                raise RuntimeError("boom")

        exporter = EvalExporter(async_eval=False, evaluators=[BrokenEvaluator()])
        span = make_llm_span()
        exporter.export(span)
        assert "BrokenEval" not in (span.attributes.get("eval_scores") or {})
        assert "BrokenEval" in span.attributes["eval_errors"]
        assert "boom" in span.attributes["eval_errors"]["BrokenEval"]


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

        exporter = EvalExporter(async_eval=False, evaluators=[CountingEvaluator()])
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
        exporter = EvalExporter(async_eval=False, evaluators=[_ConstantEvaluator()])
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

        exporter = EvalExporter(async_eval=False, evaluators=[AlwaysRaises()])
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

        with patch("peekr.eval._judge.openai") as mock_openai:
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

        with patch("peekr.eval._judge.openai") as mock_openai:
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
        with patch("peekr.eval._judge.openai", None):
            with patch("peekr.eval._judge.anthropic") as mock_anthropic:
                mock_client = MagicMock()
                mock_anthropic.Anthropic.return_value = mock_client
                mock_client.messages.create.return_value = mock_response
                rubric = Rubric("Be factual")
                span = make_llm_span(output="Paris is the capital of France.")
                score = rubric.evaluate(span)

        assert score == pytest.approx(0.7)

    def test_rubric_raises_judge_unavailable_when_no_llm_installed(self):
        """Behaviour change: previously raised ImportError. Now raises the
        more specific JudgeUnavailable so callers (EvalExporter) can record
        it as a judge problem rather than a programming error."""
        from peekr.eval._judge import JudgeUnavailable
        with patch("peekr.eval._judge.openai", None), patch("peekr.eval._judge.anthropic", None):
            rubric = Rubric("Be concise")
            span = make_llm_span()
            with pytest.raises(JudgeUnavailable):
                rubric.evaluate(span)


# ---------------------------------------------------------------------------
# Hallucination (mocked — no real API calls)
# ---------------------------------------------------------------------------

class TestParseScore:
    def test_parses_plain_float(self):
        assert _parse_score("0.42") == pytest.approx(0.42)

    def test_parses_float_with_surrounding_text(self):
        assert _parse_score("Score: 0.73 (mostly grounded)") == pytest.approx(0.73)

    def test_clamps_above_one(self):
        assert _parse_score("1.5") == pytest.approx(1.0)

    def test_clamps_below_zero(self):
        assert _parse_score("-0.2") == pytest.approx(0.0)

    def test_raises_on_no_number(self):
        with pytest.raises(ValueError):
            _parse_score("not a score")


class TestHallucination:
    def test_returns_one_when_output_is_empty(self):
        # Nothing to hallucinate — don't penalize.
        with patch("peekr.eval._judge.openai") as mock_openai:
            evaluator = Hallucination()
            span = make_llm_span(output="")
            assert evaluator.evaluate(span) == pytest.approx(1.0)
            mock_openai.chat.completions.create.assert_not_called()

    def test_returns_one_when_no_context_available(self):
        # No grounding source → not evaluable; don't poison the metric.
        with patch("peekr.eval._judge.openai") as mock_openai:
            evaluator = Hallucination()
            span = Span(name="openai.chat.completions", trace_id="t1")
            span.attributes["output"] = "Paris is the capital of France."
            span.finish()
            assert evaluator.evaluate(span) == pytest.approx(1.0)
            mock_openai.chat.completions.create.assert_not_called()

    def test_calls_openai_and_returns_score(self):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "0.9"

        with patch("peekr.eval._judge.openai") as mock_openai:
            mock_openai.chat.completions.create.return_value = mock_response
            evaluator = Hallucination()
            span = make_llm_span(
                input_text="The Eiffel Tower is in Paris.",
                output="The Eiffel Tower is in Paris.",
            )
            assert evaluator.evaluate(span) == pytest.approx(0.9)
            mock_openai.chat.completions.create.assert_called_once()

    def test_prompt_contains_context_and_output(self):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "0.5"
        context = "France's capital is Paris."
        output = "France's capital is Lyon."

        with patch("peekr.eval._judge.openai") as mock_openai:
            mock_openai.chat.completions.create.return_value = mock_response
            evaluator = Hallucination()
            span = make_llm_span(input_text=context, output=output)
            evaluator.evaluate(span)

        call_kwargs = mock_openai.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        prompt_text = messages[0]["content"]
        assert context in prompt_text
        assert output in prompt_text

    def test_context_extractor_overrides_input(self):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "1.0"

        retrieved_doc = "Berlin is the capital of Germany."
        with patch("peekr.eval._judge.openai") as mock_openai:
            mock_openai.chat.completions.create.return_value = mock_response
            evaluator = Hallucination(
                context_extractor=lambda s: retrieved_doc,
            )
            span = make_llm_span(input_text="ignored", output="Berlin is in Germany.")
            evaluator.evaluate(span)

        prompt_text = mock_openai.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert retrieved_doc in prompt_text
        assert "ignored" not in prompt_text

    def test_uses_custom_model_when_provided(self):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "0.8"

        with patch("peekr.eval._judge.openai") as mock_openai:
            mock_openai.chat.completions.create.return_value = mock_response
            evaluator = Hallucination(model="gpt-4o")
            span = make_llm_span(input_text="ctx", output="out")
            evaluator.evaluate(span)

        assert mock_openai.chat.completions.create.call_args.kwargs["model"] == "gpt-4o"

    def test_falls_back_to_anthropic_when_no_openai(self):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="0.65")]

        with patch("peekr.eval._judge.openai", None):
            with patch("peekr.eval._judge.anthropic") as mock_anthropic:
                mock_client = MagicMock()
                mock_anthropic.Anthropic.return_value = mock_client
                mock_client.messages.create.return_value = mock_response
                evaluator = Hallucination()
                span = make_llm_span(input_text="ctx", output="out")
                score = evaluator.evaluate(span)

        assert score == pytest.approx(0.65)

    def test_raises_judge_unavailable_when_no_llm_installed(self):
        """Behaviour change: previously raised ImportError. Now raises the
        more specific JudgeUnavailable so EvalExporter records this as a
        judge problem (eval_errors) rather than as a 0.0 score."""
        from peekr.eval._judge import JudgeUnavailable
        with patch("peekr.eval._judge.openai", None), \
             patch("peekr.eval._judge.anthropic", None):
            evaluator = Hallucination()
            span = make_llm_span(input_text="ctx", output="out")
            with pytest.raises(JudgeUnavailable):
                evaluator.evaluate(span)

    def test_name(self):
        assert Hallucination().name == "Hallucination"

    def test_integration_via_eval_exporter(self):
        # Hallucination should plug into EvalExporter and write to eval_scores.
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "0.3"

        with patch("peekr.eval._judge.openai") as mock_openai:
            mock_openai.chat.completions.create.return_value = mock_response
            exporter = EvalExporter(async_eval=False, evaluators=[Hallucination()])
            span = make_llm_span(input_text="The sky is blue.", output="The sky is green.")
            exporter.export(span)

        assert span.attributes["eval_scores"]["Hallucination"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Hallucination — detailed (RAGAS-style claim decomposition)
# ---------------------------------------------------------------------------

class TestParseClaims:
    def test_parses_valid_json(self):
        text = '{"claims": [{"text": "A is true", "verdict": "supported"}]}'
        out = _parse_claims(text)
        assert out == [{"text": "A is true", "verdict": "supported"}]

    def test_extracts_json_from_surrounding_prose(self):
        text = 'Here is the analysis: {"claims": [{"text": "X", "verdict": "contradicted"}]}'
        out = _parse_claims(text)
        assert out == [{"text": "X", "verdict": "contradicted"}]

    def test_unknown_verdict_falls_back_to_unsupported(self):
        text = '{"claims": [{"text": "X", "verdict": "maybe"}]}'
        assert _parse_claims(text) == [{"text": "X", "verdict": "unsupported"}]

    def test_skips_claims_with_empty_text(self):
        text = '{"claims": [{"text": "", "verdict": "supported"}, {"text": "Y", "verdict": "supported"}]}'
        assert _parse_claims(text) == [{"text": "Y", "verdict": "supported"}]

    def test_empty_claims_list(self):
        assert _parse_claims('{"claims": []}') == []

    def test_raises_on_no_json(self):
        with pytest.raises(ValueError):
            _parse_claims("no json here")


class TestHallucinationDetailed:
    def _judge_response(self, claims: list[dict]) -> MagicMock:
        mock = MagicMock()
        mock.choices[0].message.content = json.dumps({"claims": claims})
        return mock

    def test_score_is_supported_over_total(self):
        with patch("peekr.eval._judge.openai") as mock_openai:
            mock_openai.chat.completions.create.return_value = self._judge_response([
                {"text": "Claim 1", "verdict": "supported"},
                {"text": "Claim 2", "verdict": "supported"},
                {"text": "Claim 3", "verdict": "contradicted"},
                {"text": "Claim 4", "verdict": "unsupported"},
            ])
            evaluator = Hallucination(detailed=True)
            span = make_llm_span(input_text="ctx", output="out")
            score = evaluator.evaluate(span)
        assert score == pytest.approx(0.5)

    def test_writes_details_to_span(self):
        with patch("peekr.eval._judge.openai") as mock_openai:
            mock_openai.chat.completions.create.return_value = self._judge_response([
                {"text": "A", "verdict": "supported"},
                {"text": "B", "verdict": "contradicted"},
            ])
            evaluator = Hallucination(detailed=True)
            span = make_llm_span(input_text="ctx", output="out")
            evaluator.evaluate(span)

        details = span.attributes["hallucination_details"]
        assert details["total"] == 2
        assert details["supported"] == 1
        assert details["contradicted"] == 1
        assert details["unsupported"] == 0
        assert details["score"] == pytest.approx(0.5)
        assert len(details["claims"]) == 2

    def test_no_claims_extracted_returns_one(self):
        # Output is non-empty but the judge finds no factual claims → score 1.0.
        with patch("peekr.eval._judge.openai") as mock_openai:
            mock_openai.chat.completions.create.return_value = self._judge_response([])
            evaluator = Hallucination(detailed=True)
            span = make_llm_span(input_text="ctx", output="How are you?")
            score = evaluator.evaluate(span)
        assert score == pytest.approx(1.0)
        assert span.attributes["hallucination_details"]["total"] == 0

    def test_integration_via_eval_exporter_writes_both(self):
        with patch("peekr.eval._judge.openai") as mock_openai:
            mock_openai.chat.completions.create.return_value = self._judge_response([
                {"text": "A", "verdict": "supported"},
                {"text": "B", "verdict": "unsupported"},
            ])
            exporter = EvalExporter(async_eval=False, evaluators=[Hallucination(detailed=True)])
            span = make_llm_span(input_text="ctx", output="out")
            exporter.export(span)

        assert span.attributes["eval_scores"]["Hallucination"] == pytest.approx(0.5)
        assert span.attributes["hallucination_details"]["total"] == 2


# ---------------------------------------------------------------------------
# Async eval path — background scoring + flush + re-export to storage
# ---------------------------------------------------------------------------

class TestAsyncEval:
    def test_async_scores_reach_storage_after_flush(self):
        """The async worker must re-export the scored span through storage
        exporters. (Regression: a broken relative import made every
        background eval die with ImportError inside its Future, silently
        dropping the scores it had just computed.)"""
        from peekr.exporters import _exporters

        collected = []

        class Collector:
            _is_storage = True

            def export(self, span):
                collected.append(span)

        saved = list(_exporters)
        _exporters[:] = [Collector()]
        try:
            exporter = EvalExporter(evaluators=[_ConstantEvaluator(score=0.7, name_override="bg")])
            span = make_llm_span()
            exporter.export(span)
            exporter.flush(timeout=10)
        finally:
            _exporters[:] = saved

        assert collected, "async eval must re-export the scored span to storage"
        assert collected[-1].attributes["eval_scores"]["bg"] == pytest.approx(0.7)
        # The original span object is untouched — async scores live on the
        # re-exported copy (storage upserts on span_id).
        assert "eval_scores" not in span.attributes

    def test_flush_noop_when_nothing_pending(self):
        exporter = EvalExporter(evaluators=[_ConstantEvaluator(score=0.5)])
        exporter.flush(timeout=1)  # must not raise or hang

    def test_async_worker_failures_do_not_leak_to_caller(self):
        class Boom(BaseEvaluator):
            def evaluate(self, span):
                raise RuntimeError("judge down")

        exporter = EvalExporter(evaluators=[Boom()])
        span = make_llm_span()
        exporter.export(span)  # must not raise
        exporter.flush(timeout=10)
