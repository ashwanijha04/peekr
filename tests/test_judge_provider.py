"""Regression tests for the silent-zero-score bug pack.

Each test maps to a numbered problem reported by users:

  1. Provider detection was capability-based (`openai is not None`) instead
     of credential-based, so installs with `openai` as a transitive dep but
     only an `ANTHROPIC_API_KEY` set fell through to OpenAI, failed auth,
     and got silently scored as 0.0.
  2. `_judge` had no path to fall back to Anthropic when openai was
     importable-but-broken.
  3. EvalExporter swallowed evaluator exceptions and wrote `0.0`, making
     "judge crashed" indistinguishable from "judge ran and rated 0.0".
  4. No `judge_provider=` override existed to force a specific provider.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from peekr.eval import EvalExporter
from peekr.eval._judge import (
    JudgeUnavailable,
    call_judge,
    select_provider,
)
from peekr.eval.hallucination import Hallucination
from peekr.eval.rubric import Rubric
from peekr.span import Span


def _mock_openai_response(text: str = "0.9") -> MagicMock:
    m = MagicMock()
    m.choices[0].message.content = text
    return m


def _mock_anthropic_response(text: str = "0.7") -> MagicMock:
    m = MagicMock()
    m.content = [MagicMock(text=text)]
    return m


# ---------------------------------------------------------------------------
# select_provider — env-key-aware
# ---------------------------------------------------------------------------


class TestSelectProvider:
    def test_prefers_anthropic_when_only_anthropic_key_is_set(self):
        """The original symptom: openai is importable (transitive dep) but no
        OPENAI_API_KEY. Selection must pick Anthropic."""
        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True),
            patch("peekr.eval._judge.openai", MagicMock()),
            patch("peekr.eval._judge.anthropic", MagicMock()),
        ):
            assert select_provider("auto") == "anthropic"

    def test_prefers_openai_when_only_openai_key_is_set(self):
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-oai-test"}, clear=True),
            patch("peekr.eval._judge.openai", MagicMock()),
            patch("peekr.eval._judge.anthropic", MagicMock()),
        ):
            assert select_provider("auto") == "openai"

    def test_openai_wins_when_both_keys_are_set_for_back_compat(self):
        with (
            patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "sk-1", "ANTHROPIC_API_KEY": "sk-2"},
                clear=True,
            ),
            patch("peekr.eval._judge.openai", MagicMock()),
            patch("peekr.eval._judge.anthropic", MagicMock()),
        ):
            assert select_provider("auto") == "openai"

    def test_falls_back_to_importable_sdk_when_no_env_keys(self):
        """Legacy code path — ambient auth (Azure cred provider, on-prem
        proxy, etc.) — should still work. openai wins the tie."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("peekr.eval._judge.openai", MagicMock()),
            patch("peekr.eval._judge.anthropic", MagicMock()),
        ):
            assert select_provider("auto") == "openai"

    def test_falls_back_to_anthropic_when_openai_unimportable(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("peekr.eval._judge.openai", None),
            patch("peekr.eval._judge.anthropic", MagicMock()),
        ):
            assert select_provider("auto") == "anthropic"

    def test_raises_when_neither_sdk_is_installed(self):
        with (
            patch("peekr.eval._judge.openai", None),
            patch("peekr.eval._judge.anthropic", None),
        ):
            with pytest.raises(JudgeUnavailable):
                select_provider("auto")

    def test_explicit_openai_requires_openai_sdk(self):
        with patch("peekr.eval._judge.openai", None):
            with pytest.raises(
                JudgeUnavailable, match="openai package is not installed"
            ):
                select_provider("openai")

    def test_explicit_anthropic_requires_anthropic_sdk(self):
        with patch("peekr.eval._judge.anthropic", None):
            with pytest.raises(
                JudgeUnavailable, match="anthropic package is not installed"
            ):
                select_provider("anthropic")

    def test_explicit_overrides_env_keys(self):
        """If the user says `judge_provider="anthropic"`, that wins even if
        OPENAI_API_KEY is set (and OPENAI is importable)."""
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk-1"}, clear=True),
            patch("peekr.eval._judge.openai", MagicMock()),
            patch("peekr.eval._judge.anthropic", MagicMock()),
        ):
            assert select_provider("anthropic") == "anthropic"


# ---------------------------------------------------------------------------
# call_judge — actual provider dispatch
# ---------------------------------------------------------------------------


class TestCallJudge:
    def test_uses_openai_when_selected(self):
        mock_oai = MagicMock()
        mock_oai.chat.completions.create.return_value = _mock_openai_response("0.83")
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk"}, clear=True),
            patch("peekr.eval._judge.openai", mock_oai),
            patch("peekr.eval._judge.anthropic", MagicMock()),
        ):
            text = call_judge("Score this", max_tokens=10)
        assert text == "0.83"
        mock_oai.chat.completions.create.assert_called_once()

    def test_uses_anthropic_when_only_anthropic_key_is_set(self):
        mock_anth = MagicMock()
        mock_client = MagicMock()
        mock_anth.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = _mock_anthropic_response("0.55")

        with (
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant"}, clear=True),
            patch("peekr.eval._judge.openai", MagicMock()),
            patch("peekr.eval._judge.anthropic", mock_anth),
        ):
            text = call_judge("Score this")
        assert text == "0.55"
        mock_client.messages.create.assert_called_once()

    def test_lets_provider_auth_errors_bubble_up(self):
        """If the configured provider raises (e.g. auth), call_judge does
        NOT swallow — the EvalExporter records it in eval_errors."""
        mock_oai = MagicMock()
        mock_oai.chat.completions.create.side_effect = RuntimeError("401 Unauthorized")
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "bad"}, clear=True),
            patch("peekr.eval._judge.openai", mock_oai),
        ):
            with pytest.raises(RuntimeError, match="401 Unauthorized"):
                call_judge("Score this")


# ---------------------------------------------------------------------------
# Hallucination + Rubric — judge_provider override flows through
# ---------------------------------------------------------------------------


class TestEvaluatorOverride:
    def test_hallucination_judge_provider_forces_anthropic(self):
        mock_anth = MagicMock()
        client = MagicMock()
        mock_anth.Anthropic.return_value = client
        client.messages.create.return_value = _mock_anthropic_response("0.42")

        # Critically: OPENAI_API_KEY is set AND openai SDK is importable, but
        # the explicit override picks Anthropic anyway.
        with (
            patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "sk", "ANTHROPIC_API_KEY": "sk-a"},
                clear=True,
            ),
            patch("peekr.eval._judge.openai", MagicMock()),
            patch("peekr.eval._judge.anthropic", mock_anth),
        ):
            ev = Hallucination(judge_provider="anthropic")
            s = Span(name="openai.chat.completions", trace_id="t")
            s.attributes["input"] = "ctx"
            s.attributes["output"] = "answer"
            score = ev.evaluate(s)
        assert score == pytest.approx(0.42)
        client.messages.create.assert_called_once()

    def test_rubric_judge_provider_forces_anthropic(self):
        mock_anth = MagicMock()
        client = MagicMock()
        mock_anth.Anthropic.return_value = client
        client.messages.create.return_value = _mock_anthropic_response("0.6")

        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "sk"}, clear=True),
            patch("peekr.eval._judge.openai", MagicMock()),
            patch("peekr.eval._judge.anthropic", mock_anth),
        ):
            ev = Rubric("Be concise", judge_provider="anthropic")
            s = Span(name="openai.chat.completions", trace_id="t")
            s.attributes["output"] = "fine"
            score = ev.evaluate(s)
        assert score == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# EvalExporter — distinguish crash from real 0.0
# ---------------------------------------------------------------------------


class _ConstantEvaluator:
    def __init__(self, score=1.0, name_override="const"):
        self._score = score
        self._name = name_override

    @property
    def name(self):
        return self._name

    def evaluate(self, span):
        return self._score


class _CrashingEvaluator:
    name = "BoomEval"

    def evaluate(self, span):
        raise RuntimeError("401 Unauthorized (no OPENAI_API_KEY)")


class TestEvalExporterDistinguishesCrashFromZero:
    def test_crash_records_eval_errors_not_zero_score(self):
        """The main fix: a judge exception must NOT show up as 'fully
        hallucinated' on the dashboard."""
        exporter = EvalExporter(async_eval=False, evaluators=[_CrashingEvaluator()])
        s = Span(name="openai.chat.completions", trace_id="t")
        s.attributes["output"] = "real answer"
        s.finish()
        exporter.export(s)

        # The bad old behaviour:  eval_scores would be {"BoomEval": 0.0}.
        # New: eval_scores is absent for that evaluator; eval_errors carries
        # a useful error string.
        assert "BoomEval" not in (s.attributes.get("eval_scores") or {})
        assert "BoomEval" in s.attributes["eval_errors"]
        assert "401 Unauthorized" in s.attributes["eval_errors"]["BoomEval"]
        assert "RuntimeError" in s.attributes["eval_errors"]["BoomEval"]

    def test_real_zero_score_still_recorded(self):
        """We must not break the case where the evaluator legitimately
        returns 0.0 — that's a meaningful score, not an error."""
        exporter = EvalExporter(
            async_eval=False,
            evaluators=[_ConstantEvaluator(score=0.0, name_override="legit_zero")],
        )
        s = Span(name="openai.chat.completions", trace_id="t")
        s.attributes["output"] = "garbage"
        s.finish()
        exporter.export(s)
        assert s.attributes["eval_scores"]["legit_zero"] == 0.0
        assert "eval_errors" not in s.attributes

    def test_one_evaluator_crash_does_not_block_others(self):
        """If two evaluators are configured and one crashes, the other's
        score must still be recorded."""
        exporter = EvalExporter(
            async_eval=False,
            evaluators=[
                _CrashingEvaluator(),
                _ConstantEvaluator(score=0.9, name_override="ok"),
            ],
        )
        s = Span(name="anthropic.messages", trace_id="t")
        s.finish()
        exporter.export(s)
        assert s.attributes["eval_scores"]["ok"] == 0.9
        assert "BoomEval" in s.attributes["eval_errors"]


# ---------------------------------------------------------------------------
# Dashboard surfaces the errors
# ---------------------------------------------------------------------------


class TestDashboardCarriesEvalErrors:
    def test_rows_include_eval_errors_attribute(self):
        from peekr import dashboard as dash

        spans = [
            {
                "name": "openai.chat.completions",
                "trace_id": "t",
                "span_id": "s",
                "parent_id": None,
                "start_time": 0,
                "end_time": 1,
                "duration_ms": 1000,
                "status": "ok",
                "attributes": {
                    "input": "hi",
                    "output": "hello",
                    "eval_errors": {"Hallucination": "RuntimeError: 401 Unauthorized"},
                },
            }
        ]
        rows = dash._rows(spans)
        assert rows[0]["eval_errors"] == {
            "Hallucination": "RuntimeError: 401 Unauthorized"
        }

    def test_rows_for_healthy_span_have_empty_eval_errors(self):
        from peekr import dashboard as dash

        spans = [
            {
                "name": "openai.chat.completions",
                "trace_id": "t",
                "span_id": "s",
                "parent_id": None,
                "start_time": 0,
                "end_time": 1,
                "duration_ms": 1000,
                "status": "ok",
                "attributes": {"input": "hi", "output": "hello"},
            }
        ]
        rows = dash._rows(spans)
        assert rows[0]["eval_errors"] == {}
