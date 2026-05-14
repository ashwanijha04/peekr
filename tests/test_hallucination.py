from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from peekr.eval.hallucination import (
    Faithfulness,
    AnswerRelevance,
    ContextRelevance,
    _parse_score,
    _strip_code_fence,
)
from peekr.span import Span


class TestParseScore:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("0.8", 0.8),
            ("Score: 0.42", 0.42),
            ("1.0", 1.0),
            ("0", 0.0),
            ("87", 0.87),       # treated as percent
            ("150", 1.0),        # clamped
            ("-0.5", 0.0),       # clamped
        ],
    )
    def test_parses_various_formats(self, text, expected):
        assert _parse_score(text) == expected

    def test_no_number_returns_zero(self):
        assert _parse_score("not a number") == 0.0


class TestStripCodeFence:
    def test_removes_json_fence(self):
        s = "```json\n{\"a\": 1}\n```"
        assert _strip_code_fence(s) == '{"a": 1}'

    def test_removes_plain_fence(self):
        s = "```\n{\"a\": 1}\n```"
        assert _strip_code_fence(s) == '{"a": 1}'

    def test_passthrough(self):
        assert _strip_code_fence('{"a": 1}') == '{"a": 1}'


class TestFaithfulness:
    def test_score_with_supported_claims(self):
        f = Faithfulness()
        judge_response = (
            '{"claims": [{"claim": "X is 5", "supported": true},'
            ' {"claim": "X is real", "supported": true}]}'
        )
        with patch("peekr.eval.hallucination._call_judge", return_value=judge_response):
            score = f.score("X is 5 and X is real", context="X = 5")
        assert score == 1.0

    def test_score_with_partial_support(self):
        f = Faithfulness()
        judge_response = (
            '{"claims": [{"claim": "X is 5", "supported": true},'
            ' {"claim": "X is purple", "supported": false}]}'
        )
        with patch("peekr.eval.hallucination._call_judge", return_value=judge_response):
            score = f.score("answer", context="ctx")
        assert score == 0.5

    def test_score_with_no_claims_is_perfect(self):
        # No factual claims = nothing to be wrong about
        f = Faithfulness()
        with patch("peekr.eval.hallucination._call_judge", return_value='{"claims": []}'):
            assert f.score("hi", context="ctx") == 1.0

    def test_handles_fenced_json(self):
        f = Faithfulness()
        wrapped = '```json\n{"claims": [{"claim": "x", "supported": false}]}\n```'
        with patch("peekr.eval.hallucination._call_judge", return_value=wrapped):
            assert f.score("a", context="b") == 0.0

    def test_falls_back_to_float_parser_on_bad_json(self):
        f = Faithfulness()
        with patch("peekr.eval.hallucination._call_judge", return_value="0.7"):
            assert f.score("a", context="b") == 0.7

    def test_evaluate_uses_span_attrs(self):
        span = Span(
            name="openai.chat.completions",
            trace_id="t",
            attributes={
                "output": "the moon is cheese",
                "grounding.context": "the moon is rock",
            },
        )
        with patch(
            "peekr.eval.hallucination._call_judge",
            return_value='{"claims": [{"claim": "moon is cheese", "supported": false}]}',
        ):
            assert Faithfulness().evaluate(span) == 0.0

    def test_evaluate_missing_context_returns_zero(self):
        span = Span(
            name="openai.chat.completions",
            trace_id="t",
            attributes={"output": "hi"},
        )
        assert Faithfulness().evaluate(span) == 0.0


class TestAnswerRelevance:
    def test_score(self):
        with patch("peekr.eval.hallucination._call_judge", return_value="0.9"):
            assert AnswerRelevance().score("answer", query="q") == 0.9

    def test_evaluate(self):
        span = Span(
            name="openai.chat.completions",
            trace_id="t",
            attributes={"output": "Paris", "grounding.query": "Capital of France?"},
        )
        with patch("peekr.eval.hallucination._call_judge", return_value="1.0"):
            assert AnswerRelevance().evaluate(span) == 1.0


class TestContextRelevance:
    def test_score(self):
        with patch("peekr.eval.hallucination._call_judge", return_value="0.6"):
            assert ContextRelevance().score(context="ctx", query="q") == 0.6


class TestGrounding:
    def test_set_grounding(self):
        from peekr.session import set_grounding, get_grounding

        # Need a span for set_grounding to attach to
        from peekr.context import start_span, end_span

        span, token = start_span("openai.chat.completions")
        try:
            set_grounding(context="docs", query="q1")
            assert get_grounding("context") == "docs"
            assert get_grounding("query") == "q1"
            assert span.attributes["grounding.context"] == "docs"
        finally:
            end_span(span, token)

    def test_session_grounding(self):
        from peekr.session import session, get_grounding

        with session(grounding={"context": "ctx", "query": "q"}):
            assert get_grounding("context") == "ctx"
            assert get_grounding("query") == "q"

    def test_list_context_joined(self):
        from peekr.eval.hallucination import _grounding_from_span

        span = Span(
            name="x",
            trace_id="t",
            attributes={"grounding.context": ["doc1", "doc2"]},
        )
        assert "doc1" in _grounding_from_span(span, "context")
        assert "doc2" in _grounding_from_span(span, "context")
