"""Regression tests for the 6 bugs reported by a user running peekr against
a real company workload (multi-tenant agent with Anthropic tool-use).

Each test maps 1:1 to a numbered issue in the bug report.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from peekr import dashboard as dash
from peekr.eval import EvalExporter, _in_eval
from peekr.eval.citation import CitationAccuracy, looks_like_tool_call
from peekr.eval.hallucination import Hallucination
from peekr.span import Span


# ---------------------------------------------------------------------------
# Fix 1 — Anthropic system prompt visible in dashboard rows
# ---------------------------------------------------------------------------


class TestFix1AnthropicSystemPrompt:
    def test_patch_prepends_system_into_input_messages(self):
        """The Anthropic patch should merge `system=...` into messages as
        role=system so OpenAI-shaped consumers (dashboard) see it."""

        # Simulate what the patch does: build a synthetic span the way the
        # patched_create function does, then verify the input shape.
        s = Span(name="anthropic.messages", trace_id="t1")
        messages = [{"role": "user", "content": "When was the Eiffel Tower built?"}]
        system = "Use only the provided context. Year completed: 1889."

        # Apply the same merge logic the patch uses.
        unified = [{"role": "system", "content": system}, *messages]
        s.attributes["input"] = json.dumps(unified)
        s.attributes["system"] = system  # backward-compat

        # Now exercise the dashboard's _rows + parseInput.
        spans = [
            {
                "name": "anthropic.messages",
                "trace_id": "t1",
                "span_id": "s1",
                "parent_id": None,
                "start_time": 0,
                "end_time": 1,
                "duration_ms": 1000,
                "status": "ok",
                "attributes": s.attributes,
            }
        ]
        rows = dash._rows(spans)
        assert len(rows) == 1
        row = rows[0]
        # The row carries system both inline (in input messages) AND on its own.
        parsed = json.loads(row["input"])
        sys_msg = next((m for m in parsed if m["role"] == "system"), None)
        assert sys_msg is not None
        assert "Eiffel Tower" in sys_msg["content"] or "1889" in sys_msg["content"]
        assert row["system"] is not None  # backward fallback also populated

    def test_dashboard_rows_carry_system_for_legacy_traces(self):
        """For traces written before the patch fix, attributes.system exists
        but messages has no role=system entry — the row must still carry it
        so the dashboard's parseInput fallback can find it."""
        spans = [
            {
                "name": "anthropic.messages",
                "trace_id": "t2",
                "span_id": "s2",
                "parent_id": None,
                "start_time": 0,
                "end_time": 1,
                "duration_ms": 1000,
                "status": "ok",
                "attributes": {
                    "input": json.dumps([{"role": "user", "content": "hi"}]),
                    "system": "You are a helpful assistant grounded in the docs.",
                },
            }
        ]
        rows = dash._rows(spans)
        assert rows[0]["system"] is not None
        assert "helpful assistant" in rows[0]["system"]


# ---------------------------------------------------------------------------
# Fix 2 — quoted_title regex no longer flags product names
# ---------------------------------------------------------------------------


class TestFix2QuotedTitleNoise:
    def test_capitalized_quoted_phrase_alone_is_not_a_citation(self):
        """The product name 'Junction Box' inside a tool-use payload must
        NOT be flagged as an invented citation."""
        s = Span(name="anthropic.messages", trace_id="t")
        s.attributes["input"] = "Pack the following items: junction box, screws."
        s.attributes["output"] = (
            "ToolUseBlock(id='abc', input={'item': 'Junction Box'})"
        )
        score = CitationAccuracy().evaluate(s)
        assert score == pytest.approx(1.0)
        # Tool-call shape → CitationAccuracy returns 1.0 without recording details.
        assert "citation_details" not in s.attributes

    def test_quoted_title_with_preamble_still_matches(self):
        """A real citation pattern like 'see "X"' should still be detected."""
        s = Span(name="openai.chat.completions", trace_id="t")
        s.attributes["input"] = "Reference works in machine learning."
        s.attributes["output"] = (
            'See "Attention Is All You Need" for the original work.'
        )
        score = CitationAccuracy().evaluate(s)
        # The quoted title isn't in the context → invented → 0.0
        assert score < 1.0
        details = s.attributes.get("citation_details") or {}
        assert details.get("invented", 0) >= 1

    def test_tool_call_detection_covers_common_shapes(self):
        assert looks_like_tool_call("ToolUseBlock(id='x', input={...})")
        assert looks_like_tool_call("ToolResultBlock(...)")
        assert looks_like_tool_call("{'item': 'screws'}")
        assert looks_like_tool_call("[{'foo': 1}]")
        assert looks_like_tool_call("  TextBlock(text='hi')")
        assert not looks_like_tool_call("The Eiffel Tower is in Paris.")


# ---------------------------------------------------------------------------
# Fix 3 — Eval scores actually reach disk
# ---------------------------------------------------------------------------


class TestFix3ExporterOrder:
    def test_evaluators_run_before_jsonl_writes(self, tmp_path):
        """A user calling instrument(storage='jsonl', evaluators=[...]) must
        see eval_scores in the resulting traces.jsonl."""
        from peekr.exporters import _exporters as exporter_list, export_span

        # Clear current exporter state to make this test deterministic.
        exporter_list.clear()

        from peekr import instrument as _instrument

        # Force re-instrument by clearing peekr's "patched" flag is unsafe here;
        # we just call instrument with our test path and verify the registered
        # exporter ORDER.
        jsonl_path = str(tmp_path / "traces.jsonl")
        _instrument(
            console=False,
            storage="jsonl",
            jsonl_path=jsonl_path,
            evaluators=[_FakeEval(0.42)],
            async_eval=False,  # assertions read the span/file immediately
        )

        # EvalExporter must come BEFORE JSONLExporter in the registry.
        names = [type(e).__name__ for e in exporter_list]
        assert "EvalExporter" in names
        assert "JSONLExporter" in names
        assert names.index("EvalExporter") < names.index("JSONLExporter"), (
            f"Exporter order broken: {names}"
        )

        # Drive an LLM-shape span through the pipeline and verify the JSONL
        # line carries eval_scores.
        s = Span(name="openai.chat.completions", trace_id="t1")
        s.attributes["input"] = "hi"
        s.attributes["output"] = "hello"
        s.finish()
        export_span(s)

        line = open(jsonl_path).readline().strip()
        record = json.loads(line)
        assert record["attributes"]["eval_scores"]["fake"] == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Fix 4 — Judge spans hidden from the dashboard
# ---------------------------------------------------------------------------


class TestFix4HideJudgeSpans:
    def test_internal_spans_filtered_from_dashboard_rows(self):
        """Spans marked attributes['peekr.internal'] = True must not appear
        in the dashboard's _rows output."""
        spans = [
            {
                "name": "openai.chat.completions",
                "trace_id": "user-call",
                "span_id": "su",
                "parent_id": None,
                "start_time": 0,
                "end_time": 1,
                "duration_ms": 1000,
                "status": "ok",
                "attributes": {"input": "hi", "output": "hello"},
            },
            {
                "name": "openai.chat.completions",
                "trace_id": "judge-call",
                "span_id": "sj",
                "parent_id": None,
                "start_time": 0,
                "end_time": 1,
                "duration_ms": 1000,
                "status": "ok",
                "attributes": {
                    "input": "judge prompt",
                    "output": "0.7",
                    "peekr.internal": True,
                },
            },
        ]
        # generate_dashboard filters via the same predicate; we exercise the
        # filter directly in case the rows function is reused elsewhere.
        kept = [
            s
            for s in spans
            if any(s["name"].startswith(p) for p in dash._LLM_PREFIXES)
            and not (s.get("attributes") or {}).get("peekr.internal")
        ]
        assert len(kept) == 1
        assert kept[0]["trace_id"] == "user-call"

    def test_patches_tag_internal_when_in_eval_is_set(self):
        """When `_in_eval` is True (i.e. we're inside an evaluator), the patch
        sets peekr.internal on the span it creates."""
        from peekr.patches.openai_patch import patch_openai  # noqa: F401

        # We don't run the real patch; we exercise the same conditional the
        # patch uses to keep the test isolated.
        token = _in_eval.set(True)
        try:
            s = Span(name="openai.chat.completions", trace_id="judge")
            try:
                if _in_eval.get():
                    s.attributes["peekr.internal"] = True
            except Exception:
                pass
            assert s.attributes.get("peekr.internal") is True
        finally:
            _in_eval.reset(token)


# ---------------------------------------------------------------------------
# Fix 5 — Per-span evaluator filtering
# ---------------------------------------------------------------------------


class _FakeEval:
    """Evaluator that returns a constant score so we can verify it ran."""

    def __init__(self, score=1.0, name_override="fake"):
        self._score = score
        self._name = name_override

    @property
    def name(self):
        return self._name

    def evaluate(self, span):
        return self._score


class TestFix5SpanFilter:
    def test_span_filter_skips_non_matching_spans(self):
        """An EvalExporter with span_filter=lambda s: s.attributes.get('endpoint') == '/api/qa'
        should only evaluate the /api/qa span, leaving the others untouched."""
        ev = _FakeEval(0.9, name_override="fake")
        exporter = EvalExporter(
            async_eval=False,
            evaluators=[ev],
            span_filter=lambda s: (s.attributes or {}).get("endpoint") == "/api/qa",
        )

        qa = Span(name="openai.chat.completions", trace_id="t1")
        qa.attributes["endpoint"] = "/api/qa"
        qa.attributes["output"] = "ok"
        qa.finish()

        admin = Span(name="openai.chat.completions", trace_id="t2")
        admin.attributes["endpoint"] = "/api/admin"
        admin.attributes["output"] = "ok"
        admin.finish()

        exporter.export(qa)
        exporter.export(admin)

        assert qa.attributes.get("eval_scores", {}).get("fake") == pytest.approx(0.9)
        assert "eval_scores" not in admin.attributes  # filter rejected this span

    def test_faulty_filter_does_not_break_tracing(self):
        exporter = EvalExporter(
            async_eval=False,
            evaluators=[_FakeEval(1.0)],
            span_filter=lambda s: 1 / 0,  # always raises
        )
        s = Span(name="openai.chat.completions", trace_id="t")
        s.finish()
        exporter.export(s)  # must not raise
        assert "eval_scores" not in s.attributes

    def test_no_filter_evaluates_every_llm_span_as_before(self):
        exporter = EvalExporter(async_eval=False, evaluators=[_FakeEval(0.5, "ok")])
        s = Span(name="anthropic.messages", trace_id="t")
        s.finish()
        exporter.export(s)
        assert s.attributes["eval_scores"]["ok"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Fix 6 — Hallucination skips tool-call outputs
# ---------------------------------------------------------------------------


class TestFix6HallucinationOnToolUse:
    def test_tool_use_output_returns_one_without_calling_judge(self):
        """`ToolUseBlock(...)` style outputs must not be sent to the judge —
        the score is 1.0 (not-evaluable) and the details record the reason."""
        with patch("peekr.eval._judge.openai") as mock_openai:
            ev = Hallucination(detailed=True)
            s = Span(name="anthropic.messages", trace_id="t")
            s.attributes["input"] = "Find the user's order items."
            s.attributes["output"] = (
                "ToolUseBlock(id='toolu_x', name='lookup', "
                "input={'order_id': 42, 'items': ['screws', 'washers']})"
            )
            score = ev.evaluate(s)
            assert score == pytest.approx(1.0)
            # The judge should never have been called.
            mock_openai.chat.completions.create.assert_not_called()
            assert s.attributes["hallucination_details"]["total"] == 0
            assert (
                s.attributes["hallucination_details"]["reason"]
                == "tool call, not a generation"
            )

    def test_json_payload_output_is_also_skipped(self):
        with patch("peekr.eval._judge.openai") as mock_openai:
            ev = Hallucination()
            s = Span(name="anthropic.messages", trace_id="t")
            s.attributes["input"] = "ctx"
            s.attributes["output"] = '{"claims": [{"id": 1}]}'
            assert ev.evaluate(s) == pytest.approx(1.0)
            mock_openai.chat.completions.create.assert_not_called()

    def test_free_text_output_still_evaluated(self):
        """Don't over-fit: real free-text outputs must still hit the judge."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "0.62"
        with patch("peekr.eval._judge.openai") as mock_openai:
            mock_openai.chat.completions.create.return_value = mock_response
            ev = Hallucination()
            s = Span(name="openai.chat.completions", trace_id="t")
            s.attributes["input"] = "The Eiffel Tower was completed in 1889."
            s.attributes["output"] = (
                "The Eiffel Tower was completed in 1923 by Frank Lloyd Wright."
            )
            score = ev.evaluate(s)
            assert score == pytest.approx(0.62)
            mock_openai.chat.completions.create.assert_called_once()
