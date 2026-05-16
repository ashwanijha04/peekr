from __future__ import annotations

import pytest

pytest.importorskip("opentelemetry")

from peekr.otel import OTelExporter, _llm_attributes
from peekr.span import Span


class TestLLMAttributes:
    def test_translates_llm_span(self):
        span = Span(
            name="openai.chat.completions",
            trace_id="t",
            attributes={
                "model": "gpt-4o",
                "input": '[{"role":"user","content":"hi"}]',
                "output": "hello",
                "tokens_input": 12,
                "tokens_output": 3,
                "tokens_total": 15,
                "user_id": "u1",
                "session_id": "s1",
            },
        )
        attrs = _llm_attributes(span.to_dict())
        assert attrs["openinference.span.kind"] == "LLM"
        assert attrs["llm.model_name"] == "gpt-4o"
        assert attrs["llm.system"] == "openai"
        assert attrs["llm.token_count.prompt"] == 12
        assert attrs["llm.token_count.completion"] == 3
        assert attrs["llm.token_count.total"] == 15
        assert attrs["input.mime_type"] == "application/json"
        assert attrs["output.mime_type"] == "text/plain"
        assert attrs["user.id"] == "u1"
        assert attrs["session.id"] == "s1"

    def test_translates_tool_span_as_chain(self):
        span = Span(
            name="tool.search",
            trace_id="t",
            attributes={"input": "q", "output": "results"},
        )
        attrs = _llm_attributes(span.to_dict())
        assert attrs["openinference.span.kind"] == "CHAIN"

    def test_error_attribute(self):
        span = Span(
            name="openai.chat.completions",
            trace_id="t",
            status="error",
            attributes={"error": "boom"},
        )
        attrs = _llm_attributes(span.to_dict())
        assert attrs["exception.message"] == "boom"

    def test_includes_eval_scores(self):
        span = Span(
            name="openai.chat.completions",
            trace_id="t",
            attributes={"eval_scores": {"Faithfulness": 0.8}},
        )
        attrs = _llm_attributes(span.to_dict())
        assert attrs["peekr.eval.Faithfulness"] == 0.8

    def test_includes_guardrails(self):
        span = Span(
            name="openai.chat.completions",
            trace_id="t",
            attributes={
                "guardrails": {"input": {"PII": {"passed": False}}},
                "guardrail_violations": ["input.PII"],
            },
        )
        attrs = _llm_attributes(span.to_dict())
        assert "peekr.guardrails" in attrs
        assert attrs["peekr.guardrail_violations"] == "input.PII"


class TestOTelExporter:
    def test_export_uses_in_memory_provider(self):
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        memory = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(memory))
        tracer = provider.get_tracer("test")

        ex = OTelExporter(tracer=tracer)

        # Build a peekr root span + child
        import time
        root = Span(
            name="agent.run",
            trace_id="t1",
            attributes={"input": "q", "output": "a"},
        )
        root.start_time = time.time() - 1
        root.finish()
        ex.export(root)

        child = Span(
            name="openai.chat.completions",
            trace_id="t1",
            parent_id=root.span_id,
            attributes={"model": "gpt-4o", "tokens_total": 10},
        )
        child.start_time = time.time() - 0.5
        child.finish()
        ex.export(child)

        exported = memory.get_finished_spans()
        assert len(exported) == 2
        names = {s.name for s in exported}
        assert "agent.run" in names
        assert "openai.chat.completions" in names
