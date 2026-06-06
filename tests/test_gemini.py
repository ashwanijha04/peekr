"""Smoke tests for the Gemini patch.

The full SDK isn't installed in CI by default, so we mock the modules.
What we verify here:
  - patch_gemini() is a no-op when neither google-genai nor
    google-generativeai is importable (must not raise).
  - The text / usage extraction helpers handle both response shapes.
  - Patched calls produce a span with model + tokens captured.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from peekr.patches.gemini_patch import (
    patch_gemini,
    _extract_text,
    _extract_usage,
    _serialize_contents,
    _wrap_generate,
)
from peekr.exporters import add_exporter, _exporters
from peekr.span import Span


@pytest.fixture(autouse=True)
def _clear_exporters():
    """Each test starts with no registered exporters."""
    saved = list(_exporters)
    _exporters.clear()
    yield
    _exporters.clear()
    _exporters.extend(saved)


class TestNoOpWhenSDKMissing:
    def test_patch_gemini_safe_without_sdks(self):
        """Calling patch_gemini() with no Gemini SDK installed must not raise."""
        # Belt and braces — even if google.genai is somehow importable in
        # this env, the call must be safe to invoke twice (idempotent).
        patch_gemini()
        patch_gemini()


class TestExtractors:
    def test_extract_text_from_response_text_attr(self):
        resp = MagicMock(text="hello world", candidates=None)
        assert _extract_text(resp) == "hello world"

    def test_extract_text_from_candidates(self):
        part = MagicMock(text="from candidate")
        content = MagicMock(parts=[part])
        candidate = MagicMock(content=content)
        resp = MagicMock(text=None, candidates=[candidate])
        assert "from candidate" in _extract_text(resp)

    def test_extract_usage_handles_missing(self):
        resp = MagicMock(spec=[])  # no usage_metadata attribute
        assert _extract_usage(resp) == {}

    def test_extract_usage_reads_metadata(self):
        meta = MagicMock(
            prompt_token_count=12,
            candidates_token_count=5,
            total_token_count=17,
        )
        resp = MagicMock(usage_metadata=meta)
        usage = _extract_usage(resp)
        assert usage["tokens_input"] == 12
        assert usage["tokens_output"] == 5
        assert usage["tokens_total"] == 17

    def test_serialize_contents_truncates(self):
        long = "x" * 5000
        out = _serialize_contents(long)
        assert len(out) <= 1100
        assert out.endswith("…")


class TestWrappedCall:
    def test_wrapped_call_emits_span(self):
        captured: list[Span] = []

        class _Capture:
            def export(self, span):
                captured.append(span)

        add_exporter(_Capture())

        # Stand-in for Models.generate_content
        def fake_generate(self, contents, model="gemini-2.0-flash"):
            resp = MagicMock(text="hello")
            resp.usage_metadata = MagicMock(
                prompt_token_count=10,
                candidates_token_count=2,
                total_token_count=12,
            )
            resp.candidates = None
            return resp

        wrapped = _wrap_generate(
            fake_generate, is_method=True, name="gemini.generate_content"
        )

        class FakeModels:
            model_name = "gemini-2.0-flash"

        wrapped(FakeModels(), "Hello?", model="gemini-2.0-flash")

        assert len(captured) == 1
        span = captured[0]
        assert span.name == "gemini.generate_content"
        assert span.attributes["model"] == "gemini-2.0-flash"
        assert span.attributes["tokens_total"] == 12
        assert span.attributes["output"] == "hello"
        assert span.status == "ok"
