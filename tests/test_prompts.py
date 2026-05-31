"""Tests for peekr.prompts — no real API key needed."""
from __future__ import annotations
import unittest.mock as mock
import json
import pytest
from peekr.prompts import Prompt, get


# ── Prompt.render ─────────────────────────────────────────────────────────────

class TestPromptRender:
    def _make(self, content: str, variables: list) -> Prompt:
        return Prompt("test", 1, content, variables, None, None)

    def test_no_variables(self):
        p = self._make("You are helpful.", [])
        assert p.render() == "You are helpful."

    def test_single_variable(self):
        p = self._make("Context: {{context}}", ["context"])
        assert p.render(context="hello world") == "Context: hello world"

    def test_multiple_variables(self):
        p = self._make("{{a}} and {{b}}", ["a", "b"])
        assert p.render(a="foo", b="bar") == "foo and bar"

    def test_extra_kwargs_ignored(self):
        p = self._make("{{x}}", ["x"])
        # extra kwarg shouldn't raise
        assert p.render(x="val", unused="ignored") == "val"

    def test_missing_variable_raises(self):
        p = self._make("{{required}}", ["required"])
        with pytest.raises(KeyError, match="required"):
            p.render()

    def test_multiline_content(self):
        content = "System: {{role}}\n\nUser context: {{ctx}}"
        p = self._make(content, ["role", "ctx"])
        result = p.render(role="assistant", ctx="docs")
        assert result == "System: assistant\n\nUser context: docs"

    def test_repr(self):
        p = self._make("{{x}}", ["x"])
        r = repr(p)
        assert "test" in r and "version=1" in r


# ── prompts.get ───────────────────────────────────────────────────────────────

class TestPromptGet:
    def _mock_response(self, data: dict):
        import io
        resp = mock.MagicMock()
        resp.__enter__ = lambda s: s
        resp.__exit__ = mock.MagicMock(return_value=False)
        resp.read.return_value = json.dumps(data).encode()
        return resp

    def test_returns_prompt(self):
        payload = {
            "name": "rag_answer",
            "version": 3,
            "content": "Answer using {{context}}.",
            "variables": ["context"],
            "model": "gpt-4o-mini",
            "notes": "Improved",
        }
        with mock.patch("urllib.request.urlopen", return_value=self._mock_response(payload)):
            p = get("rag_answer", api_key="pk_live_test")

        assert p.name == "rag_answer"
        assert p.version == 3
        assert p.model == "gpt-4o-mini"
        assert p.variables == ["context"]

    def test_missing_api_key_raises(self):
        with pytest.raises(ValueError, match="API key required"):
            get("any", api_key=None)

    def test_prompt_not_found_raises(self):
        import urllib.error
        error = urllib.error.HTTPError(
            url="", code=404, msg="Not Found", hdrs=None, fp=None  # type: ignore
        )
        error.read = lambda: json.dumps({"error": "prompt_not_found"}).encode()
        with mock.patch("urllib.request.urlopen", side_effect=error):
            with pytest.raises(ValueError, match="not found"):
                get("missing", api_key="pk_live_test")

    def test_render_after_get(self):
        payload = {
            "name": "tmpl",
            "version": 1,
            "content": "Hello {{name}}!",
            "variables": ["name"],
            "model": None,
            "notes": None,
        }
        with mock.patch("urllib.request.urlopen", return_value=self._mock_response(payload)):
            p = get("tmpl", api_key="pk_live_test")
        assert p.render(name="Alice") == "Hello Alice!"
