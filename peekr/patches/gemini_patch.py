"""Auto-instrumentation for Google Gemini.

Supports both the modern ``google-genai`` SDK (``Client().models.generate_content``)
and the legacy ``google-generativeai`` SDK (``genai.GenerativeModel(...).generate_content``).

Captures: model, prompt, response text, prompt/output/total tokens.
Streaming responses are wrapped so token usage is captured at stream-end.
"""
from __future__ import annotations

import json
from typing import Any

from ..context import start_span, end_span
from ..exporters import export_span

_TRUNCATE = 1000


def _serialize_contents(contents: Any) -> str:
    if contents is None:
        return ""
    try:
        text = json.dumps(contents, default=str)
    except Exception:
        text = str(contents)
    return text if len(text) <= _TRUNCATE else text[:_TRUNCATE] + "…"


def _extract_text(response: Any) -> str:
    """Pull the assistant text out of either Gemini SDK's response shape."""
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
    candidates = getattr(response, "candidates", None) or []
    parts_text = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if part_text:
                parts_text.append(part_text)
    return "".join(parts_text)


def _extract_usage(response: Any) -> dict[str, int]:
    """Extract token usage from Gemini's usage_metadata."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return {}
    prompt = getattr(usage, "prompt_token_count", 0) or 0
    output = (
        getattr(usage, "candidates_token_count", None)
        or getattr(usage, "output_token_count", 0)
        or 0
    )
    total = (
        getattr(usage, "total_token_count", None)
        or (prompt + output)
    )
    return {
        "tokens_input": prompt,
        "tokens_output": output,
        "tokens_total": total,
    }


class _GeminiStreamWrapper:
    """Wraps a Gemini streaming response to capture usage at stream-end."""

    def __init__(self, stream, span, token):
        self._stream = stream
        self._span = span
        self._token = token
        self._done = False

    def __iter__(self):
        try:
            last_chunk = None
            for chunk in self._stream:
                last_chunk = chunk
                yield chunk
            if last_chunk is not None:
                usage = _extract_usage(last_chunk)
                if usage:
                    self._span.attributes.update(usage)
            self._span.status = "ok"
        except Exception as exc:
            self._span.status = "error"
            self._span.attributes["error"] = str(exc)
            raise
        finally:
            self._finish()

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def _finish(self):
        if not self._done:
            self._done = True
            end_span(self._span, self._token)
            export_span(self._span)


def _wrap_generate(original, *, is_method: bool, name: str):
    def patched(*args, **kwargs):
        span, token = start_span(name)
        if is_method:
            self = args[0]
            call_args = args[1:]
            model = kwargs.get("model") or getattr(self, "model_name", "unknown")
        else:
            self = None
            call_args = args
            model = kwargs.get("model") or (call_args[0] if call_args else "unknown")
        span.attributes["model"] = model

        contents = kwargs.get("contents")
        if contents is None and call_args:
            # positional contents arg
            for arg in call_args:
                if isinstance(arg, (str, list, dict)):
                    contents = arg
                    break
        if contents is not None:
            span.attributes["input"] = _serialize_contents(contents)

        is_streaming = bool(
            kwargs.get("stream")
            or "stream" in name
            or "stream" in (original.__name__ if hasattr(original, "__name__") else "")
        )

        try:
            result = original(*args, **kwargs)
            if is_streaming and hasattr(result, "__iter__"):
                return _GeminiStreamWrapper(result, span, token)

            usage = _extract_usage(result)
            if usage:
                span.attributes.update(usage)
            output = _extract_text(result)
            if output:
                span.attributes["output"] = (
                    output[:_TRUNCATE] + "…" if len(output) > _TRUNCATE else output
                )
            span.status = "ok"
            return result
        except Exception as exc:
            span.status = "error"
            span.attributes["error"] = str(exc)
            raise
        finally:
            if not is_streaming:
                end_span(span, token)
                export_span(span)

    return patched


def patch_gemini() -> None:
    """Patch both google-genai (new) and google-generativeai (legacy) SDKs."""
    _patch_google_genai()
    _patch_google_generativeai()


def _patch_google_genai() -> None:
    try:
        from google.genai import models as _models_mod  # type: ignore
    except Exception:
        return

    Models = getattr(_models_mod, "Models", None)
    if Models is None:
        return

    if not getattr(Models.generate_content, "_peekr_patched", False):
        original = Models.generate_content
        wrapped = _wrap_generate(original, is_method=True, name="gemini.generate_content")
        wrapped._peekr_patched = True  # type: ignore[attr-defined]
        Models.generate_content = wrapped

    if hasattr(Models, "generate_content_stream") and not getattr(
        Models.generate_content_stream, "_peekr_patched", False
    ):
        original = Models.generate_content_stream
        wrapped = _wrap_generate(original, is_method=True, name="gemini.generate_content_stream")
        wrapped._peekr_patched = True  # type: ignore[attr-defined]
        Models.generate_content_stream = wrapped


def _patch_google_generativeai() -> None:
    try:
        import google.generativeai as genai  # type: ignore
    except Exception:
        return

    GenerativeModel = getattr(genai, "GenerativeModel", None)
    if GenerativeModel is None:
        return

    if not getattr(GenerativeModel.generate_content, "_peekr_patched", False):
        original = GenerativeModel.generate_content
        wrapped = _wrap_generate(original, is_method=True, name="gemini.generate_content")
        wrapped._peekr_patched = True  # type: ignore[attr-defined]
        GenerativeModel.generate_content = wrapped
