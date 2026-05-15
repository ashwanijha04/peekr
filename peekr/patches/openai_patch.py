from __future__ import annotations
import json

from ..context import start_span, end_span
from ..exporters import export_span

_TRUNCATE = 1000


def _extract_text(choice) -> str:
    try:
        return choice.message.content or ""
    except Exception:
        return ""


class _OpenAIStreamWrapper:
    """
    Wraps an OpenAI stream to capture token usage when the stream finishes.

    OpenAI sends a final chunk with usage data (and no content) when
    stream_options={"include_usage": True} is set. We inject that option
    automatically and read usage off the final chunk.
    """

    def __init__(self, stream, span, token):
        self._stream = stream
        self._span = span
        self._token = token
        self._done = False

    def __iter__(self):
        try:
            for chunk in self._stream:
                usage = getattr(chunk, "usage", None)
                if usage:
                    self._span.attributes["tokens_input"]  = usage.prompt_tokens
                    self._span.attributes["tokens_output"] = usage.completion_tokens
                    self._span.attributes["tokens_total"]  = usage.total_tokens
                yield chunk
            self._span.status = "ok"
        except Exception as e:
            self._span.status = "error"
            self._span.attributes["error"] = str(e)
            raise
        finally:
            self._finish()

    def __enter__(self):
        self._stream.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        result = self._stream.__exit__(exc_type, exc_val, exc_tb)
        if exc_type and not self._done:
            self._span.status = "error"
            self._span.attributes["error"] = str(exc_val)
        self._finish()
        return result

    def __getattr__(self, name):
        return getattr(self._stream, name)

    def _finish(self):
        if not self._done:
            self._done = True
            end_span(self._span, self._token)
            export_span(self._span)


def patch_openai():
    try:
        import openai
    except ImportError:
        return

    original_create = openai.chat.completions.create

    def patched_create(*args, **kwargs):
        span, token = start_span("openai.chat.completions")
        span.attributes["model"] = kwargs.get("model", "unknown")
        # Mark spans created while inside an evaluator (LLM-as-judge) so the
        # dashboard / aggregations can filter them out. They're still written
        # to JSONL so users can audit judge costs if they want.
        try:
            from ..eval import _in_eval as _peekr_in_eval
            if _peekr_in_eval.get():
                span.attributes["peekr.internal"] = True
        except Exception:  # pragma: no cover — eval may not be importable in trim builds
            pass

        messages = kwargs.get("messages", [])
        if messages:
            prompt = json.dumps(messages, default=str)
            span.attributes["input"] = prompt[:_TRUNCATE] + "…" if len(prompt) > _TRUNCATE else prompt

        is_streaming = kwargs.get("stream", False)

        if is_streaming:
            # Inject include_usage so the final chunk carries token counts
            opts = dict(kwargs.get("stream_options") or {})
            opts["include_usage"] = True
            kwargs = {**kwargs, "stream_options": opts}

        try:
            result = original_create(*args, **kwargs)

            if is_streaming:
                return _OpenAIStreamWrapper(result, span, token)

            usage = getattr(result, "usage", None)
            if usage:
                span.attributes["tokens_input"]  = usage.prompt_tokens
                span.attributes["tokens_output"] = usage.completion_tokens
                span.attributes["tokens_total"]  = usage.total_tokens
            if result.choices:
                output = _extract_text(result.choices[0])
                span.attributes["output"] = output[:_TRUNCATE] + "…" if len(output) > _TRUNCATE else output
            span.status = "ok"
            return result
        except Exception as e:
            span.status = "error"
            span.attributes["error"] = str(e)
            raise
        finally:
            if not is_streaming:
                end_span(span, token)
                export_span(span)

    openai.chat.completions.create = patched_create
