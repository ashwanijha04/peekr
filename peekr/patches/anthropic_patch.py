from __future__ import annotations
import json

from ..context import start_span, end_span
from ..exporters import export_span

_TRUNCATE = 1000


class _AnthropicStreamWrapper:
    """
    Wraps an Anthropic stream to capture token usage from stream events.

    Anthropic sends token counts across two event types:
      - message_start  → event.message.usage.input_tokens
      - message_delta  → event.usage.output_tokens
    """

    def __init__(self, stream, span, token):
        self._stream = stream
        self._span = span
        self._token = token
        self._done = False

    def __iter__(self):
        input_tokens = 0
        output_tokens = 0
        try:
            for event in self._stream:
                event_type = getattr(event, "type", None)
                if event_type == "message_start":
                    usage = getattr(getattr(event, "message", None), "usage", None)
                    if usage:
                        input_tokens = getattr(usage, "input_tokens", 0)
                elif event_type == "message_delta":
                    usage = getattr(event, "usage", None)
                    if usage:
                        output_tokens = getattr(usage, "output_tokens", 0)
                yield event

            if input_tokens or output_tokens:
                self._span.attributes["tokens_input"]  = input_tokens
                self._span.attributes["tokens_output"] = output_tokens
                self._span.attributes["tokens_total"]  = input_tokens + output_tokens
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


def patch_anthropic():
    try:
        import anthropic
    except ImportError:
        return

    original_create = anthropic.resources.messages.Messages.create

    def patched_create(self, *args, **kwargs):
        span, token = start_span("anthropic.messages")
        span.attributes["model"] = kwargs.get("model", "unknown")

        messages = kwargs.get("messages", [])
        if messages:
            prompt = json.dumps(messages, default=str)
            span.attributes["input"] = prompt[:_TRUNCATE] + "…" if len(prompt) > _TRUNCATE else prompt

        system = kwargs.get("system")
        if system:
            span.attributes["system"] = system[:_TRUNCATE] + "…" if len(system) > _TRUNCATE else system

        is_streaming = kwargs.get("stream", False)

        try:
            result = original_create(self, *args, **kwargs)

            if is_streaming:
                return _AnthropicStreamWrapper(result, span, token)

            usage = getattr(result, "usage", None)
            if usage:
                span.attributes["tokens_input"]  = usage.input_tokens
                span.attributes["tokens_output"] = usage.output_tokens
                span.attributes["tokens_total"]  = usage.input_tokens + usage.output_tokens
            if result.content:
                output = result.content[0].text if hasattr(result.content[0], "text") else str(result.content[0])
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

    anthropic.resources.messages.Messages.create = patched_create
