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


def patch_openai():
    try:
        import openai
    except ImportError:
        return

    original_create = openai.chat.completions.create

    def patched_create(*args, **kwargs):
        span, token = start_span("openai.chat.completions")
        span.attributes["model"] = kwargs.get("model", "unknown")

        messages = kwargs.get("messages", [])
        if messages:
            prompt = json.dumps(messages, default=str)
            span.attributes["input"] = prompt[:_TRUNCATE] + "…" if len(prompt) > _TRUNCATE else prompt

        try:
            result = original_create(*args, **kwargs)
            usage = getattr(result, "usage", None)
            if usage:
                span.attributes["tokens_input"] = usage.prompt_tokens
                span.attributes["tokens_output"] = usage.completion_tokens
                span.attributes["tokens_total"] = usage.total_tokens
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
            end_span(span, token)
            export_span(span)

    openai.chat.completions.create = patched_create
