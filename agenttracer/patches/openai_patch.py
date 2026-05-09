from ..context import start_span, end_span
from ..exporters import export_span


def patch_openai():
    try:
        import openai
    except ImportError:
        return

    # Patch sync create
    original_create = openai.chat.completions.create

    def patched_create(*args, **kwargs):
        span, token = start_span("openai.chat.completions")
        span.attributes["model"] = kwargs.get("model", "unknown")
        try:
            result = original_create(*args, **kwargs)
            usage = getattr(result, "usage", None)
            if usage:
                span.attributes["tokens_input"] = usage.prompt_tokens
                span.attributes["tokens_output"] = usage.completion_tokens
                span.attributes["tokens_total"] = usage.total_tokens
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
