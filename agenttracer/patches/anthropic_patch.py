from ..context import start_span, end_span
from ..exporters import export_span


def patch_anthropic():
    try:
        import anthropic
    except ImportError:
        return

    original_create = anthropic.resources.messages.Messages.create

    def patched_create(self, *args, **kwargs):
        span, token = start_span("anthropic.messages")
        span.attributes["model"] = kwargs.get("model", "unknown")
        try:
            result = original_create(self, *args, **kwargs)
            usage = getattr(result, "usage", None)
            if usage:
                span.attributes["tokens_input"] = usage.input_tokens
                span.attributes["tokens_output"] = usage.output_tokens
                span.attributes["tokens_total"] = usage.input_tokens + usage.output_tokens
            span.status = "ok"
            return result
        except Exception as e:
            span.status = "error"
            span.attributes["error"] = str(e)
            raise
        finally:
            end_span(span, token)
            export_span(span)

    anthropic.resources.messages.Messages.create = patched_create
