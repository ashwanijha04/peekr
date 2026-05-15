from __future__ import annotations
import json

from ..context import start_span, end_span
from ..exporters import export_span

_TRUNCATE = 1000
_BEDROCK_SERVICE = "bedrock-runtime"
_PATCHED_OPS = ("Converse", "ConverseStream")


class _BedrockStreamWrapper:
    """
    Wraps a Bedrock EventStream to capture token usage.

    Bedrock sends token counts in a 'metadata' event:
      event["metadata"]["usage"]["inputTokens"]
      event["metadata"]["usage"]["outputTokens"]
    """

    def __init__(self, stream, span, token):
        self._stream = stream
        self._span = span
        self._token = token
        self._done = False

    def __iter__(self):
        try:
            for event in self._stream:
                if "metadata" in event:
                    usage = event["metadata"].get("usage", {})
                    inp = usage.get("inputTokens", 0)
                    out = usage.get("outputTokens", 0)
                    if inp or out:
                        self._span.attributes["tokens_input"]  = inp
                        self._span.attributes["tokens_output"] = out
                        self._span.attributes["tokens_total"]  = inp + out
                yield event
            self._span.status = "ok"
        except Exception as e:
            self._span.status = "error"
            self._span.attributes["error"] = str(e)
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


def patch_bedrock():
    try:
        import botocore.client
    except ImportError:
        return

    original = botocore.client.BaseClient._make_api_call

    def patched(self, operation_name, api_params):
        try:
            service = self.meta.service_model.service_name
        except Exception:
            return original(self, operation_name, api_params)

        if service != _BEDROCK_SERVICE or operation_name not in _PATCHED_OPS:
            return original(self, operation_name, api_params)

        is_streaming = operation_name == "ConverseStream"
        span_name = "bedrock.converse_stream" if is_streaming else "bedrock.converse"
        span, token = start_span(span_name)
        span.attributes["model"] = api_params.get("modelId", "unknown")
        try:
            from ..eval import _in_eval as _peekr_in_eval
            if _peekr_in_eval.get():
                span.attributes["peekr.internal"] = True
        except Exception:  # pragma: no cover
            pass

        messages = api_params.get("messages", [])
        if messages:
            prompt = json.dumps(messages, default=str)
            span.attributes["input"] = prompt[:_TRUNCATE] + "…" if len(prompt) > _TRUNCATE else prompt

        system = api_params.get("system")
        if system:
            s = json.dumps(system, default=str)
            span.attributes["system"] = s[:_TRUNCATE] + "…" if len(s) > _TRUNCATE else s

        try:
            result = original(self, operation_name, api_params)

            if is_streaming:
                result["stream"] = _BedrockStreamWrapper(result["stream"], span, token)
                return result

            usage = result.get("usage", {})
            if usage:
                span.attributes["tokens_input"]  = usage.get("inputTokens", 0)
                span.attributes["tokens_output"] = usage.get("outputTokens", 0)
                span.attributes["tokens_total"]  = usage.get("totalTokens", 0)

            try:
                output = result["output"]["message"]["content"][0]["text"]
                span.attributes["output"] = output[:_TRUNCATE] + "…" if len(output) > _TRUNCATE else output
            except (KeyError, IndexError, TypeError):
                pass

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

    botocore.client.BaseClient._make_api_call = patched
