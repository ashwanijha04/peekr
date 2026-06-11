from __future__ import annotations
import json

from ..context import start_span, end_span, GuardrailError, _run_input_guards
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
                self._span.attributes["tokens_input"] = input_tokens
                self._span.attributes["tokens_output"] = output_tokens
                self._span.attributes["tokens_total"] = input_tokens + output_tokens
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
    # `import anthropic` alone does not always populate the
    # `anthropic.resources` attribute — Python only resolves submodules
    # as attributes once they have been imported. Some test environments
    # (notably ones where `anthropic` is mocked or only partially imported)
    # then crash with AttributeError on the dotted-path lookup. Force the
    # submodule via an explicit from-import and catch both shapes of
    # failure so this function never raises — otherwise `_patched` never
    # gets set and every subsequent `instrument()` retries it.
    try:
        from anthropic.resources.messages import Messages  # type: ignore
    except (ImportError, AttributeError):
        return

    if getattr(Messages.create, "_peekr_patched", False):
        return  # idempotent — safe to call instrument() multiple times

    original_create = Messages.create

    def patched_create(self, *args, **kwargs):
        try:
            from ..eval import _in_eval as _peekr_eval_guard

            if _peekr_eval_guard.get():
                return original_create(self, *args, **kwargs)
        except Exception:
            pass

        span, token = start_span("anthropic.messages")
        span.attributes["model"] = kwargs.get("model", "unknown")
        # Tag judge-LLM spans so the dashboard can hide them. The recursion
        # guard already stops them from being scored; this stops them from
        # appearing as duplicate cards in the worst-offenders panel.
        try:
            from ..eval import _in_eval as _peekr_in_eval

            if _peekr_in_eval.get():
                span.attributes["peekr.internal"] = True
        except Exception:  # pragma: no cover
            pass

        messages = kwargs.get("messages", []) or []
        system = kwargs.get("system")

        # Merge system into the messages array under role="system" so consumers
        # written for the OpenAI/chat shape (peekr dashboard, evaluators using
        # `attributes.input`) see the system prompt without special-casing the
        # Anthropic schema. The standalone `attributes.system` is also kept so
        # SQLite queries that already filter on it keep working.
        unified_messages: list = list(messages)
        if system:
            sys_str = (
                system if isinstance(system, str) else json.dumps(system, default=str)
            )
            unified_messages = [
                {"role": "system", "content": sys_str},
                *unified_messages,
            ]
            span.attributes["system"] = (
                sys_str[:_TRUNCATE] + "…" if len(sys_str) > _TRUNCATE else sys_str
            )

        if unified_messages:
            prompt = json.dumps(unified_messages, default=str)
            span.attributes["input"] = (
                prompt[:_TRUNCATE] + "…" if len(prompt) > _TRUNCATE else prompt
            )

        is_streaming = kwargs.get("stream", False)

        try:
            _run_input_guards(span)
            result = original_create(self, *args, **kwargs)

            if is_streaming:
                return _AnthropicStreamWrapper(result, span, token)

            usage = getattr(result, "usage", None)
            if usage:
                span.attributes["tokens_input"] = usage.input_tokens
                span.attributes["tokens_output"] = usage.output_tokens
                span.attributes["tokens_total"] = (
                    usage.input_tokens + usage.output_tokens
                )
            if result.content:
                output = (
                    result.content[0].text
                    if hasattr(result.content[0], "text")
                    else str(result.content[0])
                )
                span.attributes["output"] = (
                    output[:_TRUNCATE] + "…" if len(output) > _TRUNCATE else output
                )
            span.status = "ok"
            return result
        except GuardrailError:
            raise
        except Exception as e:
            span.status = "error"
            span.attributes["error"] = str(e)
            raise
        finally:
            if not is_streaming:
                end_span(span, token)
                export_span(span)

    patched_create._peekr_patched = True  # type: ignore[attr-defined]
    Messages.create = patched_create

    # ── AsyncMessages.create ───────────────────────────────────────────────
    try:
        from anthropic.resources.messages import AsyncMessages  # type: ignore

        if getattr(AsyncMessages.create, "_peekr_patched", False):
            return

        original_async_create = AsyncMessages.create

        async def patched_async_create(self, *args, **kwargs):
            try:
                from ..eval import _in_eval as _peekr_eval_guard

                if _peekr_eval_guard.get():
                    return await original_async_create(self, *args, **kwargs)
            except Exception:
                pass

            span, token = start_span("anthropic.messages")
            span.attributes["model"] = kwargs.get("model", "unknown")
            try:
                from ..eval import _in_eval as _peekr_in_eval

                if _peekr_in_eval.get():
                    span.attributes["peekr.internal"] = True
            except Exception:
                pass

            messages = kwargs.get("messages", []) or []
            system = kwargs.get("system")
            unified_messages: list = list(messages)
            if system:
                sys_str = (
                    system
                    if isinstance(system, str)
                    else json.dumps(system, default=str)
                )
                unified_messages = [
                    {"role": "system", "content": sys_str},
                    *unified_messages,
                ]
                span.attributes["system"] = (
                    sys_str[:_TRUNCATE] + "…" if len(sys_str) > _TRUNCATE else sys_str
                )
            if unified_messages:
                prompt = json.dumps(unified_messages, default=str)
                span.attributes["input"] = (
                    prompt[:_TRUNCATE] + "…" if len(prompt) > _TRUNCATE else prompt
                )

            is_streaming = kwargs.get("stream", False)
            try:
                result = await original_async_create(self, *args, **kwargs)
                if is_streaming:
                    return _AnthropicStreamWrapper(result, span, token)
                usage = getattr(result, "usage", None)
                if usage:
                    span.attributes["tokens_input"] = usage.input_tokens
                    span.attributes["tokens_output"] = usage.output_tokens
                    span.attributes["tokens_total"] = (
                        usage.input_tokens + usage.output_tokens
                    )
                if result.content:
                    output = (
                        result.content[0].text
                        if hasattr(result.content[0], "text")
                        else str(result.content[0])
                    )
                    span.attributes["output"] = (
                        output[:_TRUNCATE] + "…" if len(output) > _TRUNCATE else output
                    )
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

        patched_async_create._peekr_patched = True  # type: ignore[attr-defined]
        AsyncMessages.create = patched_async_create

    except (ImportError, AttributeError):
        pass
