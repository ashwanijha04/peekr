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


def _mark_eval_span(span) -> None:
    try:
        from ..eval import _in_eval as _peekr_in_eval
        if _peekr_in_eval.get():
            span.attributes["peekr.internal"] = True
    except Exception:
        pass


class _OpenAIStreamWrapper:
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


class _AsyncOpenAIStreamWrapper:
    def __init__(self, stream, span, token):
        self._stream = stream
        self._span = span
        self._token = token
        self._done = False

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        try:
            async for chunk in self._stream:
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

    async def __aenter__(self):
        await self._stream.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        result = await self._stream.__aexit__(exc_type, exc_val, exc_tb)
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

    # ── sync chat completions ──────────────────────────────────────────────
    original_chat_create = openai.chat.completions.create

    def patched_chat_create(*args, **kwargs):
        span, token = start_span("openai.chat.completions")
        span.attributes["model"] = kwargs.get("model", "unknown")
        _mark_eval_span(span)
        messages = kwargs.get("messages", [])
        if messages:
            prompt = json.dumps(messages, default=str)
            span.attributes["input"] = prompt[:_TRUNCATE] + "…" if len(prompt) > _TRUNCATE else prompt
        is_streaming = kwargs.get("stream", False)
        if is_streaming:
            opts = dict(kwargs.get("stream_options") or {})
            opts["include_usage"] = True
            kwargs = {**kwargs, "stream_options": opts}
        try:
            result = original_chat_create(*args, **kwargs)
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

    openai.chat.completions.create = patched_chat_create

    # ── async chat completions ─────────────────────────────────────────────
    try:
        original_async_chat_create = openai.resources.chat.completions.AsyncCompletions.create

        async def patched_async_chat_create(self_client, *args, **kwargs):
            span, token = start_span("openai.chat.completions")
            span.attributes["model"] = kwargs.get("model", "unknown")
            _mark_eval_span(span)
            messages = kwargs.get("messages", [])
            if messages:
                prompt = json.dumps(messages, default=str)
                span.attributes["input"] = prompt[:_TRUNCATE] + "…" if len(prompt) > _TRUNCATE else prompt
            is_streaming = kwargs.get("stream", False)
            if is_streaming:
                opts = dict(kwargs.get("stream_options") or {})
                opts["include_usage"] = True
                kwargs = {**kwargs, "stream_options": opts}
            try:
                result = await original_async_chat_create(self_client, *args, **kwargs)
                if is_streaming:
                    return _AsyncOpenAIStreamWrapper(result, span, token)
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

        openai.resources.chat.completions.AsyncCompletions.create = patched_async_chat_create
    except Exception:
        pass

    # ── sync embeddings ────────────────────────────────────────────────────
    try:
        original_embed_create = openai.embeddings.create

        def patched_embed_create(*args, **kwargs):
            span, token = start_span("openai.embeddings")
            span.attributes["model"] = kwargs.get("model", "unknown")
            input_val = kwargs.get("input", "")
            if isinstance(input_val, str):
                span.attributes["input_chars"] = len(input_val)
            elif isinstance(input_val, list):
                span.attributes["input_count"] = len(input_val)
            try:
                result = original_embed_create(*args, **kwargs)
                usage = getattr(result, "usage", None)
                if usage:
                    span.attributes["tokens_input"] = usage.prompt_tokens
                    span.attributes["tokens_total"] = usage.total_tokens
                span.attributes["embedding_dims"] = len(result.data[0].embedding) if result.data else 0
                span.status = "ok"
                return result
            except Exception as e:
                span.status = "error"
                span.attributes["error"] = str(e)
                raise
            finally:
                end_span(span, token)
                export_span(span)

        openai.embeddings.create = patched_embed_create
    except Exception:
        pass

    # ── async embeddings ───────────────────────────────────────────────────
    try:
        original_async_embed_create = openai.resources.embeddings.AsyncEmbeddings.create

        async def patched_async_embed_create(self_client, *args, **kwargs):
            span, token = start_span("openai.embeddings")
            span.attributes["model"] = kwargs.get("model", "unknown")
            input_val = kwargs.get("input", "")
            if isinstance(input_val, str):
                span.attributes["input_chars"] = len(input_val)
            elif isinstance(input_val, list):
                span.attributes["input_count"] = len(input_val)
            try:
                result = await original_async_embed_create(self_client, *args, **kwargs)
                usage = getattr(result, "usage", None)
                if usage:
                    span.attributes["tokens_input"] = usage.prompt_tokens
                    span.attributes["tokens_total"] = usage.total_tokens
                span.attributes["embedding_dims"] = len(result.data[0].embedding) if result.data else 0
                span.status = "ok"
                return result
            except Exception as e:
                span.status = "error"
                span.attributes["error"] = str(e)
                raise
            finally:
                end_span(span, token)
                export_span(span)

        openai.resources.embeddings.AsyncEmbeddings.create = patched_async_embed_create
    except Exception:
        pass
