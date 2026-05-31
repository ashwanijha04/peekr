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


def _make_chat_patch(original):
    def patched(self_or_first, *args, **kwargs):
        # Works for both bound (instance) and unbound calls
        if callable(getattr(self_or_first, 'model', None)):
            # called as unbound — self_or_first IS the instance
            bound_args = args
            bound_self = self_or_first
        else:
            bound_args = (self_or_first,) + args
            bound_self = None

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
            if bound_self is not None:
                result = original(bound_self, *bound_args, **kwargs)
            else:
                result = original(*bound_args, **kwargs)
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
    return patched


def _make_async_chat_patch(original):
    async def patched(self_client, *args, **kwargs):
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
            result = await original(self_client, *args, **kwargs)
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
    return patched


def _make_embed_patch(original):
    def patched(self_client, *args, **kwargs):
        span, token = start_span("openai.embeddings")
        span.attributes["model"] = kwargs.get("model", "unknown")
        input_val = kwargs.get("input", "")
        if isinstance(input_val, str):
            span.attributes["input_chars"] = len(input_val)
        elif isinstance(input_val, list):
            span.attributes["input_count"] = len(input_val)
        try:
            result = original(self_client, *args, **kwargs)
            usage = getattr(result, "usage", None)
            if usage:
                span.attributes["tokens_input"] = usage.prompt_tokens
                span.attributes["tokens_total"] = usage.total_tokens
            if result.data:
                span.attributes["embedding_dims"] = len(result.data[0].embedding)
            span.status = "ok"
            return result
        except Exception as e:
            span.status = "error"
            span.attributes["error"] = str(e)
            raise
        finally:
            end_span(span, token)
            export_span(span)
    return patched


def _make_async_embed_patch(original):
    async def patched(self_client, *args, **kwargs):
        span, token = start_span("openai.embeddings")
        span.attributes["model"] = kwargs.get("model", "unknown")
        input_val = kwargs.get("input", "")
        if isinstance(input_val, str):
            span.attributes["input_chars"] = len(input_val)
        elif isinstance(input_val, list):
            span.attributes["input_count"] = len(input_val)
        try:
            result = await original(self_client, *args, **kwargs)
            usage = getattr(result, "usage", None)
            if usage:
                span.attributes["tokens_input"] = usage.prompt_tokens
                span.attributes["tokens_total"] = usage.total_tokens
            if result.data:
                span.attributes["embedding_dims"] = len(result.data[0].embedding)
            span.status = "ok"
            return result
        except Exception as e:
            span.status = "error"
            span.attributes["error"] = str(e)
            raise
        finally:
            end_span(span, token)
            export_span(span)
    return patched


def patch_openai():
    try:
        import openai
    except ImportError:
        return

    # ── sync chat completions — patch the CLASS so all instances are covered ──
    try:
        Completions = openai.resources.chat.completions.Completions
        if not getattr(Completions.create, "_peekr_patched", False):
            orig = Completions.create
            Completions.create = _make_chat_patch(orig)
            Completions.create._peekr_patched = True
    except Exception:
        pass

    # ── async chat completions ────────────────────────────────────────────────
    try:
        AsyncCompletions = openai.resources.chat.completions.AsyncCompletions
        if not getattr(AsyncCompletions.create, "_peekr_patched", False):
            orig = AsyncCompletions.create
            AsyncCompletions.create = _make_async_chat_patch(orig)
            AsyncCompletions.create._peekr_patched = True
    except Exception:
        pass

    # ── sync embeddings ───────────────────────────────────────────────────────
    try:
        Embeddings = openai.resources.embeddings.Embeddings
        if not getattr(Embeddings.create, "_peekr_patched", False):
            orig = Embeddings.create
            Embeddings.create = _make_embed_patch(orig)
            Embeddings.create._peekr_patched = True
    except Exception:
        pass

    # ── async embeddings ──────────────────────────────────────────────────────
    try:
        AsyncEmbeddings = openai.resources.embeddings.AsyncEmbeddings
        if not getattr(AsyncEmbeddings.create, "_peekr_patched", False):
            orig = AsyncEmbeddings.create
            AsyncEmbeddings.create = _make_async_embed_patch(orig)
            AsyncEmbeddings.create._peekr_patched = True
    except Exception:
        pass
