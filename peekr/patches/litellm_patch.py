from __future__ import annotations

from ..context import start_span, end_span
from ..exporters import export_span

_TRUNCATE = 1000

_PROVIDER_PREFIXES = (
    ("gpt-", "openai"),
    ("o1", "openai"),
    ("o3", "openai"),
    ("o4", "openai"),
    ("claude-", "anthropic"),
    ("command", "cohere"),
    ("mistral", "mistral"),
    ("mixtral", "mistral"),
    ("gemini", "google"),
    ("palm", "google"),
    ("llama", "meta"),
    ("titan", "amazon"),
    ("nova", "amazon"),
    ("j2-", "ai21"),
    ("jamba", "ai21"),
    ("cohere.", "cohere"),
    ("anthropic/", "anthropic"),
    ("openai/", "openai"),
    ("mistral/", "mistral"),
    ("google/", "google"),
    ("meta/", "meta"),
    ("meta-llama/", "meta"),
)


def _derive_provider(model: str) -> str:
    if not model or model == "unknown":
        return "unknown"
    lower = model.lower()
    # handle "provider/model" slash notation first
    if "/" in lower:
        prefix = lower.split("/")[0]
        if prefix in (
            "openai",
            "anthropic",
            "cohere",
            "mistral",
            "google",
            "meta",
            "amazon",
            "ai21",
            "bedrock",
            "groq",
            "together",
            "deepinfra",
            "perplexity",
            "fireworks",
        ):
            return prefix
    for prefix, provider in _PROVIDER_PREFIXES:
        if lower.startswith(prefix):
            return provider
    return "unknown"


def _extract_output(response) -> str:
    try:
        return response.choices[0].message.content or ""
    except Exception:
        return ""


def _extract_usage(response, span) -> None:
    try:
        hidden = getattr(response, "_hidden_params", {}) or {}
        prompt_tokens = hidden.get("prompt_tokens")
        completion_tokens = hidden.get("completion_tokens")
        usage = getattr(response, "usage", None)
        if prompt_tokens is None and usage:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
        if completion_tokens is None and usage:
            completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = None
        if usage:
            total_tokens = getattr(usage, "total_tokens", None)
        if prompt_tokens is not None:
            span.attributes["tokens_input"] = prompt_tokens
        if completion_tokens is not None:
            span.attributes["tokens_output"] = completion_tokens
        if total_tokens is not None:
            span.attributes["tokens_total"] = total_tokens
    except Exception:
        pass


class _LiteLLMStreamWrapper:
    def __init__(self, stream, span, token):
        self._stream = stream
        self._span = span
        self._token = token
        self._done = False
        self._parts: list[str] = []

    def __iter__(self):
        try:
            for chunk in self._stream:
                # try usage from _hidden_params first, then .usage
                try:
                    hidden = getattr(chunk, "_hidden_params", {}) or {}
                    prompt_tokens = hidden.get("prompt_tokens")
                    completion_tokens = hidden.get("completion_tokens")
                    usage = getattr(chunk, "usage", None)
                    if prompt_tokens is None and usage:
                        prompt_tokens = getattr(usage, "prompt_tokens", None)
                    if completion_tokens is None and usage:
                        completion_tokens = getattr(usage, "completion_tokens", None)
                    total_tokens = (
                        getattr(usage, "total_tokens", None) if usage else None
                    )
                    if prompt_tokens is not None:
                        self._span.attributes["tokens_input"] = prompt_tokens
                    if completion_tokens is not None:
                        self._span.attributes["tokens_output"] = completion_tokens
                    if total_tokens is not None:
                        self._span.attributes["tokens_total"] = total_tokens
                except Exception:
                    pass
                try:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        self._parts.append(delta)
                except (AttributeError, IndexError):
                    pass
                yield chunk
            if self._parts:
                out = "".join(self._parts)
                self._span.attributes["output"] = (
                    out[:_TRUNCATE] + "…" if len(out) > _TRUNCATE else out
                )
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


class _AsyncLiteLLMStreamWrapper:
    def __init__(self, stream, span, token):
        self._stream = stream
        self._span = span
        self._token = token
        self._done = False
        self._parts: list[str] = []

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        try:
            async for chunk in self._stream:
                try:
                    hidden = getattr(chunk, "_hidden_params", {}) or {}
                    prompt_tokens = hidden.get("prompt_tokens")
                    completion_tokens = hidden.get("completion_tokens")
                    usage = getattr(chunk, "usage", None)
                    if prompt_tokens is None and usage:
                        prompt_tokens = getattr(usage, "prompt_tokens", None)
                    if completion_tokens is None and usage:
                        completion_tokens = getattr(usage, "completion_tokens", None)
                    total_tokens = (
                        getattr(usage, "total_tokens", None) if usage else None
                    )
                    if prompt_tokens is not None:
                        self._span.attributes["tokens_input"] = prompt_tokens
                    if completion_tokens is not None:
                        self._span.attributes["tokens_output"] = completion_tokens
                    if total_tokens is not None:
                        self._span.attributes["tokens_total"] = total_tokens
                except Exception:
                    pass
                try:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        self._parts.append(delta)
                except (AttributeError, IndexError):
                    pass
                yield chunk
            if self._parts:
                out = "".join(self._parts)
                self._span.attributes["output"] = (
                    out[:_TRUNCATE] + "…" if len(out) > _TRUNCATE else out
                )
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


def _make_completion_patch(original):
    def patched(*args, **kwargs):
        model = kwargs.get("model", "unknown")
        span, token = start_span("litellm.completion")
        span.attributes["model"] = model
        span.attributes["provider"] = _derive_provider(model)
        messages = kwargs.get("messages", "")
        inp = str(messages)
        span.attributes["input"] = (
            inp[:_TRUNCATE] + "…" if len(inp) > _TRUNCATE else inp
        )
        is_streaming = kwargs.get("stream", False)
        try:
            result = original(*args, **kwargs)
            if is_streaming:
                return _LiteLLMStreamWrapper(result, span, token)
            _extract_usage(result, span)
            output = _extract_output(result)
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

    return patched


def _make_acompletion_patch(original):
    async def patched(*args, **kwargs):
        model = kwargs.get("model", "unknown")
        span, token = start_span("litellm.completion")
        span.attributes["model"] = model
        span.attributes["provider"] = _derive_provider(model)
        messages = kwargs.get("messages", "")
        inp = str(messages)
        span.attributes["input"] = (
            inp[:_TRUNCATE] + "…" if len(inp) > _TRUNCATE else inp
        )
        is_streaming = kwargs.get("stream", False)
        try:
            result = await original(*args, **kwargs)
            if is_streaming:
                return _AsyncLiteLLMStreamWrapper(result, span, token)
            _extract_usage(result, span)
            output = _extract_output(result)
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

    return patched


def patch_litellm() -> None:
    try:
        import litellm
    except ImportError:
        return
    if getattr(litellm, "_peekr_patched", False):
        return

    # ── sync completion ───────────────────────────────────────────────────────
    try:
        orig_completion = litellm.completion
        if not getattr(orig_completion, "_peekr_patched", False):
            patched_completion = _make_completion_patch(orig_completion)
            patched_completion._peekr_patched = True
            litellm.completion = patched_completion
    except Exception:
        pass

    # ── async completion ──────────────────────────────────────────────────────
    try:
        orig_acompletion = litellm.acompletion
        if not getattr(orig_acompletion, "_peekr_patched", False):
            patched_acompletion = _make_acompletion_patch(orig_acompletion)
            patched_acompletion._peekr_patched = True
            litellm.acompletion = patched_acompletion
    except Exception:
        pass

    litellm._peekr_patched = True
