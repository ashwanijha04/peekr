"""Shared LLM-as-judge plumbing for the eval module.

Centralises provider selection so the rule "do I have a working judge?"
lives in exactly one place. Both `Hallucination` and `Rubric` delegate
to `call_judge()` here — fixing them in lockstep.

Why this exists: previous versions checked only whether the `openai`
package was importable. If `openai` was a transitive dependency but no
`OPENAI_API_KEY` was set, the judge call raised `AuthenticationError`,
`EvalExporter` caught it and silently stored `0.0`, and every span on
the dashboard read as "fully hallucinated" — with no hint the judge
itself had failed. Reported by users running peekr against a real
Anthropic-only workload.

The fix: check *credentials*, not just *importability*. And distinguish
"judge unavailable" (`JudgeUnavailable`) from "judge ran and rated 0.0"
so the storage path can flag the former without overwriting genuine
low scores.
"""

from __future__ import annotations

import os
from typing import Literal, Optional

try:
    import openai  # type: ignore
except ImportError:
    openai = None  # type: ignore

try:
    import anthropic  # type: ignore
except ImportError:
    anthropic = None  # type: ignore


JudgeProvider = Literal["auto", "openai", "anthropic"]


class JudgeUnavailable(RuntimeError):
    """Raised when no LLM judge can be invoked.

    Distinct from arbitrary LLM-side errors (rate limits, network, malformed
    response) so the EvalExporter can present a meaningful "judge missing"
    state to the user rather than scoring everything as fully hallucinated.
    """


def select_provider(preference: JudgeProvider = "auto") -> str:
    """Decide which SDK to call.

    - ``"openai"``     → require the openai SDK; raise if missing.
    - ``"anthropic"``  → require the anthropic SDK; raise if missing.
    - ``"auto"``       → prefer whichever provider has BOTH a usable SDK
      AND credentials (env var). If only one has both, pick it. If neither
      has credentials, fall back to whichever SDK is importable so ambient
      auth flows still work (Azure cred providers, on-prem proxies, etc.).
      Raises ``JudgeUnavailable`` if neither SDK is importable.
    """
    if preference == "openai":
        if openai is None:
            raise JudgeUnavailable(
                "judge_provider='openai' but the openai package is not installed"
            )
        return "openai"
    if preference == "anthropic":
        if anthropic is None:
            raise JudgeUnavailable(
                "judge_provider='anthropic' but the anthropic package is not installed"
            )
        return "anthropic"

    has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))
    has_anthropic_key = bool(os.environ.get("ANTHROPIC_API_KEY"))

    # Prefer whichever has BOTH an importable SDK and credentials in env.
    # OpenAI wins ties for back-compat with peekr ≤ 0.3.0 behaviour.
    if openai is not None and has_openai_key:
        return "openai"
    if anthropic is not None and has_anthropic_key:
        return "anthropic"

    # No env credentials detected for either. Try whichever package is
    # importable and hope for ambient auth — this is the legacy code path,
    # kept so existing deployments using non-env auth keep working.
    if openai is not None:
        return "openai"
    if anthropic is not None:
        return "anthropic"

    raise JudgeUnavailable(
        "no LLM judge SDK installed. Install one of: "
        "pip install 'peekr[openai]' or pip install 'peekr[anthropic]'"
    )


def call_judge(
    prompt: str,
    *,
    max_tokens: int = 10,
    model: Optional[str] = None,
    provider: JudgeProvider = "auto",
    fallback: str = "",
) -> str:
    """Run a single LLM-as-judge call and return the raw response text.

    - Raises ``JudgeUnavailable`` when no working judge is configured.
    - Lets provider-side errors (auth failure, rate limit, network) bubble
      up so the EvalExporter can record them as `eval_errors`.
    - ``fallback`` is returned if the provider responds with empty content.
    """
    chosen = select_provider(provider)

    if chosen == "openai":
        response = openai.chat.completions.create(
            model=model or "gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or fallback

    if chosen == "anthropic":
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model or "claude-haiku-4-5-20251001",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text if response.content else fallback

    raise JudgeUnavailable(f"unknown provider returned by select_provider: {chosen!r}")
