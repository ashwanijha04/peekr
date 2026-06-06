"""peekr.prompts — fetch and render version-controlled prompts from Peekr Cloud.

Usage::

    import peekr

    # Fetch active version at agent startup
    prompt = peekr.prompts.get(
        name="rag_answer",
        api_key="pk_live_…",                      # or set PEEKR_API_KEY env var
        endpoint="https://peekr.starkspherelabs.com",  # or PEEKR_ENDPOINT env var
    )

    # Render variables and call LLM
    system = prompt.render(context=docs, language="en")
    client.chat.completions.create(
        model=prompt.model or "gpt-4o-mini",
        messages=[{"role": "system", "content": system}],
    )

    # Access raw fields
    print(prompt.version)     # int — e.g. 3
    print(prompt.variables)   # list of variable names — e.g. ["context", "language"]
    print(prompt.content)     # raw template string
    print(prompt.model)       # pinned model or None
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
import json
from typing import Any

_DEFAULT_ENDPOINT = "https://peekr.starkspherelabs.com"


class Prompt:
    """A fetched and rendered prompt version."""

    def __init__(
        self,
        name: str,
        version: int,
        content: str,
        variables: list[str],
        model: str | None,
        notes: str | None,
    ) -> None:
        self.name = name
        self.version = version
        self.content = content
        self.variables = variables
        self.model = model
        self.notes = notes

    def render(self, **kwargs: Any) -> str:
        """Replace ``{{variable}}`` placeholders with provided values.

        Raises ``KeyError`` if a required variable is missing.
        Extra kwargs (not present in the template) are silently ignored.

        Examples
        --------
        ::

            system = prompt.render(context=docs, language="en", tone="professional")
        """
        result = self.content
        for var in self.variables:
            if var not in kwargs:
                raise KeyError(
                    f"Missing variable '{{{{ {var} }}}}' required by prompt '{self.name}' v{self.version}. "
                    f"Pass it as a keyword argument: prompt.render({var}=...)"
                )
            result = result.replace(f"{{{{{var}}}}}", str(kwargs[var]))
        return result

    def __repr__(self) -> str:
        return f"<Prompt name={self.name!r} version={self.version} variables={self.variables}>"


def get(
    name: str,
    *,
    api_key: str | None = None,
    endpoint: str | None = None,
    timeout: float = 10.0,
) -> Prompt:
    """Fetch the active version of a named prompt from Peekr Cloud.

    Parameters
    ----------
    name
        The prompt slug as defined in the Peekr Cloud dashboard.
    api_key
        Your ``pk_live_`` key. Falls back to the ``PEEKR_API_KEY`` environment
        variable if not provided.
    endpoint
        Peekr Cloud URL. Falls back to ``PEEKR_ENDPOINT`` env var, then
        ``https://peekr.starkspherelabs.com``.
    timeout
        Request timeout in seconds. Default 10.

    Returns
    -------
    Prompt
        The active prompt version. Use ``.render(**kwargs)`` to fill variables.

    Raises
    ------
    ValueError
        If the API key is missing or the prompt is not found.
    RuntimeError
        On network or server error.
    """
    key = api_key or os.environ.get("PEEKR_API_KEY")
    if not key:
        raise ValueError(
            "Peekr API key required. Pass api_key= or set PEEKR_API_KEY env var."
        )

    base = (endpoint or os.environ.get("PEEKR_ENDPOINT") or _DEFAULT_ENDPOINT).rstrip(
        "/"
    )
    url = f"{base}/api/v1/prompts?name={urllib.request.quote(name)}"

    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {key}",
            "User-Agent": "peekr-python",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        try:
            detail = json.loads(body)
            if detail.get("error") == "prompt_not_found":
                raise ValueError(
                    f"Prompt '{name}' not found. Create it in the Peekr Cloud dashboard."
                ) from e
        except (json.JSONDecodeError, AttributeError):
            pass
        raise RuntimeError(f"Peekr API error {e.code}: {body[:200]}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise RuntimeError(f"Peekr request failed: {e}") from e

    return Prompt(
        name=data.get("name", name),
        version=data.get("version", 0),
        content=data.get("content", ""),
        variables=data.get("variables", []),
        model=data.get("model"),
        notes=data.get("notes"),
    )


def list_all(
    *,
    api_key: str | None = None,
    endpoint: str | None = None,
    timeout: float = 10.0,
) -> list[dict]:
    """List all prompts and their active versions for a project.

    Returns a list of dicts: ``[{name, description, active_version, variables, model}]``
    """
    key = api_key or os.environ.get("PEEKR_API_KEY")
    if not key:
        raise ValueError("Peekr API key required.")

    base = (endpoint or os.environ.get("PEEKR_ENDPOINT") or _DEFAULT_ENDPOINT).rstrip(
        "/"
    )
    req = urllib.request.Request(
        f"{base}/api/v1/prompts",
        headers={"Authorization": f"Bearer {key}", "User-Agent": "peekr-python"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        return data.get("prompts", [])
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        raise RuntimeError(f"Peekr request failed: {e}") from e
