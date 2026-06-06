"""
CloudComplianceGuard — fetches proprietary compliance rules from Peekr Cloud
at instrument() time and enforces them locally on every LLM span.

The rules themselves are proprietary and stored in Peekr Cloud.
This module is only a thin authenticated client + local enforcer.

Usage (via instrument())::

    peekr.instrument(
        exporter=peekr.HTTPExporter(
            endpoint="https://peekr.starkspherelabs.com",
            api_key="pk_live_…",
        ),
        compliance=["FDCPA", "HIPAA"],   # fetched from Cloud
    )

Or standalone::

    guard = peekr.guard.CloudComplianceGuard(
        api_key="pk_live_…",
        endpoint="https://peekr.starkspherelabs.com",
        packs=["FDCPA"],
    )
    peekr.instrument(guardrails=[guard])
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
import json
from typing import TYPE_CHECKING

from ..context import GuardrailError

if TYPE_CHECKING:
    from ..span import Span

_LLM_PREFIXES = ("openai.", "anthropic.", "bedrock.", "gemini.")


class CloudComplianceGuard:
    """Fetches compliance rules from Peekr Cloud and enforces them locally.

    Parameters
    ----------
    api_key
        Your ``pk_live_`` key — authenticates the rule fetch.
    endpoint
        Peekr Cloud URL. Defaults to ``https://peekr.starkspherelabs.com``.
    packs
        List of regulation pack names to enable, e.g. ``["FDCPA", "HIPAA"]``.
        ``None`` = all packs enabled for your project.
    action
        ``"raise"`` (default) — raises ``GuardrailError`` on violation.
        ``"warn"``  — records violation on span but never blocks the response.
    timeout
        Seconds to wait for the rule-fetch request. Default 10.
    """

    _blocks: bool = True  # post-storage, may raise
    _input_guard: bool = True  # also run pre-call for prohibited_input rules

    def __init__(
        self,
        api_key: str,
        endpoint: str = "https://peekr.starkspherelabs.com",
        packs: list[str] | None = None,
        action: str = "raise",
        timeout: int = 10,
    ) -> None:
        if action not in ("raise", "warn"):
            raise ValueError(f"action must be 'raise' or 'warn'; got {action!r}")

        self.action = action
        self._blocks = action == "raise"
        self._packs = packs
        self._api_key = api_key
        self._endpoint = endpoint.rstrip("/")
        self._timeout = timeout
        self._rules_loaded = False

        # Populated on first use (lazy) so server startup is never blocked
        self._output_rules: list[tuple[re.Pattern, str, str]] = []
        self._input_rules: list[tuple[re.Pattern, str, str]] = []
        self._disclosures: list[tuple[str, str, str]] = []
        self._pack_actions: dict[str, str] = {}

    @property
    def name(self) -> str:
        packs = ", ".join(self._pack_actions.keys()) or "none"
        return f"CloudComplianceGuard({packs})"

    def _fetch_rules(self, api_key: str, endpoint: str, timeout: int) -> None:
        url = f"{endpoint}/v1/compliance"
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")[:200]
            raise RuntimeError(
                f"CloudComplianceGuard: failed to fetch rules from {url} "
                f"(HTTP {e.code}): {body}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"CloudComplianceGuard: could not reach {url}: {e}"
            ) from e

        flags = re.IGNORECASE | re.DOTALL
        packs_data = data.get("packs", [])
        rules_data = data.get("rules", [])

        # Build pack→action map; filter by requested packs.
        # User-specified action is the ceiling — "warn" always wins over pack's "raise".
        for p in packs_data:
            name = p.get("name", "")
            if self._packs is None or name in self._packs:
                pack_action = p.get("action", self.action)
                # If the user asked for "warn", never escalate to "raise"
                self._pack_actions[name] = (
                    self.action if self.action == "warn" else pack_action
                )

        enabled_packs = set(self._pack_actions.keys())

        for rule in rules_data:
            pack = rule.get("pack", "")
            if pack not in enabled_packs:
                continue
            rtype = rule.get("rule_type", "")
            pattern = rule.get("pattern", "")
            desc = rule.get("description", pattern[:80])

            if rtype == "prohibited_output":
                try:
                    self._output_rules.append((re.compile(pattern, flags), desc, pack))
                except re.error:
                    pass  # malformed pattern — skip silently

            elif rtype == "prohibited_input":
                try:
                    self._input_rules.append((re.compile(pattern, flags), desc, pack))
                except re.error:
                    pass

            elif rtype == "required_disclosure":
                self._disclosures.append((pattern, desc, pack))

        total = (
            len(self._output_rules) + len(self._input_rules) + len(self._disclosures)
        )
        print(
            f"[peekr] CloudComplianceGuard: loaded {total} rules "
            f"across {len(enabled_packs)} pack(s): {', '.join(sorted(enabled_packs))}"
        )

    def _ensure_rules(self) -> None:
        if not self._rules_loaded:
            self._fetch_rules(self._api_key, self._endpoint, self._timeout)
            self._rules_loaded = True

    # ── Enforcement ───────────────────────────────────────────────────────────

    def _pack_action(self, pack: str) -> str:
        return self._pack_actions.get(pack, self.action)

    def run(self, span: "Span") -> None:
        """Post-call: checks output patterns and required disclosures."""
        self._ensure_rules()
        if not any(span.name.startswith(p) for p in _LLM_PREFIXES):
            return

        output = span.attributes.get("output", "") or ""
        violations = []

        for pattern, desc, pack in self._output_rules:
            match = pattern.search(output)
            if match:
                violations.append(
                    {
                        "type": "prohibited_output",
                        "pack": pack,
                        "matched": match.group(0)[:100],
                        "rule": desc,
                    }
                )

        for text, desc, pack in self._disclosures:
            if not re.search(re.escape(text[:60]), output, re.IGNORECASE):
                violations.append(
                    {
                        "type": "missing_disclosure",
                        "pack": pack,
                        "matched": f"MISSING: '{text[:80]}'",
                        "rule": desc,
                    }
                )

        if not violations:
            return

        span.attributes.setdefault("compliance_violations", [])
        span.attributes["compliance_violations"].extend(violations)

        # Respect per-pack action — raise if ANY pack is set to raise
        should_raise = any(self._pack_action(v["pack"]) == "raise" for v in violations)
        msg = "Compliance violation(s): " + " | ".join(
            f"[{v['pack']}] {v['rule'][:60]}" for v in violations[:3]
        )
        if should_raise:
            raise GuardrailError(msg, guardrail_name=self.name, span=span)

    def run_input(self, span: "Span") -> None:
        """Pre-call: checks input patterns (prohibited questions, etc.)."""
        self._ensure_rules()
        if not self._input_rules:
            return
        prompt = span.attributes.get("input", "") or ""
        for pattern, desc, pack in self._input_rules:
            match = pattern.search(prompt)
            if match:
                v = {
                    "type": "prohibited_input",
                    "pack": pack,
                    "matched": match.group(0)[:100],
                    "rule": desc,
                }
                span.attributes.setdefault("compliance_violations", [])
                span.attributes["compliance_violations"].append(v)
                if self._pack_action(pack) == "raise":
                    raise GuardrailError(
                        f"[{pack}] Prohibited input detected: {desc}",
                        guardrail_name=self.name,
                        span=span,
                    )
