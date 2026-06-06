from __future__ import annotations
import json
import sqlite3
import sys
from collections import defaultdict


def main():
    if len(sys.argv) < 2:
        _print_help()
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "login":
        _cmd_login(sys.argv[2:])
    elif cmd == "init":
        _cmd_init(sys.argv[2:])
    elif cmd == "deploy":
        _cmd_deploy(sys.argv[2:])
    elif cmd == "status":
        _cmd_status(sys.argv[2:])
    elif cmd in ("traces", "view"):
        args = sys.argv[2:]
        if cmd == "traces" and "--open" not in args:
            args = args + ["--open"]
        show_io = "--io" in args
        open_browser = "--open" in args
        args = [a for a in args if not a.startswith("--")]
        path = args[0] if args else _default_path()
        if open_browser:
            _cmd_traces_browser(path)
        else:
            view_traces(path, show_io=show_io)
    elif cmd == "replay":
        _cmd_replay(sys.argv[2:])
    elif cmd == "cost":
        args = sys.argv[2:]
        args = [a for a in args if not a.startswith("--")]
        path = args[0] if args else _default_path()
        _cmd_cost(path)
    elif cmd == "compliance":
        _cmd_compliance(sys.argv[2:])
    elif cmd == "dashboard":
        _cmd_dashboard(sys.argv[2:])
    else:
        print(f"Unknown command: {cmd!r}")
        _print_help()
        sys.exit(1)


def _print_help() -> None:
    print("Usage: peekr <command> [options]")
    print()
    print("Cloud commands:")
    print(
        "  login              Open browser to sign in (only time you need the browser)"
    )
    print("  init               Scaffold peekr.yaml from current cloud state")
    print(
        "  deploy [file]      Push peekr.yaml to Peekr Cloud  (default: ./peekr.yaml)"
    )
    print("  status             Show what's deployed for this project")
    print("  compliance list    Show enabled compliance packs")
    print("  compliance enable  <PACK> [--action raise|warn]  Enable a pack")
    print("  compliance disable <PACK>                        Disable a pack")
    print()
    print("Local trace commands:")
    print("  traces [file]      Open local traces in browser dashboard")
    print("  view [file]        Print local traces in terminal")
    print("  cost [file]        Summarise token costs")
    print("  replay <trace_id>  Replay a recorded trace")
    print("  dashboard [file]   Generate static HTML report")


# ── Auth helpers ──────────────────────────────────────────────────────────────


def _config_path() -> str:
    import os

    return os.path.expanduser("~/.config/peekr/config.json")


def _load_config() -> dict:
    import os

    p = _config_path()
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return {}


def _save_config(data: dict) -> None:
    import os

    p = _config_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=2)


def _api_key_from_env_or_config(config: dict | None = None) -> str | None:
    import os

    k = os.environ.get("PEEKR_API_KEY")
    if k:
        return k
    c = config or _load_config()
    return c.get("api_key")


def _endpoint_from_env_or_config(config: dict | None = None) -> str:
    import os

    e = os.environ.get("PEEKR_ENDPOINT")
    if e:
        return e.rstrip("/")
    c = config or _load_config()
    return c.get("endpoint", "https://peekr.starkspherelabs.com").rstrip("/")


# ── peekr login ───────────────────────────────────────────────────────────────


def _cmd_login(args: list[str]) -> None:
    """
    peekr login [--key pk_live_...]

    With --key: save the API key directly (CI-friendly, no browser).
    Without --key: open the dashboard so you can copy your key, then prompt.
    """

    # Allow passing key directly (CI / agent-friendly)
    key = None
    i = 0
    while i < len(args):
        if args[i] in ("--key", "-k") and i + 1 < len(args):
            key = args[i + 1]
            i += 2
        else:
            i += 1

    if not key:
        url = "https://peekr.starkspherelabs.com/dashboard"
        print(f"Opening {url} …")
        try:
            import webbrowser

            webbrowser.open(url)
        except Exception:
            pass
        print()
        print("Copy your API key from Settings → API Keys, then paste it here.")
        print("(or run:  peekr login --key pk_live_...)")
        print()
        key = input("API key: ").strip()

    if not key.startswith("pk_"):
        print("✗ That doesn't look like a Peekr API key (should start with pk_)")
        sys.exit(1)

    # Verify the key works
    endpoint = _endpoint_from_env_or_config()
    try:
        import urllib.request
        import urllib.error

        req = urllib.request.Request(
            f"{endpoint}/api/v1/status",
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        project_id = data.get("project_id", "unknown")
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print("✗ Invalid API key — check your key and try again.")
        else:
            print(f"✗ Server error: HTTP {e.code}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Could not reach {endpoint}: {e}")
        sys.exit(1)

    cfg = _load_config()
    cfg["api_key"] = key
    cfg["endpoint"] = endpoint
    _save_config(cfg)

    print(f"✓ Logged in  (project: {project_id})")
    print(f"  Config saved to {_config_path()}")
    print()
    print("Next steps:")
    print("  peekr init       # scaffold peekr.yaml from current cloud state")
    print("  peekr deploy     # push peekr.yaml to Peekr Cloud")


# ── peekr init ────────────────────────────────────────────────────────────────


def _cmd_init(args: list[str]) -> None:
    """
    peekr init [--out peekr.yaml]

    Pull the current project state from Peekr Cloud and write peekr.yaml.
    Safe to re-run — won't overwrite an existing file without --force.
    """
    import os

    out_path = "peekr.yaml"
    force = False
    i = 0
    while i < len(args):
        if args[i] in ("--out", "-o") and i + 1 < len(args):
            out_path = args[i + 1]
            i += 2
        elif args[i] == "--force":
            force = True
            i += 1
        else:
            i += 1

    if os.path.exists(out_path) and not force:
        print(f"✗ {out_path} already exists. Use --force to overwrite.")
        sys.exit(1)

    cfg = _load_config()
    api_key = _api_key_from_env_or_config(cfg)
    if not api_key:
        print("✗ Not logged in. Run:  peekr login")
        sys.exit(1)

    endpoint = _endpoint_from_env_or_config(cfg)
    status = _fetch_status(api_key, endpoint)

    lines = [
        "# peekr.yaml — declarative config for Peekr Cloud",
        "# Edit and run `peekr deploy` to push changes.",
        "# Docs: https://peekr.starkspherelabs.com/docs",
        "",
        f"project: {status.get('project_id', 'YOUR_PROJECT_SLUG')}",
        "",
    ]

    # Prompts
    lines.append("prompts:")
    prompts = status.get("prompts", [])
    if not prompts:
        lines += [
            "  # example-prompt:",
            "  #   description: What this prompt does",
            "  #   model: gpt-4o-mini",
            "  #   temperature: 0.3",
            "  #   content: |",
            "  #     You are a helpful assistant. {{user_name}}",
        ]
    for p in prompts:
        lines.append(f"  {p['name']}:")
        if p.get("description"):
            lines.append(f"    description: {json.dumps(p['description'])}")
        if p.get("model"):
            lines.append(f"    model: {p['model']}")
        lines.append(f"    # version: {p.get('active_version', 1)} (last deployed)")
        lines.append("    content: |")
        lines.append("      # fetch from dashboard to populate")
    lines.append("")

    # Guardrails
    lines.append("guardrails:")
    guardrails = status.get("guardrails", [])
    if not guardrails:
        lines += [
            "  # - name: PII — Email",
            "  #   type: blocked_pattern",
            "  #   value: '[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+'",
            "  #   action: redact",
            "  #   fields: [input, output]",
        ]
    for g in guardrails:
        lines += [
            f"  - name: {json.dumps(g['name'])}",
            f"    type: {g['rule_type']}",
            f"    value: {json.dumps(g['value'])}",
            f"    action: {g['action']}",
            f"    fields: {json.dumps(g.get('fields', ['output']))}",
            f"    enabled: {str(g.get('enabled', True)).lower()}",
        ]
    lines.append("")

    # Compliance
    lines.append("compliance:")
    packs = [c["name"] for c in status.get("compliance", [])]
    if packs:
        lines.append(f"  packs: {json.dumps(packs)}")
    else:
        lines.append("  packs: []  # e.g. [HIPAA, FDCPA, GDPR]")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"✓ Wrote {out_path}")
    print(
        f"  {len(prompts)} prompt(s), {len(guardrails)} guardrail(s), {len(packs)} compliance pack(s)"
    )
    print()
    print("Edit peekr.yaml then run:  peekr deploy")


# ── peekr deploy ──────────────────────────────────────────────────────────────


def _cmd_deploy(args: list[str]) -> None:
    """
    peekr deploy [peekr.yaml] [--dry-run]

    Push the contents of peekr.yaml to Peekr Cloud.
    Prompts and guardrails are upserted; unchanged items are skipped.
    """
    import os
    import urllib.request
    import urllib.error

    path = "peekr.yaml"
    dry_run = False
    i = 0
    while i < len(args):
        if args[i] == "--dry-run":
            dry_run = True
            i += 1
        elif not args[i].startswith("--"):
            path = args[i]
            i += 1
        else:
            i += 1

    if not os.path.exists(path):
        print(f"✗ {path} not found. Run `peekr init` to create one.")
        sys.exit(1)

    cfg = _load_config()
    api_key = _api_key_from_env_or_config(cfg)
    if not api_key:
        print("✗ Not logged in. Run:  peekr login")
        sys.exit(1)

    endpoint = _endpoint_from_env_or_config(cfg)
    payload = _parse_yaml(path)

    prompt_count = len(payload.get("prompts") or {})
    guard_count = len(payload.get("guardrails") or [])
    pack_count = len((payload.get("compliance") or {}).get("packs") or [])

    print(f"  {path}  →  {endpoint}")
    print(
        f"  {prompt_count} prompt(s)  ·  {guard_count} guardrail(s)  ·  {pack_count} compliance pack(s)"
    )

    if dry_run:
        print()
        print("Dry run — nothing sent.")
        return

    print()

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{endpoint}/api/v1/deploy",
        data=data,
        method="PUT",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:300]
        print(f"✗ Deploy failed: HTTP {e.code}  {body}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Could not reach {endpoint}: {e}")
        sys.exit(1)

    # Prompts
    p = result.get("prompts", {})
    for name in p.get("upserted", []):
        print(f"  ✓ prompt   {name}  (updated)")
    for name in p.get("unchanged", []):
        print(f"  · prompt   {name}  (unchanged)")

    # Guardrails
    g = result.get("guardrails", {})
    for name in g.get("upserted", []):
        print(f"  ✓ guardrail  {name}  (updated)")
    for name in g.get("unchanged", []):
        print(f"  · guardrail  {name}  (unchanged)")

    # Compliance
    c = result.get("compliance", {})
    for name in c.get("enabled", []):
        print(f"  ✓ compliance  {name}  (enabled)")

    total_changed = len(p.get("upserted", [])) + len(g.get("upserted", []))
    print()
    if total_changed:
        print(f"Deploy complete  ({total_changed} change(s))")
    else:
        print("Already up to date.")


# ── peekr status ──────────────────────────────────────────────────────────────


def _cmd_status(args: list[str]) -> None:
    """peekr status — show what's deployed for this project."""
    cfg = _load_config()
    api_key = _api_key_from_env_or_config(cfg)
    if not api_key:
        print("✗ Not logged in. Run:  peekr login")
        sys.exit(1)

    endpoint = _endpoint_from_env_or_config(cfg)
    status = _fetch_status(api_key, endpoint)

    print(f"Project: {status.get('project_id', '?')}")
    print(f"Endpoint: {endpoint}")
    print()

    # Prompts
    prompts = status.get("prompts", [])
    print(f"Prompts  ({len(prompts)})")
    if prompts:
        for p in prompts:
            v = f"v{p['active_version']}" if p.get("active_version") else "no versions"
            m = f"  [{p['model']}]" if p.get("model") else ""
            updated = p.get("updated_at", "")[:10]
            print(f"  {p['name']:<30} {v}{m}  updated {updated}")
    else:
        print("  (none — run `peekr deploy` to push prompts)")
    print()

    # Guardrails
    guardrails = status.get("guardrails", [])
    print(f"Guardrails  ({len(guardrails)})")
    if guardrails:
        for g in guardrails:
            enabled = "on " if g.get("enabled") else "off"
            print(f"  [{enabled}]  {g['name']:<38} {g['rule_type']}  →  {g['action']}")
    else:
        print("  (none)")
    print()

    # Compliance
    packs = status.get("compliance", [])
    print(f"Compliance packs  ({len(packs)})")
    if packs:
        for c in packs:
            print(
                f"  {c['display_name']:<30} severity={c.get('severity', '?')}  action={c['action']}"
            )
    else:
        print("  (none enabled)")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _cmd_compliance(args: list[str]) -> None:
    """
    peekr compliance list
    peekr compliance enable  <PACK> [--action raise|warn]
    peekr compliance disable <PACK>
    """
    import urllib.request
    import urllib.error

    if not args or args[0] in ("list", "ls"):
        cfg = _load_config()
        api_key = _api_key_from_env_or_config(cfg)
        if not api_key:
            print("✗ Not logged in. Run:  peekr login")
            sys.exit(1)
        endpoint = _endpoint_from_env_or_config(cfg)
        status = _fetch_status(api_key, endpoint)
        packs = status.get("compliance", [])
        print(f"Compliance packs ({len(packs)} enabled)\n")
        if not packs:
            print("  None enabled.")
            print()
            print("Enable one:  peekr compliance enable HIPAA")
        else:
            for p in packs:
                print(f"  ✓  {p['display_name']:<32} action={p['action']}")
        return

    subcmd = args[0]  # enable | disable
    if subcmd not in ("enable", "disable"):
        print(f"Unknown compliance subcommand: {subcmd!r}")
        print("  peekr compliance list")
        print("  peekr compliance enable  <PACK> [--action raise|warn]")
        print("  peekr compliance disable <PACK>")
        sys.exit(1)

    rest = args[1:]
    if not rest or rest[0].startswith("--"):
        print(f"Usage: peekr compliance {subcmd} <PACK_NAME>")
        print("Example packs: HIPAA, FDCPA, FINRA, GDPR, UAE_PDPL, UAE_DHA, KSA_PDPL")
        sys.exit(1)

    pack_name = rest[0].upper()
    action = "raise"
    for i, a in enumerate(rest[1:]):
        if a == "--action" and i + 1 < len(rest) - 1:
            action = rest[i + 2]

    cfg = _load_config()
    api_key = _api_key_from_env_or_config(cfg)
    if not api_key:
        print("✗ Not logged in. Run:  peekr login")
        sys.exit(1)
    endpoint = _endpoint_from_env_or_config(cfg)

    if subcmd == "enable":
        payload = {"compliance": {"packs": [pack_name]}}
        # Use the deploy endpoint — it upserts packs
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{endpoint}/api/v1/deploy",
            data=data,
            method="PUT",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                result = json.loads(resp.read())
            enabled = result.get("compliance", {}).get("enabled", [])
            if pack_name in enabled or pack_name.upper() in [
                e.upper() for e in enabled
            ]:
                print(f"✓ {pack_name} enabled  (action={action})")
                print("  Your SDK will pick this up automatically on next startup.")
                print("  No code change needed — compliance is auto-discovered.")
            else:
                print(f"⚠ Pack '{pack_name}' not found. Check the pack name.")
                print(
                    "  Available: HIPAA, FDCPA, FINRA, GDPR, EU_AI_ACT, UAE_PDPL, UAE_DHA,"
                )
                print(
                    "             UAE_CBUAE, UAE_RERA, KSA_PDPL, UAE_DIFC, TCPA, UPL, EEOC_ADA"
                )
        except urllib.error.HTTPError as e:
            print(f"✗ Error: HTTP {e.code}  {e.read().decode()[:200]}")
            sys.exit(1)

    else:  # disable
        # DELETE from project_compliance via deploy with empty packs (workaround: use compliance API)
        req = urllib.request.Request(
            f"{endpoint}/api/v1/compliance",
            data=json.dumps({"pack": pack_name, "enabled": False}).encode(),
            method="PUT",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                json.loads(resp.read())
            print(f"✓ {pack_name} disabled")
            print("  Changes take effect on next SDK startup.")
        except urllib.error.HTTPError as e:
            print(f"✗ Error: HTTP {e.code}  {e.read().decode()[:200]}")
            sys.exit(1)


def _cmd_traces_browser(path: str) -> None:
    """Generate dashboard HTML and open it in the default browser."""
    import os
    import tempfile
    import webbrowser
    from .dashboard import generate_dashboard

    out = os.path.join(tempfile.gettempdir(), "peekr_traces.html")
    generate_dashboard(path, output=out)
    webbrowser.open(f"file://{out}")
    print(f"Traces opened  ({path})")
    print(f"File: {out}")


def _fetch_status(api_key: str, endpoint: str) -> dict:
    import urllib.request
    import urllib.error

    req = urllib.request.Request(
        f"{endpoint}/api/v1/status",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print("✗ Invalid API key. Run:  peekr login")
        else:
            print(f"✗ Server error: HTTP {e.code}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Could not reach {endpoint}: {e}")
        sys.exit(1)


def _parse_yaml(path: str) -> dict:
    """
    Minimal YAML parser for peekr.yaml — handles the subset we need
    without requiring PyYAML as a dependency.
    Falls back to PyYAML if installed.
    """
    try:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return _normalise_yaml(data or {})
    except ImportError:
        pass

    # Fallback: very small hand-rolled parser for peekr.yaml structure
    with open(path) as f:
        raw = f.read()

    try:
        import yaml as _y  # might be pyyaml under a different name

        data = _y.safe_load(raw)
        return _normalise_yaml(data or {})
    except Exception:
        pass

    print("✗ PyYAML is not installed. Install it with:  pip install pyyaml")
    print("  Then re-run peekr deploy.")
    sys.exit(1)


def _normalise_yaml(data: dict) -> dict:
    """Convert peekr.yaml structure to the API payload shape."""
    out: dict = {}

    raw_prompts = data.get("prompts") or {}
    if raw_prompts:
        out["prompts"] = {}
        for name, spec in raw_prompts.items():
            if not spec or not spec.get("content"):
                continue
            out["prompts"][name] = {
                "content": spec["content"],
                "description": spec.get("description"),
                "model": spec.get("model"),
                "temperature": spec.get("temperature"),
                "notes": spec.get("notes"),
            }

    raw_guardrails = data.get("guardrails") or []
    if raw_guardrails:
        out["guardrails"] = []
        for rule in raw_guardrails:
            if not rule.get("name") or not rule.get("value"):
                continue
            out["guardrails"].append(
                {
                    "name": rule["name"],
                    "type": rule.get("type", "blocked_pattern"),
                    "value": str(rule["value"]),
                    "action": rule.get("action", "warn"),
                    "fields": rule.get("fields", ["output"]),
                    "enabled": rule.get("enabled", True),
                }
            )

    compliance = data.get("compliance") or {}
    packs = compliance.get("packs") or []
    if packs:
        out["compliance"] = {"packs": packs}

    return out


def _cmd_dashboard(args: list[str]) -> None:
    """peekr dashboard [path] [-o report.html] — emit a static HTML report."""
    output = "dashboard.html"
    rest: list[str] = []
    i = 0
    while i < len(args):
        if args[i] in ("-o", "--out") and i + 1 < len(args):
            output = args[i + 1]
            i += 2
        else:
            rest.append(args[i])
            i += 1
    path = rest[0] if rest else _default_path()

    from .dashboard import generate_dashboard  # noqa: PLC0415

    out = generate_dashboard(path, output=output)
    print(f"Dashboard written to {out}")
    print(f"Open it with:  open {out}")


def _cmd_replay(args: list[str]) -> None:
    """Handle: peekr replay <trace_id> [--db traces.db] [--jsonl traces.jsonl]"""
    if not args or args[0].startswith("--"):
        print("Usage: peekr replay <trace_id> [--db traces.db] [--jsonl traces.jsonl]")
        sys.exit(1)

    trace_id = args[0]
    rest = args[1:]

    db_path = None
    jsonl_path = None
    i = 0
    while i < len(rest):
        if rest[i] == "--db" and i + 1 < len(rest):
            db_path = rest[i + 1]
            i += 2
        elif rest[i] == "--jsonl" and i + 1 < len(rest):
            jsonl_path = rest[i + 1]
            i += 2
        else:
            i += 1

    from .replay import replay_trace  # noqa: PLC0415

    try:
        new_trace_id = replay_trace(
            trace_id=trace_id,
            db_path=db_path,
            jsonl_path=jsonl_path,
        )
    except Exception as exc:
        print(f"Replay failed: {exc}")
        sys.exit(1)

    print(f"Replayed trace {trace_id[:8]} → new trace {new_trace_id[:8]}")
    print()

    # Show the new trace from the same storage
    storage_path = db_path or jsonl_path or _default_path()
    view_traces(storage_path)


def _cmd_cost(path: str) -> None:
    """peekr cost <traces.jsonl|traces.db> — cost breakdown + top-10 hotspots."""
    if path.endswith(".db"):
        spans = _read_sqlite(path)
    else:
        spans = _read_jsonl(path)

    if not spans:
        return

    # ── per-span cost records (LLM calls only) ────────────────────────────────
    records = []
    for s in spans:
        attrs = s.get("attributes") or {}
        inp = attrs.get("tokens_input", 0)
        out = attrs.get("tokens_output", 0)
        dur = s.get("duration_ms") or 0.0
        cost = (inp / 1_000_000 * 0.80) + (out / 1_000_000 * 4.00)
        records.append(
            {
                "name": s["name"],
                "model": attrs.get("model", ""),
                "status": s.get("status", "ok"),
                "tokens_input": inp,
                "tokens_output": out,
                "tokens_total": inp + out,
                "cost": cost,
                "duration_ms": dur,
            }
        )

    llm_records = [r for r in records if r["tokens_total"] > 0]
    total_cost = sum(r["cost"] for r in llm_records)
    total_input = sum(r["tokens_input"] for r in llm_records)
    total_output = sum(r["tokens_output"] for r in llm_records)
    total_dur = sum(r["duration_ms"] for r in llm_records)
    errors = sum(1 for r in records if r["status"] == "error")

    # ── summary ───────────────────────────────────────────────────────────────
    W = 60
    print()
    print("─" * W)
    print(f"  peekr cost  ·  {path}")
    print("─" * W)
    print(f"  Total spans        : {len(spans):,}")
    print(f"  LLM calls          : {len(llm_records):,}")
    print(f"  Errors             : {errors}")
    print(f"  Total input tokens : {total_input:,}")
    print(f"  Total output tokens: {total_output:,}")
    print(f"  Total LLM time     : {total_dur / 1000:.1f}s")
    print(f"  Total cost (est.)  : ${total_cost:.5f}  (Haiku rates: $0.80/$4.00 per M)")
    print("─" * W)

    # ── breakdown by operation ─────────────────────────────────────────────────
    by_op: dict[str, dict] = defaultdict(
        lambda: {
            "calls": 0,
            "input": 0,
            "output": 0,
            "cost": 0.0,
            "duration_ms": 0.0,
            "errors": 0,
        }
    )
    for r in records:
        key = f"{r['name']}" + (f"  [{r['model']}]" if r["model"] else "")
        by_op[key]["calls"] += 1
        by_op[key]["input"] += r["tokens_input"]
        by_op[key]["output"] += r["tokens_output"]
        by_op[key]["cost"] += r["cost"]
        by_op[key]["duration_ms"] += r["duration_ms"]
        by_op[key]["errors"] += 1 if r["status"] == "error" else 0

    print()
    print("  Cost by operation:")
    print(
        f"  {'Operation':<48} {'Calls':>5}  {'Cost':>9}  {'Avg/call':>9}  {'Avg ms':>7}"
    )
    print("  " + "─" * (W - 2))
    for op, s in sorted(by_op.items(), key=lambda x: -x[1]["cost"]):
        avg_cost = s["cost"] / max(s["calls"], 1)
        avg_ms = s["duration_ms"] / max(s["calls"], 1)
        err_flag = "  \033[31m(!)\033[0m" if s["errors"] else ""
        print(
            f"  {op:<48} {s['calls']:>5}  ${s['cost']:>8.5f}  ${avg_cost:>8.5f}  {avg_ms:>6.0f}ms{err_flag}"
        )

    # ── top 10 hotspots ───────────────────────────────────────────────────────
    def _hotspot_score(r: dict) -> float:
        # normalise cost and latency, weight cost 60% / latency 40%
        max_cost = max((x["cost"] for x in llm_records), default=1) or 1
        max_dur = max((x["duration_ms"] for x in llm_records), default=1) or 1
        return 0.6 * (r["cost"] / max_cost) + 0.4 * (r["duration_ms"] / max_dur)

    if llm_records:
        ranked = sorted(llm_records, key=_hotspot_score, reverse=True)[:10]
        print()
        print("  Top 10 hottest calls  (60% cost · 40% latency):")
        print(
            f"  {'#':<3} {'Operation':<40} {'In':>7} {'Out':>6} {'Cost':>9} {'ms':>7}  {'Model'}"
        )
        print("  " + "─" * (W - 2))
        for i, r in enumerate(ranked, 1):
            err = " \033[31m!\033[0m" if r["status"] == "error" else ""
            print(
                f"  {i:<3} {r['name']:<40} "
                f"{r['tokens_input']:>7,} {r['tokens_output']:>6,} "
                f"${r['cost']:>8.5f} {r['duration_ms']:>6.0f}ms  "
                f"{r['model']}{err}"
            )

        # top 5 slowest (if different from hottest)
        slowest = sorted(llm_records, key=lambda x: -x["duration_ms"])[:5]
        if slowest[0] != ranked[0]:
            print()
            print("  Top 5 slowest calls:")
            print(f"  {'#':<3} {'Operation':<40} {'ms':>7}  {'Tokens':>7}  {'Cost':>9}")
            print("  " + "─" * (W - 2))
            for i, r in enumerate(slowest, 1):
                print(
                    f"  {i:<3} {r['name']:<40} "
                    f"{r['duration_ms']:>6.0f}ms  "
                    f"{r['tokens_total']:>7,}  "
                    f"${r['cost']:>8.5f}"
                )

    print()


def _default_path() -> str:
    import os

    if os.path.exists("traces.db"):
        return "traces.db"
    return "traces.jsonl"


def view_traces(path: str, show_io: bool = False):
    if path.endswith(".db"):
        spans = _read_sqlite(path)
    else:
        spans = _read_jsonl(path)

    if not spans:
        return

    traces = defaultdict(list)
    for span in spans:
        traces[span["trace_id"]].append(span)

    for i, (trace_id, trace_spans) in enumerate(traces.items()):
        if i > 0:
            print()
        total_ms = sum(
            s.get("duration_ms") or 0 for s in trace_spans if s["parent_id"] is None
        )
        total_tokens = sum(
            (s.get("attributes") or {}).get("tokens_total", 0) for s in trace_spans
        )
        token_str = f"  {total_tokens} tokens" if total_tokens else ""
        print(f"Trace {trace_id[:8]}  {total_ms:.0f}ms{token_str}")
        print("─" * 48)
        roots = [s for s in trace_spans if s["parent_id"] is None]
        for root in roots:
            _print_span(root, trace_spans, indent=0, show_io=show_io)


def _read_jsonl(path: str) -> list[dict]:
    try:
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"No traces file at {path}")
        return []


def _read_sqlite(path: str) -> list[dict]:
    try:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM spans ORDER BY start_time").fetchall()
        conn.close()
        spans = []
        for r in rows:
            s = dict(r)
            s["attributes"] = json.loads(s["attributes"] or "{}")
            spans.append(s)
        return spans
    except (sqlite3.OperationalError, FileNotFoundError):
        print(f"No traces database at {path}")
        return []


def _print_span(span, all_spans, indent, show_io):
    duration = f"{span['duration_ms']:.0f}ms" if span.get("duration_ms") else "  ?"
    attrs = span.get("attributes") or {}
    model = f" [{attrs['model']}]" if "model" in attrs else ""
    tokens = f" {attrs['tokens_total']}tok" if "tokens_total" in attrs else ""
    error = " \033[31mERROR\033[0m" if span["status"] == "error" else ""

    connector = "└─ " if indent > 0 else ""
    prefix = "   " * indent + connector
    print(f"{prefix}\033[1m{span['name']}\033[0m{model}  {duration}{tokens}{error}")

    if show_io:
        io_prefix = "   " * (indent + 1)
        if "input" in attrs:
            print(f"{io_prefix}\033[2min:  {attrs['input'][:120]}\033[0m")
        if "output" in attrs:
            print(f"{io_prefix}\033[2mout: {attrs['output'][:120]}\033[0m")
        if "error" in attrs and span["status"] == "error":
            print(f"{io_prefix}\033[31merr: {attrs['error']}\033[0m")

    children = [s for s in all_spans if s.get("parent_id") == span["span_id"]]
    for child in children:
        _print_span(child, all_spans, indent + 1, show_io)
