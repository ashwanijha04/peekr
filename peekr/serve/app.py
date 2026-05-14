"""Flask app for ``peekr serve``.

Single-tenant, no auth, runs on ``localhost``. Reads existing traces.db /
traces.jsonl. No writes, no telemetry. Vanilla HTML + CSS + a small amount of
JS — no build step.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from .data import (
    Filters,
    TraceStore,
    build_tree,
    open_store,
    trace_overview,
)


PAGE_SIZE = 50


def _require_flask():
    try:
        import flask  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "peekr serve requires Flask. Install it with:\n"
            "    pip install 'peekr[serve]'\n"
            "or:\n"
            "    pip install flask"
        ) from e


def create_app(db: Optional[str] = None, jsonl: Optional[str] = None):
    """Build a Flask app bound to the chosen storage backend."""
    _require_flask()
    from flask import Flask, abort, jsonify, render_template, request

    store: TraceStore = open_store(db=db, jsonl=jsonl)

    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config["TRACE_STORE"] = store

    # ── filters / helpers exposed to templates ───────────────────────────────
    @app.template_filter("fmt_ts")
    def fmt_ts(value):
        if not value:
            return ""
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        except (TypeError, ValueError):
            return ""

    @app.template_filter("fmt_ms")
    def fmt_ms(value):
        try:
            return f"{float(value):,.0f}ms"
        except (TypeError, ValueError):
            return "—"

    @app.template_filter("fmt_cost")
    def fmt_cost(value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "$0.00000"
        return f"${v:.5f}"

    @app.template_filter("short_id")
    def short_id(value, n: int = 8):
        if not value:
            return ""
        return str(value)[:n]

    @app.context_processor
    def inject_globals():
        return {"source": store.source}

    # ── routes ──────────────────────────────────────────────────────────────
    @app.route("/")
    def index():
        page = max(1, int(request.args.get("page", "1") or 1))
        filters = _filters_from_request(request.args)
        traces, total = store.list_traces(
            filters,
            limit=PAGE_SIZE,
            offset=(page - 1) * PAGE_SIZE,
        )
        return render_template(
            "trace_list.html",
            traces=traces,
            total=total,
            page=page,
            page_size=PAGE_SIZE,
            page_count=max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE),
            filters=filters,
            facets=store.facets(),
            request_args=request.args,
        )

    @app.route("/trace/<trace_id>")
    def trace_detail(trace_id: str):
        spans = store.get_trace(trace_id)
        if not spans:
            abort(404)
        return render_template(
            "trace_detail.html",
            trace_id=trace_id,
            spans=spans,
            tree=build_tree(spans),
            overview=trace_overview(spans),
        )

    @app.route("/api/trace/<trace_id>/span/<span_id>")
    def api_span(trace_id: str, span_id: str):
        # Returns the I/O attributes for a single span. The detail page calls
        # this lazily so a 50 KB prompt only crosses the wire when the user
        # actually expands the span.
        spans = store.get_trace(trace_id)
        for s in spans:
            if s["span_id"] == span_id:
                attrs = s.get("attributes") or {}
                return jsonify(
                    {
                        "span_id": s["span_id"],
                        "name": s.get("name"),
                        "status": s.get("status"),
                        "input": attrs.get("input"),
                        "output": attrs.get("output"),
                        "error": attrs.get("error"),
                        "system": attrs.get("system"),
                        "model": attrs.get("model"),
                        "tokens_input": attrs.get("tokens_input"),
                        "tokens_output": attrs.get("tokens_output"),
                        "tokens_total": attrs.get("tokens_total"),
                        "eval_scores": attrs.get("eval_scores"),
                        "guardrail_findings": attrs.get("guardrail_findings"),
                        "experiment_variant": attrs.get("experiment_variant"),
                        "user_id": attrs.get("user_id"),
                        "session_id": attrs.get("session_id"),
                    }
                )
        abort(404)

    @app.route("/compare")
    def compare():
        a = request.args.get("a", "").strip()
        b = request.args.get("b", "").strip()
        trace_a = store.get_trace(a) if a else []
        trace_b = store.get_trace(b) if b else []
        return render_template(
            "compare.html",
            a_id=a,
            b_id=b,
            trace_a=trace_a,
            trace_b=trace_b,
            tree_a=build_tree(trace_a) if trace_a else [],
            tree_b=build_tree(trace_b) if trace_b else [],
            overview_a=trace_overview(trace_a) if trace_a else None,
            overview_b=trace_overview(trace_b) if trace_b else None,
        )

    @app.errorhandler(404)
    def not_found(_e):  # noqa: ARG001
        return render_template("404.html"), 404

    return app


def _filters_from_request(args) -> Filters:
    def _f(name):
        v = args.get(name, "").strip()
        return v or None

    min_cost = _f("min_cost")
    since = _f("since")
    until = _f("until")
    return Filters(
        user_id=_f("user"),
        session_id=_f("session"),
        since=_parse_time(since),
        until=_parse_time(until),
        min_cost=float(min_cost) if min_cost else None,
        has_evals=args.get("has_evals") in ("1", "on", "true"),
        has_guardrails=args.get("has_guardrails") in ("1", "on", "true"),
        has_errors=args.get("has_errors") in ("1", "on", "true"),
        q=_f("q"),
    )


def _parse_time(value: Optional[str]) -> Optional[float]:
    """Accept a unix-epoch float or an ISO-8601 string."""
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def run(
    host: str = "127.0.0.1",
    port: int = 8000,
    db: Optional[str] = None,
    jsonl: Optional[str] = None,
    debug: bool = False,
) -> None:
    """Boot Flask's dev server on localhost. Blocks until Ctrl-C."""
    app = create_app(db=db, jsonl=jsonl)
    store: TraceStore = app.config["TRACE_STORE"]
    print(f"peekr serve · reading {store.source}")
    print(f"             · http://{host}:{port}")
    # Force localhost-only bind even if the caller passed 0.0.0.0 to keep the
    # "local-first, no auth" promise hard to break by accident.
    if not _is_local(host) and not os.environ.get("PEEKR_ALLOW_REMOTE"):
        print(
            f"             · refusing to bind to {host!r}; "
            "set PEEKR_ALLOW_REMOTE=1 to override."
        )
        host = "127.0.0.1"
    app.run(host=host, port=port, debug=debug, use_reloader=False)


def _is_local(host: str) -> bool:
    return host in ("127.0.0.1", "localhost", "::1")
