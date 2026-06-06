"""OpenTelemetry / OpenInference exporter.

Ships peekr spans to any OTel-compatible backend (Datadog, Honeycomb, Jaeger,
Grafana Tempo, Arize Phoenix, Langfuse-OTel, etc.) by translating attributes
into the OpenInference semantic conventions used by the LLM observability
ecosystem.

Usage::

    from peekr.otel import OTelExporter
    from peekr.exporters import add_exporter

    add_exporter(OTelExporter())          # uses the OTel SDK already configured
                                          # in your app (auto-detects exporter
                                          # from OTEL_EXPORTER_OTLP_ENDPOINT etc.)

Or, configure an explicit OTLP target::

    OTelExporter(endpoint="https://api.honeycomb.io", headers={...})

Requires ``opentelemetry-api`` and ``opentelemetry-sdk``. Install with::

    pip install "peekr[otel]"
"""

from __future__ import annotations

import json
from typing import Optional


_OI_PREFIX = "openinference"


def _llm_attributes(span_dict: dict) -> dict:
    """Translate peekr span attributes → OpenInference semantic-conv attributes."""
    attrs = span_dict.get("attributes") or {}
    name = span_dict.get("name", "")
    is_llm = name.startswith(
        ("openai.", "anthropic.", "bedrock.", "gemini.", "google.")
    )
    out: dict[str, object] = {
        f"{_OI_PREFIX}.span.kind": "LLM" if is_llm else "CHAIN",
        "peekr.name": name,
    }
    if (model := attrs.get("model")) is not None:
        out["llm.model_name"] = model
        out["llm.system"] = name.split(".", 1)[0]
    if (inp := attrs.get("input")) is not None:
        out["input.value"] = inp
        out["input.mime_type"] = (
            "application/json" if _looks_json(inp) else "text/plain"
        )
    if (outp := attrs.get("output")) is not None:
        out["output.value"] = outp
        out["output.mime_type"] = (
            "application/json" if _looks_json(outp) else "text/plain"
        )
    if (t := attrs.get("tokens_input")) is not None:
        out["llm.token_count.prompt"] = t
    if (t := attrs.get("tokens_output")) is not None:
        out["llm.token_count.completion"] = t
    if (t := attrs.get("tokens_total")) is not None:
        out["llm.token_count.total"] = t
    if (sid := attrs.get("session_id")) is not None:
        out["session.id"] = sid
    if (uid := attrs.get("user_id")) is not None:
        out["user.id"] = uid
    if attrs.get("error"):
        out["exception.message"] = attrs["error"]
    if "eval_scores" in attrs:
        for k, v in attrs["eval_scores"].items():
            out[f"peekr.eval.{k}"] = v
    if "guardrails" in attrs:
        # Flatten guardrail outcomes — full payload also kept as JSON.
        out["peekr.guardrails"] = json.dumps(attrs["guardrails"], default=str)
    if "guardrail_violations" in attrs:
        out["peekr.guardrail_violations"] = ",".join(attrs["guardrail_violations"])
    return out


def _looks_json(value: object) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.lstrip()
    return stripped.startswith("{") or stripped.startswith("[")


class OTelExporter:
    """Exporter that sends spans to an OpenTelemetry tracer.

    Parameters
    ----------
    tracer:
        Optional pre-configured ``opentelemetry.trace.Tracer``. If omitted,
        the global tracer is used (so your app's existing OTel setup applies).
    endpoint, headers:
        Convenience shortcut: install an OTLP HTTP exporter on the spot.
        Requires ``opentelemetry-exporter-otlp-proto-http``.
    service_name:
        Resource attribute. Defaults to ``"peekr"``.
    """

    _is_storage = True  # subject to sampling — see context.should_persist

    def __init__(
        self,
        tracer=None,
        endpoint: Optional[str] = None,
        headers: Optional[dict] = None,
        service_name: str = "peekr",
    ) -> None:
        try:
            from opentelemetry import trace  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "OTelExporter needs opentelemetry-api / opentelemetry-sdk. "
                'Install with `pip install "peekr[otel]"`.'
            ) from exc

        if tracer is None and endpoint is not None:
            tracer = self._build_tracer(endpoint, headers or {}, service_name)
        self._tracer = tracer or trace.get_tracer("peekr")
        self._trace = trace
        # span_id -> live OTel span, used to set parent_context for children
        self._open: dict[str, object] = {}

    @staticmethod
    def _build_tracer(endpoint: str, headers: dict, service_name: str):
        from opentelemetry import trace  # type: ignore
        from opentelemetry.sdk.resources import Resource  # type: ignore
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore
            OTLPSpanExporter,
        )

        provider = TracerProvider(
            resource=Resource.create({"service.name": service_name})
        )
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=headers))
        )
        trace.set_tracer_provider(provider)
        return trace.get_tracer("peekr")

    def export(self, span) -> None:
        """Translate a peekr span into a fresh, already-ended OTel span."""
        try:
            from opentelemetry import trace  # type: ignore
            from opentelemetry.trace import Status, StatusCode  # type: ignore
        except ImportError:
            return

        d = span.to_dict()
        attrs = _llm_attributes(d)

        parent_ctx = None
        parent_id = d.get("parent_id")
        if parent_id and parent_id in self._open:
            parent_ctx = trace.set_span_in_context(self._open[parent_id])

        start_ns = int((d["start_time"] or 0) * 1e9)
        end_ns = int((d["end_time"] or d["start_time"] or 0) * 1e9)

        otel_span = self._tracer.start_span(
            name=d["name"],
            context=parent_ctx,
            start_time=start_ns,
            attributes={k: _stringify(v) for k, v in attrs.items()},
        )
        if d.get("status") == "error":
            otel_span.set_status(
                Status(StatusCode.ERROR, attrs.get("exception.message", ""))
            )
        else:
            otel_span.set_status(Status(StatusCode.OK))

        # Track open spans by peekr span_id so children can reference us.
        self._open[d["span_id"]] = otel_span
        otel_span.end(end_time=end_ns)
        # Root span: free children we tracked for this trace.
        if parent_id is None:
            self._open = {
                sid: s
                for sid, s in self._open.items()
                if getattr(s, "context", None) is None
                or getattr(s.context, "trace_id", None)
                != getattr(otel_span.context, "trace_id", None)
            }


def _stringify(value):
    """OTel attributes must be primitives or sequences of primitives."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return json.dumps(value, default=str)
    except Exception:
        return str(value)
