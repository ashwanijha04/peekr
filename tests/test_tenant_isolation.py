"""Tenant isolation tests.

Verifies that when multiple tenants run evaluators in parallel, their
hallucination scores, claim-level details, and trace data never leak across
tenant boundaries — neither in-process (the EvalExporter operating on
many spans) nor when the dashboard is filtered by tenant.

Peekr models a tenant via the `user_id` ContextVar set inside `peekr.session()`,
which is written onto `span.attributes["user_id"]` by the context machinery.
The tests below treat that field as the tenant identifier.
"""

from __future__ import annotations

import asyncio
import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from peekr.eval import EvalExporter
from peekr.eval.hallucination import Hallucination
from peekr.span import Span


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_llm_span(tenant: str, output: str, input_text: str = "ctx", trace_id: str | None = None) -> Span:
    s = Span(name="openai.chat.completions", trace_id=trace_id or f"trace-{tenant}")
    s.attributes["input"] = input_text
    s.attributes["output"] = output
    s.attributes["user_id"] = tenant
    s.finish()
    return s


def fake_judge(score_text: str) -> MagicMock:
    mock = MagicMock()
    mock.chat.completions.create.return_value.choices[0].message.content = score_text
    return mock


# ---------------------------------------------------------------------------
# 1. Eval scores never leak between spans
# ---------------------------------------------------------------------------

class TestEvalScoreIsolation:
    def test_scores_attach_only_to_their_own_span(self):
        """EvalExporter writes scores onto the span passed in; never onto siblings."""
        s_a = make_llm_span("acme",   output="answer A")
        s_b = make_llm_span("globex", output="answer B")

        with patch("peekr.eval.hallucination.openai") as mock_oai:
            # Return distinct scores so we can tell which call produced which write
            mock_oai.chat.completions.create.side_effect = [
                MagicMock(choices=[MagicMock(message=MagicMock(content="0.10"))]),
                MagicMock(choices=[MagicMock(message=MagicMock(content="0.90"))]),
            ]
            exporter = EvalExporter(evaluators=[Hallucination()])
            exporter.export(s_a)
            exporter.export(s_b)

        assert s_a.attributes["eval_scores"]["Hallucination"] == pytest.approx(0.10)
        assert s_b.attributes["eval_scores"]["Hallucination"] == pytest.approx(0.90)
        # tenant tags untouched
        assert s_a.attributes["user_id"] == "acme"
        assert s_b.attributes["user_id"] == "globex"

    def test_detailed_breakdown_does_not_leak(self):
        """A detailed-mode breakdown is only written onto the span it belongs to."""
        s_a = make_llm_span("acme",   output="A invented Frank Lloyd Wright designed it")
        s_b = make_llm_span("globex", output="B is factually correct")

        responses = [
            MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps({"claims": [
                {"text": "Frank Lloyd Wright designed it", "verdict": "contradicted"},
            ]})))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps({"claims": [
                {"text": "B is correct", "verdict": "supported"},
            ]})))]),
        ]
        with patch("peekr.eval.hallucination.openai") as mock_oai:
            mock_oai.chat.completions.create.side_effect = responses
            exporter = EvalExporter(evaluators=[Hallucination(detailed=True)])
            exporter.export(s_a)
            exporter.export(s_b)

        assert s_a.attributes["hallucination_details"]["contradicted"] == 1
        assert s_b.attributes["hallucination_details"]["supported"] == 1
        # The other tenant's claims must NOT appear on this span
        a_claims = [c["text"] for c in s_a.attributes["hallucination_details"]["claims"]]
        b_claims = [c["text"] for c in s_b.attributes["hallucination_details"]["claims"]]
        assert "B is correct" not in a_claims
        assert "Frank Lloyd Wright designed it" not in b_claims


# ---------------------------------------------------------------------------
# 2. Concurrent evaluation across tenants
# ---------------------------------------------------------------------------

class TestConcurrentTenants:
    def test_threaded_evaluation_keeps_scores_on_correct_spans(self):
        """Run many tenants in parallel threads; nothing should cross-contaminate."""
        n_tenants = 25
        spans = [make_llm_span(f"tenant_{i}", output=f"answer-{i}") for i in range(n_tenants)]

        # Each judge call returns a score uniquely tied to the tenant index.
        # We can't reuse `side_effect` (order isn't deterministic across threads),
        # so we key on the output text.
        def judge(*args, **kwargs):
            content = kwargs["messages"][0]["content"]
            # Extract the tenant index from "answer-{i}" inside the prompt
            for line in content.splitlines():
                if "answer-" in line:
                    i = int(line.split("answer-")[1].split()[0].strip("."))
                    score = round(i / n_tenants, 3)
                    return MagicMock(choices=[MagicMock(message=MagicMock(content=str(score)))])
            return MagicMock(choices=[MagicMock(message=MagicMock(content="0.5"))])

        with patch("peekr.eval.hallucination.openai") as mock_oai:
            mock_oai.chat.completions.create.side_effect = judge

            exporter = EvalExporter(evaluators=[Hallucination()])
            threads = [threading.Thread(target=exporter.export, args=(s,)) for s in spans]
            for t in threads: t.start()
            for t in threads: t.join()

        for i, span in enumerate(spans):
            expected = round(i / n_tenants, 3)
            actual = span.attributes["eval_scores"]["Hallucination"]
            assert actual == pytest.approx(expected, abs=1e-3), \
                f"Tenant {i} got score {actual} instead of {expected} — leak!"
            assert span.attributes["user_id"] == f"tenant_{i}"

    def test_async_evaluation_isolated_per_session(self):
        """peekr.session() uses ContextVars — concurrent async tasks must not bleed."""
        from peekr.session import _user_id

        async def per_tenant(tenant_id: str) -> str:
            token = _user_id.set(tenant_id)
            try:
                # Yield once so other tasks can interleave on the same loop
                await asyncio.sleep(0)
                seen = _user_id.get()
            finally:
                _user_id.reset(token)
            return seen

        async def main():
            results = await asyncio.gather(*[per_tenant(f"t{i}") for i in range(50)])
            return results

        results = asyncio.run(main())
        # Each coroutine must see exactly its own tenant id, never another's.
        assert results == [f"t{i}" for i in range(50)]


# ---------------------------------------------------------------------------
# 3. Dashboard / cost-report filtering by tenant
# ---------------------------------------------------------------------------

class TestDashboardTenantFilter:
    def _multi_tenant_corpus(self, tmp_path):
        """Two tenants, very different score distributions."""
        import time
        t0 = time.time()
        spans = []
        for i in range(20):
            tenant = "acme" if i % 2 == 0 else "globex"
            score = 0.95 if tenant == "acme" else 0.25
            spans.append({
                "name": "openai.chat.completions",
                "trace_id": f"trace-{i}",
                "span_id": f"span-{i}",
                "parent_id": None,
                "start_time": t0 + i,
                "end_time": t0 + i + 0.3,
                "duration_ms": 300.0,
                "status": "ok",
                "attributes": {
                    "model": "gpt-4o-mini",
                    "user_id": tenant,
                    "input": "Q?",
                    "output": f"A{i}",
                    "tokens_total": 100,
                    "eval_scores": {"Hallucination": score},
                },
            })
        path = tmp_path / "traces.jsonl"
        with open(path, "w") as f:
            for s in spans:
                f.write(json.dumps(s) + "\n")
        return path, spans

    def test_dashboard_computes_unfiltered_drift(self, tmp_path):
        """Sanity: full corpus shows a mid-range average across tenants."""
        from peekr.dashboard import _drift
        _, spans = self._multi_tenant_corpus(tmp_path)
        d = _drift(spans)["Hallucination"]
        assert d is not None
        assert 0.5 < d["baseline"] < 0.7   # mix of 0.95 and 0.25 averages near 0.6

    def test_per_tenant_drift_segregates_correctly(self, tmp_path):
        """Filter by tenant; drift numbers must match that tenant alone."""
        from peekr.dashboard import _drift
        _, spans = self._multi_tenant_corpus(tmp_path)

        acme   = [s for s in spans if s["attributes"].get("user_id") == "acme"]
        globex = [s for s in spans if s["attributes"].get("user_id") == "globex"]

        d_acme   = _drift(acme)["Hallucination"]
        d_globex = _drift(globex)["Hallucination"]
        # Acme is uniformly 0.95, Globex uniformly 0.25
        assert d_acme["baseline"]   == pytest.approx(0.95)
        assert d_acme["current"]    == pytest.approx(0.95)
        assert d_globex["baseline"] == pytest.approx(0.25)
        assert d_globex["current"]  == pytest.approx(0.25)
        # No bleed: one tenant's score must never show up in the other's window
        assert abs(d_acme["baseline"] - d_globex["baseline"]) > 0.5

    def test_worst_offenders_per_tenant(self, tmp_path):
        """A worst-offenders table built from one tenant's spans must not contain the other tenant's data."""
        from peekr.dashboard import _worst_offenders
        _, spans = self._multi_tenant_corpus(tmp_path)

        globex = [s for s in spans if s["attributes"].get("user_id") == "globex"]
        worst = _worst_offenders(globex, k=5)
        assert worst, "expected some globex offenders"
        # Every offender's trace_id must belong to a globex span
        globex_trace_ids = {s["trace_id"] for s in globex}
        assert all(w["trace_id"] in globex_trace_ids for w in worst)
