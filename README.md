<div align="center">

# peekr

**Observability and evaluation for AI agents.**

[![PyPI](https://img.shields.io/pypi/v/peekr)](https://pypi.org/project/peekr/)
[![CI](https://github.com/ashwanijha04/peekr/actions/workflows/ci.yml/badge.svg)](https://github.com/ashwanijha04/peekr/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

[Website](https://ashwanijha04.github.io/peekr) · [Docs](https://ashwanijha04.github.io/peekr/docs.html) · [PyPI](https://pypi.org/project/peekr/) · [TypeScript SDK](peekr-ts/README.md)

</div>

---

Peekr captures every LLM call, tool call, and framework step in your agent — what was sent, what came back, how long it took, and what it cost. Two lines of code, no backend, no account.

```python
import peekr
peekr.instrument()
```

That's it. Spans stream to `traces.jsonl` (or SQLite) and to your console. Inspect them with `peekr view`, find expensive calls with `peekr cost`, generate a self-contained dashboard with `peekr dashboard`, and score every output with built-in LLM-as-judge evaluators including RAGAS-style claim decomposition.

---

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [What you get](#what-you-get)
- [CLI](#cli)
- [Evaluators](#evaluators)
- [Dashboard](#dashboard)
- [Multi-tenant traces](#multi-tenant-traces)
- [Storage](#storage)
- [Supported clients](#supported-clients)
- [TypeScript SDK](#typescript-sdk)
- [Peekr Cloud](#peekr-cloud)
- [How it works](#how-it-works)
- [Contributing](#contributing)

---

## Install

```bash
pip install peekr                   # base
pip install "peekr[openai]"         # with OpenAI
pip install "peekr[anthropic]"      # with Anthropic
pip install "peekr[bedrock]"        # with AWS Bedrock
pip install "peekr[langchain]"      # with LangChain / LangGraph
pip install "peekr[llamaindex]"     # with LlamaIndex
pip install "peekr[crewai]"         # with CrewAI
pip install "peekr[otel]"           # with OpenTelemetry / OpenInference export
pip install "peekr[all]"            # everything
```

---

## Quick start

**1. Instrument once at startup.** Patches OpenAI, Anthropic, Bedrock, and any installed agent framework.

```python
import peekr
peekr.instrument()
```

**2. Trace your tools** so they appear in the same tree as LLM calls.

```python
from peekr import trace

@trace
def search_web(query: str) -> list[str]:
    return fetch_results(query)

@trace                       # async works
async def fetch_user(user_id: int) -> dict:
    return await db.get(user_id)
```

**3. View the trace.**

```bash
peekr view traces.jsonl          # tree view
peekr view --io traces.jsonl     # include inputs and outputs
peekr cost traces.jsonl          # cost breakdown + top hotspots
```

```
Trace a3f2b1c0  1243ms  891tok
────────────────────────────────────────────────
agent.run  1243ms
   └─ tool.search_web  210ms
         in:  {"query": "climate policy"}
         out: ["result1", "result2", ...]
   └─ openai.chat.completions [gpt-4o]  1033ms  891tok
         in:  [{"role": "user", "content": "..."}]
         out: "Based on recent research..."
```

---

## What you get

| Capability | API |
|---|---|
| Auto-instrumentation | `peekr.instrument()` — patches OpenAI, Anthropic, Bedrock, LangChain, LlamaIndex, CrewAI |
| Tool tracing | `@peekr.trace` on any sync or async function |
| Sessions | `with peekr.session(user_id="alice", tenant_id="acme"): ...` |
| Multi-tenant schema | `tenant_id` and `retention_class` first-class on every span |
| Alerts | `instrument(alerts=[peekr.alert.ErrorRate(0.05)])` |
| LLM-as-judge eval | `instrument(evaluators=[peekr.eval.Rubric("Be concise")])` |
| Hallucination detection | `instrument(evaluators=[peekr.eval.Hallucination()])` |
| Claim-level (RAGAS) hallucination | `Hallucination(detailed=True)` — per-claim verdicts |
| Drift dashboard | `peekr dashboard traces.db -o report.html` |
| Feedback + fine-tuning export | `peekr.feedback(trace_id, rating="good")` |
| A/B experiments | `@peekr.experiment(variants=["control", "test"])` |
| Trace replay | `peekr replay <trace_id>` |
| TypeScript SDK | `npm install @peekr/sdk` — same wire format |
| OpenTelemetry export | `add_exporter(peekr.OTelExporter())` — OpenInference-shaped spans into any OTel pipeline |

### Failure modes peekr catches that timing alone won't

A profiler tells you a function was slow. Peekr also tells you it returned the wrong shape and the LLM had no idea.

```
agent.run  2100ms
   └─ tool.fetch_user  12ms     out: null         ← tool returned null
   └─ openai.chat       2088ms  in: "User profile: null..."   ← LLM got garbage
```

Slow steps are obvious in the tree, with the cost broken out:

```
agent.run  4300ms
   └─ tool.search_web   3800ms  ← 88% of latency. Cache, don't swap models.
   └─ openai.chat        490ms
```

Token growth across runs surfaces unbounded conversation history:

```
Trace 1:  18,432 tokens
Trace 2:  21,104 tokens
Trace 3:  24,891 tokens   ← summarise after N turns
```

And prod-vs-local divergence is a tool I/O diff, not guesswork:

```
local:  out: [{"id": 1, "qty": 42}]
prod:   out: []   ← upstream pipeline bug, not agent logic
```

---

## CLI

### `peekr view`

Tree view of every trace, optionally with inputs and outputs.

```bash
peekr view traces.jsonl
peekr view --io traces.jsonl
peekr view traces.db          # SQLite works the same way
```

### `peekr cost`

Where money and time went, with a top-10 hotspots list ranked by composite cost-and-latency score.

```bash
peekr cost traces.jsonl
```

```
────────────────────────────────────────────────────────────
  peekr cost  ·  traces.jsonl
────────────────────────────────────────────────────────────
  Total spans        : 8,022
  LLM calls          : 85
  Errors             : 0
  Total input tokens : 130,807
  Total output tokens: 10,274
  Total LLM time     : 161.9s
  Total cost (est.)  : $0.14574
────────────────────────────────────────────────────────────

  Top 10 hottest calls  (60% cost · 40% latency):
  #   Operation                In      Out      Cost      ms  Model
  1   anthropic.messages    5,066     264 $ 0.00511   2965ms  claude-haiku-4-5
  2   anthropic.messages    3,924     376 $ 0.00464   3458ms  claude-haiku-4-5
  ...
```

### `peekr dashboard`

Self-contained HTML report — see [Dashboard](#dashboard).

### `peekr replay`

Re-run a stored trace through the live SDK, with the same inputs.

```bash
peekr replay a3f2b1c0
```

---

## Evaluators

Score every LLM output for groundedness, conciseness, or any custom rubric. Scores land on the span as `attributes.eval_scores`.

```python
import peekr

peekr.instrument(evaluators=[
    peekr.eval.Hallucination(),                  # 0.0 = hallucinated, 1.0 = grounded
    peekr.eval.Rubric("Answer is concise and direct"),
    peekr.eval.NotEmpty(),
    peekr.eval.NoError(),
])
```

```
openai.chat [gpt-4o]  843ms  312tok
   in:  "When was the Eiffel Tower built?"
   out: "The Eiffel Tower was built in 1923 by Frank Lloyd Wright."
   eval_scores: {Hallucination: 0.0, Rubric: 0.9, NotEmpty: 1.0}
```

For RAG flows, point `Hallucination` at the retrieved document instead of the prompt:

```python
peekr.eval.Hallucination(
    context_extractor=lambda span: span.attributes.get("retrieved_docs", "")
)
```

### Claim-level (RAGAS-style) detection

For *why* a response was scored low — not just *what* the score was — set `detailed=True`. The judge decomposes the output into atomic claims and assigns each one a verdict (`supported` / `contradicted` / `unsupported`), the same pipeline RAGAS Faithfulness uses.

```python
peekr.instrument(evaluators=[peekr.eval.Hallucination(detailed=True)])
```

```jsonc
// span.attributes.hallucination_details
{
  "total": 3, "supported": 1, "contradicted": 2, "unsupported": 0, "score": 0.33,
  "claims": [
    {"text": "The Eiffel Tower is in Paris",         "verdict": "supported"},
    {"text": "It was built in 1923",                 "verdict": "contradicted"},
    {"text": "It was designed by Frank Lloyd Wright", "verdict": "contradicted"}
  ]
}
```

Use simple mode for cheap monitoring across many traces; detailed mode for the cases worth investigating. Cost is roughly one judge call per scored span.

Query the lowest-scoring traces from SQLite to find regressions:

```sql
SELECT trace_id,
       json_extract(attributes, '$.eval_scores.Hallucination') AS score,
       json_extract(attributes, '$.output')                    AS output
FROM spans
WHERE score IS NOT NULL AND score < 0.5
ORDER BY start_time DESC;
```

---

## Dashboard

Generate a self-contained HTML observability report. No server, no build step — open the file in a browser, or attach it to a Slack message.

```bash
peekr dashboard traces.db -o report.html   # SQLite
peekr dashboard traces.jsonl               # writes ./dashboard.html
```

Five tabs (`1`–`5` to switch, `/` to search, `R` to clear filters, `Esc` to close panels):

| Tab | Purpose |
|---|---|
| **Overview** | Health hero (0–100), narrative summary of what's happening, top 3 action items |
| **Traces** | Search and filter every trace; click any row for full I/O, claim verdicts, citations |
| **Quality** | Rolling chart with thresholds, score distribution, channel × time heatmap |
| **Diagnose** | AI-generated likely causes, severity-tagged action lists, worst-offender cards with side-by-side context vs answer |
| **Help** | Setup checklist, glossary, evaluator snippets, troubleshooting |

A persistent filter bar (tenant · model · endpoint · time range) refilters every panel across every tab in one click. Tab and filter state live in the URL hash so links are shareable.

To populate the channel breakdown, peekr reads `attributes.model` automatically and `tenant_id` from the span schema. Attach an endpoint yourself in your request handler:

```python
from peekr import trace, get_current_span

@trace
def handle_request(req):
    get_current_span().attributes["endpoint"] = req.path
    return call_llm(...)
```

Full screenshots and tab-by-tab walkthrough → [docs](https://ashwanijha04.github.io/peekr/docs.html#dashboard).

---

## Multi-tenant traces

Every span carries two first-class fields — `tenant_id` (the customer org) and `retention_class` (a storage-tier hint). They're separate from `user_id` (the end-user) so a B2B agent can tag both without conflict.

```python
import peekr
peekr.instrument(tenant_id="acme", retention_class="default")

with peekr.session(user_id="alice", tenant_id="acme",
                   retention_class="long"):
    run_agent()
```

Resolution order, highest priority first:

1. `peekr.session(tenant_id=..., retention_class=...)`
2. `peekr.instrument(tenant_id=..., retention_class=...)`
3. Env vars `PEEKR_TENANT_ID` / `PEEKR_RETENTION_CLASS`

Both fields are top-level columns in SQLite (indexed) and top-level keys in JSONL — query without `json_extract`:

```sql
SELECT tenant_id, COUNT(*) FROM spans GROUP BY tenant_id;
SELECT * FROM spans WHERE retention_class = 'long' AND start_time > ?;
```

`retention_class` is a free-form string in the OSS SDK. Recommended values are `default`, `short`, `long`, and `pii`; the meaning of each is enforced by your storage tier (or by [Peekr Cloud](#peekr-cloud) when you're ready).

---

## Storage

```python
peekr.instrument()                    # JSONL — default, grep-able
peekr.instrument(storage="sqlite")    # SQLite — queryable, multi-process safe
peekr.instrument(storage="both")      # both
```

SQLite uses WAL mode so multiple processes (Docker, CI, parallel agents) can write at the same time. Query across runs:

```bash
# slowest tool calls
sqlite3 traces.db "
  SELECT name, ROUND(AVG(duration_ms)) avg_ms
  FROM spans GROUP BY name ORDER BY avg_ms DESC;"

# token spend by model
sqlite3 traces.db "
  SELECT json_extract(attributes,'\$.model')        AS model,
         SUM(json_extract(attributes,'\$.tokens_total')) AS tokens
  FROM spans GROUP BY model;"

# all errors
sqlite3 traces.db "
  SELECT name, trace_id, json_extract(attributes,'\$.error') AS msg
  FROM spans WHERE status = 'error';"
```

### OpenTelemetry export

Ship peekr spans into any OTel-compatible backend (Datadog, Honeycomb, Grafana Tempo, Arize Phoenix, Langfuse-OTel, etc.) by translating attributes into the [OpenInference semantic conventions](https://github.com/Arize-ai/openinference) the LLM observability ecosystem uses.

```bash
pip install "peekr[otel]"
```

```python
import peekr
from peekr.exporters import add_exporter

peekr.instrument()
add_exporter(peekr.OTelExporter())                    # uses your app's existing OTel setup
add_exporter(peekr.OTelExporter(endpoint="https://api.honeycomb.io",
                                headers={"x-honeycomb-team": "..."}))   # or configure inline
```

No agent, no collector, no separate process. Peekr writes OpenInference-shaped spans in-process, and any OTel pipeline you already operate consumes them.

### Custom exporters

Ship spans to any backend by implementing one method:

```python
from peekr.exporters import add_exporter

class MyExporter:
    def export(self, span):
        requests.post("https://my-backend.com/spans", json=span.to_dict())

peekr.instrument()
add_exporter(MyExporter())
```

### `@trace` options

```python
@trace                        # auto-names from module.function, captures I/O
@trace(name="tool.search")    # custom span name
@trace(capture_io=False)      # skip args/output (e.g. secrets)
```

---

## Supported clients

**LLM SDKs**

| Provider | SDK | Install |
|---|---|---|
| OpenAI | `openai` | `pip install "peekr[openai]"` |
| Anthropic | `anthropic` | `pip install "peekr[anthropic]"` |
| AWS Bedrock | `boto3` | `pip install "peekr[bedrock]"` |

**Agent frameworks**

| Framework | Package | Install |
|---|---|---|
| LangChain / LangGraph | `langchain-core` | `pip install "peekr[langchain]"` |
| LlamaIndex | `llama-index-core` | `pip install "peekr[llamaindex]"` |
| CrewAI | `crewai` | `pip install "peekr[crewai]"` |

`peekr.instrument()` detects whichever SDKs and frameworks are installed and patches them. Streaming is supported across all LLM SDKs. Frameworks emit chain / tool / retriever / agent / LLM spans nested in the order they actually executed:

```
crewai.crew.kickoff                       3.4s
  └─ crewai.task.execute                  3.4s   task=plan_trip
       └─ crewai.agent.execute_task       3.4s   agent=planner
            └─ openai.chat.completions    1.2s   gpt-4o  · 891tok
            └─ langchain.tool.search_web  2.1s
```

---

## TypeScript SDK

```bash
npm install @peekr/sdk
```

```ts
import { instrument, wrap, trace, withSession } from "@peekr/sdk";
import OpenAI from "openai";

instrument({ jsonlPath: "./traces.jsonl" });
const openai = wrap(new OpenAI());

await withSession(
  { user_id: "alice", tenant_id: "acme" },
  async () => {
    await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "Summarise the docs above" }],
    });
  },
);
```

The TypeScript SDK writes the same JSONL schema as Python, so a Node app's traces work with `peekr view`, `peekr cost`, and `peekr dashboard` unchanged. Full reference → [`peekr-ts/README.md`](peekr-ts/README.md).

---

## Peekr Cloud

The OSS SDK runs in your process, writes to local files, and is **MIT licensed forever** — that's not changing. When a single-process file isn't the right fit any more (multiple services, a team that needs shared dashboards, longer retention, audit-grade trace storage), Peekr Cloud is the optional managed backend.

The wire format is already in v0.3 — `tenant_id` and `retention_class` exist specifically so the spans you produce today work without modification when you connect.

```python
import peekr
peekr.instrument(
    tenant_id="acme",
    exporter=peekr.HTTPExporter(
        endpoint="https://ingest.peekr.cloud",
        api_key="pk_live_…",
    ),
)
```

`HTTPExporter` ships as a stub in v0.3 — the constructor signature is stable so you can wire your call sites today, and they won't change when the implementation lands. Until then it raises `NotImplementedError` on `.export()` so a misconfigured pipeline fails loudly rather than silently dropping spans.

**Get on the waitlist** → [GitHub Discussions](https://github.com/ashwanijha04/peekr/discussions).

---

## How it works

`instrument()` monkey-patches the OpenAI, Anthropic, and Bedrock SDK methods before your code runs. Python resolves function references at call time, so every subsequent call hits the wrapper without any change to your code.

Parent / child span relationships are tracked through `contextvars.ContextVar`, which propagates correctly across `async / await` without manual threading. The TypeScript SDK uses Node's `AsyncLocalStorage` for the same reason.

---

## Contributing

```bash
git clone https://github.com/ashwanijha04/peekr
cd peekr
pip install -e ".[dev]"
pytest
```

Open an issue before large changes. PRs welcome.

---

<div align="center">

[Website](https://ashwanijha04.github.io/peekr) · [Docs](https://ashwanijha04.github.io/peekr/docs.html) · [PyPI](https://pypi.org/project/peekr/) · [TypeScript SDK](peekr-ts/README.md) · MIT License

</div>
