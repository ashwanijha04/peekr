<div align="center">

# peekr

**Agents are black boxes. Peekr makes them transparent.**

[![PyPI](https://img.shields.io/pypi/v/peekr)](https://pypi.org/project/peekr/)
[![CI](https://github.com/ashwanijha04/peekr/actions/workflows/ci.yml/badge.svg)](https://github.com/ashwanijha04/peekr/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

[Website](https://ashwanijha04.github.io/peekr) · [Docs](https://ashwanijha04.github.io/peekr/docs.html) · [PyPI](https://pypi.org/project/peekr/) · [Changelog](#)

</div>

---

`cProfile` tells you where CPU time went in your Python code. Peekr tells you where time, tokens, and money went in your agent — and what each step actually saw and returned.

```
# cProfile
function           calls   cumtime
search_results     1       3.8s
openai.create      2       0.9s

# peekr
tool.search_web    3800ms          ← same bottleneck, now you can fix it
openai.chat        490ms  891tok   ← plus token cost you'd never see in cProfile
```

But agents fail for reasons a profiler can't catch: a tool returned `null`, the LLM received a malformed prompt, history grew until it pushed the system prompt out of the context window. Peekr captures the **semantics** — inputs, outputs, LLM context — not just timing.

Two lines to add, no backend required.

```bash
pip install peekr
```

```python
import peekr
peekr.instrument()

# Your existing agent code — zero changes needed
```

---

## How to use it

### Step 1 — Instrument

Call `peekr.instrument()` once, before any LLM calls. It patches the OpenAI and Anthropic SDKs automatically.

```python
import peekr
peekr.instrument()

import openai

response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Summarize this doc"}]
)
```

Every LLM call is now captured. Peekr writes spans to `traces.jsonl` and prints them live to the console.

---

### Step 2 — Trace your tools

Decorate your tool functions with `@trace` so they appear in the tree alongside LLM calls:

```python
from peekr import trace

@trace
def search_web(query: str) -> list[str]:
    return fetch_results(query)

@trace(name="tool.calculator")
def calculate(expression: str) -> float:
    return eval(expression)

@trace   # async works too
async def fetch_user(user_id: int) -> dict:
    return await db.get(user_id)
```

Decorated functions nest automatically under whatever called them — no wiring needed.

---

### Step 3 — View the trace

```bash
peekr view traces.jsonl          # tree view — every trace, nested
peekr view --io traces.jsonl     # include inputs and outputs
peekr cost traces.jsonl          # cost breakdown + top-10 hotspot calls
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

Now you can see exactly what happened — what went in, what came out, how long each step took, how many tokens were used.

---

### `peekr cost` — find what's expensive

`peekr cost` reads a traces file and answers: where did my money and time go?

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
  Total cost (est.)  : $0.14574  (Haiku rates: $0.80/$4.00 per M)
────────────────────────────────────────────────────────────

  Cost by operation:
  Operation                                        Calls       Cost   Avg/call   Avg ms
  ──────────────────────────────────────────────────
  anthropic.messages  [claude-haiku-4-5-20251001]     85  $ 0.14574  $ 0.00171    1905ms
  tool.search_web                                      42  $ 0.00000  $ 0.00000     210ms

  Top 10 hottest calls  (60% cost · 40% latency):
  #   Operation                           In      Out      Cost      ms  Model
  ──────────────────────────────────────────────────
  1   anthropic.messages               5,066      264 $ 0.00511   2965ms  claude-haiku-4-5-20251001
  2   anthropic.messages               3,924      376 $ 0.00464   3458ms  claude-haiku-4-5-20251001
  ...
```

The top-10 list ranks by a composite score (60% cost, 40% latency) so a call that's both slow and expensive ranks above one that's merely expensive. Use it to decide what to cache, compress, or swap for a cheaper model.

---

## What it profiles

A CPU profiler tells you a function was slow. Peekr tells you a function was slow, returned bad data, and passed it to an LLM that had no idea.

> **Full examples with annotated traces → [docs](https://ashwanijha04.github.io/peekr/docs.html)**

### Wrong answers

The exact prompt that was sent — not what you think was sent, what was *actually* sent. Spot bad tool output before it reaches the LLM.

```
agent.run  2100ms
   └─ tool.fetch_user  12ms
         out: null                ← returned null, agent didn't check
   └─ openai.chat [gpt-4o]  2088ms
         in:  "User profile: null..."   ← LLM received garbage
```

### Slow responses

```
agent.run  4300ms
   └─ tool.search_web   3800ms   ← 88% of time. Cache this, not swap models.
   └─ openai.chat        490ms
```

### High token costs

```
Trace 1:  18,432 tokens
Trace 2:  21,104 tokens
Trace 3:  24,891 tokens   ← growing = unbounded history. Summarize after 5 turns.
```

### Prod vs local bugs

```
# local:  out: [{"id": 1, "qty": 42}]
# prod:   out: []    ← data pipeline bug, not agent logic
```

---

## What's in v0.2

| Capability | API |
|---|---|
| **Session tracing** | `with peekr.session(user_id="u1"):` |
| **Alerts** | `instrument(alerts=[peekr.alert.ErrorRate(0.05)])` |
| **LLM-as-judge eval** | `instrument(evaluators=[peekr.eval.Rubric("Be concise")])` |
| **Hallucination detection** | `instrument(evaluators=[peekr.eval.Hallucination()])` |
| **Claim-level (RAGAS) hallucination** | `Hallucination(detailed=True)` — per-claim verdicts |
| **Drift dashboard** | `peekr dashboard traces.db -o report.html` |
| **Feedback + fine-tuning export** | `peekr.feedback(trace_id, rating="good")` |
| **A/B experiments** | `@peekr.experiment(variants=["control", "test"])` |
| **Trace replay** | `peekr replay <trace_id>` |

### Hallucination detection & eval scores

Score every LLM output for groundedness, conciseness, or any custom rubric. Scores land on the span as `attributes.eval_scores` — visible in `peekr view` and queryable from SQLite.

```python
import peekr

peekr.instrument(evaluators=[
    peekr.eval.Hallucination(),                       # 0.0 = fully hallucinated, 1.0 = fully grounded
    peekr.eval.Rubric("Answer is concise and direct"),
    peekr.eval.NotEmpty(),
    peekr.eval.NoError(),
])
```

```
openai.chat [gpt-4o]  843ms  312tok
   in:  "When was the Eiffel Tower built?"
   out: "The Eiffel Tower was built in 1923 by Frank Lloyd Wright."
   eval_scores: {Hallucination: 0.0, Rubric: 0.9, NotEmpty: 1.0}   ← invented facts caught
```

RAG flow? Point `Hallucination` at the retrieved document instead of the prompt:

```python
peekr.eval.Hallucination(
    context_extractor=lambda span: span.attributes.get("retrieved_docs", "")
)
```

Query the lowest-scoring traces to find regressions:

```sql
SELECT trace_id,
       json_extract(attributes,'$.eval_scores.Hallucination') AS hallucination,
       json_extract(attributes,'$.output') AS output
FROM spans
WHERE hallucination IS NOT NULL AND hallucination < 0.5
ORDER BY start_time DESC;
```

### Claim-level detection (RAGAS-style)

For *why* a response was scored low — not just *what* the score was — enable `detailed=True`. The judge decomposes the output into atomic claims and assigns each one a verdict (`supported` / `contradicted` / `unsupported`), the same pipeline RAGAS Faithfulness uses. The score is `supported_count / total_claims`, and the full breakdown is stored on the span.

```python
peekr.instrument(evaluators=[peekr.eval.Hallucination(detailed=True)])
```

```jsonc
// span.attributes.hallucination_details
{
  "total": 3, "supported": 1, "contradicted": 2, "unsupported": 0, "score": 0.33,
  "claims": [
    {"text": "The Eiffel Tower is in Paris",        "verdict": "supported"},
    {"text": "It was built in 1923",                "verdict": "contradicted"},
    {"text": "It was designed by Frank Lloyd Wright","verdict": "contradicted"}
  ]
}
```

Same prompt cost as before (~one judge call), more output tokens. Use the simple mode for cheap monitoring across many traces and detailed mode for the cases you want to investigate.

### Dashboard — `peekr dashboard`

Generate a self-contained, tabbed HTML observability report from your traces. Designed as a drop-in for any RAG or memory/agent pipeline: open the file in any browser, no server, no build step.

```bash
peekr dashboard traces.db -o report.html   # SQLite
peekr dashboard traces.jsonl               # JSONL — writes ./dashboard.html
open report.html
```

**Five tabs, one URL.** Keyboard shortcuts `1`–`5` switch tabs, `/` focuses search, `R` clears filters, `Esc` closes the trace detail panel. Tab state lives in the URL hash so links are shareable.

| Tab | When to open it |
|---|---|
| **Overview** | First time you look at the dashboard, or after an incident. Health hero (0–100 score, traffic-light dot), plain-English narrative, metric cards with sparklines, top 3 action items. |
| **Traces** | "What did this specific call look like?" — search bar (trace ID, model, content, error), sortable table, click any row to slide out a detail panel with context vs answer, claim verdicts, citations, and per-call action items. |
| **Quality** | Trend monitoring. Rolling chart with warning/critical threshold lines, score distribution histogram, channel × time heatmap, claim-verdict doughnut, citation panel. |
| **Diagnose** | When something's wrong. AI-generated "Likely causes & next steps" with severity badges and fix lists, plus the full worst-offenders panel with side-by-side context/answer highlighting. |
| **Help** | First-time setup, glossary, evaluator configuration snippets, troubleshooting, keyboard shortcuts. The setup checklist auto-ticks as you wire things up. |

**Persistent filter bar** at the top of every tab — tenant · model · endpoint chips · time range (5m / 15m / 30m / 1h / 24h / 7d / 30d / Custom datetime). One click refilters every panel across all tabs.

**What you'll see on the Overview tab:**

```
┌──────────────────────────────────────────────────────────────────────┐
│ Filter chips:  [Tenant] [Model] [Endpoint] [All / 1h / 24h / 7d]    │
├──────────────────────────────────────────────────────────────────────┤
│ ● Hallucination health: 66/100   needs attention                     │
│   30 of 134 scored calls flagged. ↓ 12 pts vs baseline 0.78.         │
│   [───sparkline───]                                                  │
├──────────────────────────────────────────────────────────────────────┤
│ What's happening:                                                    │
│ › Hallucination dropped 27 points from baseline (0.89 → 0.62).       │
│ › Worst channel: gpt-4o-mini · acme · /api/qa — mean 0.31 / 8 calls. │
│ › 4 of 12 citations were invented (33%).                             │
├──────────────────────────────────────────────────────────────────────┤
│ Hallucination 0.66 ▼-12  Rubric 0.84  Citations 0.60 ▼-15  Errors 6 │
│ → 30 flagged       → stable      → 4 invented refs      → 4.3% calls│
├──────────────────────────────────────────────────────────────────────┤
│ Likely causes & next steps:                                          │
│ [HIGH]   Model is inventing citations (33% of detected references)   │
│         What to try:                                                 │
│         1. Inspect retrieval: log returned chunks for a flagged span │
│         2. Tighten prompt: "Cite only sources in the context above"  │
│         3. Verify citations post-hoc against retrieved chunks        │
│         4. Try hybrid retrieval (BM25 + dense) for keyword queries   │
│                                                                      │
│ [MED]   Failures concentrated on endpoint = /api/qa                  │
│         What to try:                                                 │
│         1. Diff last week of deploys touching this endpoint          │
│         2. Compare this endpoint's prompt vs a healthier one         │
├──────────────────────────────────────────────────────────────────────┤
│ Score over time   (rolling 20-call mean, thresholds at 0.7 / 0.5)    │
│   1.0 ──────────────────────╮                                        │
│              ───╮          │   ╮                                     │
│   0.7 ─ ─ ─ ─ ╲ ╲ ─ ─ ─ ─ ╲ ─ ╲─ ─ warning                          │
│   0.5 ─ ─ ─ ─ ─╲╲─ ─ ─ ─ ─╲─╮ critical                              │
│   0.0 ──────────────────────╰╰──                                     │
├──────────────────────────────────────────────────────────────────────┤
│ Failure breakdown by channel & time     (red=halluc, green=grounded) │
│                                                                      │
│   model                10:00  11:00  12:00  13:00  14:00  15:00      │
│   gpt-4o-mini          0.89   0.71   0.42   0.31   0.28   0.22       │
│   gpt-4o               0.91   0.88   0.76   0.81   0.79   0.80       │
│   claude-opus-4-5      0.93   0.92   0.94   0.91   0.92   0.94       │
│                                                                      │
│   (click a red cell to filter the dashboard to that channel)         │
├──────────────────────────────────────────────────────────────────────┤
│ #1 ⬤0.00  gpt-4o-mini · acme · /api/qa                              │
│ Q: When was the Eiffel Tower built and by whom?                      │
│                                                                      │
│ ┌─SOURCE CONTEXT────────────┐ ┌─MODEL ANSWER──────────────┐         │
│ │ The Eiffel Tower was      │ │ Built in [1923]contra by  │         │
│ │ completed in 1889 for the │ │ [Frank Lloyd Wright]contra│         │
│ │ Paris World's Fair...     │ │ for the [London Olympics] │         │
│ └───────────────────────────┘ └───────────────────────────┘         │
│ contradicted: "1923" · contradicted: "Frank Lloyd Wright"            │
│ unsupported:  "London Olympics"                                      │
│                                                                      │
│ ▌What to try for this call:                                          │
│ • Numeric contradiction — add "be exact about dates" to prompt       │
│ • Proper noun substitution — instruct the model not to substitute    │
│ • Move retrieved context closer to the question (recency bias)       │
└──────────────────────────────────────────────────────────────────────┘
```

**How it helps you ship reliably:**

| Problem you have | Where the dashboard takes you |
|---|---|
| "Our app suddenly started hallucinating." | Health hero → red, narrative names the worst channel, recommendations propose causes |
| "Which model / tenant / endpoint is bad?" | Failure breakdown heatmap. Click the red cell → everything refilters |
| "When did it start?" | Heatmap rows go green → red over time buckets. Use the time-range chips to bisect |
| "What's the model actually getting wrong?" | Worst-offender cards show source context vs answer, with claims highlighted in colour |
| "How do I fix it?" | Per-span action box on every offender card prescribes fixes tied to that exact failure pattern (numeric drift, invented citations, retrieval miss, etc.) |
| "Is this a one-off or a pattern?" | Aggregate "Likely causes & next steps" panel re-runs the diagnostic engine on every filter change |

**Tag spans for the channel breakdown** — peekr reads `attributes.model` automatically, plus `attributes.user_id` (set via `peekr.session(user_id=...)`) for the tenant chip. For endpoint, attach it yourself in your request handler:

```python
from peekr import trace, get_current_span

@trace
def handle_request(req):
    get_current_span().attributes["endpoint"] = req.path
    return call_llm(...)
```

Charts are Chart.js loaded from a CDN. The data is embedded directly in the HTML file — no server, no build step, send it to a teammate as an attachment.

## Languages

| Runtime | Package | Status |
|---|---|---|
| **Python** | `pip install peekr` (this repo) | Stable — instrumentation, evaluators, dashboard, CLI |
| **TypeScript / Node.js** | `npm install @peekr/sdk` (`peekr-ts/` in this repo) | Instrumentation + JSONL exporter. Run analysis with the Python CLI. |

The TypeScript SDK writes the same JSONL schema as the Python one, so a Node app's traces can be analysed with `peekr view`, `peekr cost`, and `peekr dashboard` (which stay in Python — write-once, language-agnostic). See [`peekr-ts/README.md`](peekr-ts/README.md).

## Supported clients

**LLM SDKs**

| Provider | SDK | Install |
|---|---|---|
| **OpenAI** | `openai` | `pip install "peekr[openai]"` |
| **Anthropic** | `anthropic` | `pip install "peekr[anthropic]"` |
| **AWS Bedrock** | `boto3` | `pip install "peekr[bedrock]"` |

**Agent frameworks**

| Framework | Package | Install |
|---|---|---|
| **LangChain / LangGraph** | `langchain-core` | `pip install "peekr[langchain]"` |
| **LlamaIndex** | `llama-index-core` | `pip install "peekr[llamaindex]"` |
| **CrewAI** | `crewai` | `pip install "peekr[crewai]"` |

All auto-instrument with the same two lines — `peekr.instrument()` detects whichever SDKs and frameworks are installed and patches them. Streaming is supported across all LLM SDKs; frameworks emit chain / tool / retriever / agent / LLM spans nested in the order they actually executed.

```python
import peekr
peekr.instrument()

# OpenAI
import openai
openai.chat.completions.create(model="gpt-4o", messages=[...])

# Anthropic
import anthropic
anthropic.Anthropic().messages.create(model="claude-opus-4-5", messages=[...])

# Bedrock
import boto3
boto3.client("bedrock-runtime").converse(modelId="anthropic.claude-3-haiku-20240307-v1:0", messages=[...])
```

### Framework traces

For agent frameworks, peekr installs a callback handler / monkey-patches the execution surface so every chain, tool, retriever, agent step and LLM call shows up as its own span — nested in the order it actually executed.

```python
import peekr
peekr.instrument()

# LangChain — chain → tool → llm spans, all nested
from langchain.agents import AgentExecutor
agent_executor.invoke({"input": "what's the weather in NYC?"})

# LlamaIndex — query → retrieve → llm spans
from llama_index.core import VectorStoreIndex
index.as_query_engine().query("summarize this doc")

# CrewAI — crew.kickoff → task.execute → agent.execute_task → llm spans
from crewai import Crew
Crew(agents=[...], tasks=[...]).kickoff()
```

```
crewai.crew.kickoff                       3.4s
  └─ crewai.task.execute                  3.4s   task=plan_trip
       └─ crewai.agent.execute_task       3.4s   agent=planner
            └─ openai.chat.completions    1.2s   gpt-4o  · 891tok
            └─ langchain.tool.search_web  2.1s
```

## Installation

```bash
pip install peekr                   # base
pip install "peekr[openai]"         # with OpenAI
pip install "peekr[anthropic]"      # with Anthropic
pip install "peekr[bedrock]"        # with AWS Bedrock
pip install "peekr[langchain]"      # with LangChain
pip install "peekr[llamaindex]"     # with LlamaIndex
pip install "peekr[crewai]"         # with CrewAI
pip install "peekr[all]"            # everything
```

---

## Storage options

```python
peekr.instrument()                      # JSONL — default, grep-able
peekr.instrument(storage="sqlite")      # SQLite — queryable, multi-process safe
peekr.instrument(storage="both")        # both at once
```

### SQLite — query your traces with SQL

SQLite storage uses WAL mode so multiple processes (Docker, CI, parallel agents) can write safely at the same time. And because it's SQLite, you can query across runs:

```bash
# slowest tool calls
sqlite3 traces.db "
  SELECT name, ROUND(AVG(duration_ms)) avg_ms
  FROM spans GROUP BY name ORDER BY avg_ms DESC;"

# token spend by model
sqlite3 traces.db "
  SELECT json_extract(attributes,'$.model') model,
         SUM(json_extract(attributes,'$.tokens_total')) tokens
  FROM spans GROUP BY model;"

# all errors
sqlite3 traces.db "
  SELECT name, trace_id, json_extract(attributes,'$.error') msg
  FROM spans WHERE status = 'error';"

# cost growth over time
sqlite3 traces.db "
  SELECT trace_id,
         SUM(json_extract(attributes,'$.tokens_total')) total
  FROM spans GROUP BY trace_id ORDER BY start_time;"
```

View SQLite traces the same way as JSONL:

```bash
peekr view traces.db
peekr view --io traces.db
peekr cost traces.db
```

---

## @trace options

```python
@trace                        # auto-names from module.function, captures io
@trace(name="tool.search")    # custom span name
@trace(capture_io=False)      # skip capturing args/output (e.g. secrets)
```

---

## Custom exporters

Ship spans to any backend by implementing a single method:

```python
from peekr.exporters import add_exporter

class MyExporter:
    def export(self, span):
        requests.post("https://my-backend.com/spans", json=span.to_dict())

peekr.instrument()
add_exporter(MyExporter())
```

---

## How it works

`instrument()` monkey-patches the OpenAI and Anthropic SDK methods before your code runs. Python resolves function references at call time, so every subsequent call hits the wrapper with zero changes to your code.

Parent/child span relationships are tracked via Python's `contextvars.ContextVar`, which propagates correctly across `async/await` without manual threading.

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

[Website](https://ashwanijha04.github.io/peekr) · [Docs](https://ashwanijha04.github.io/peekr/docs.html) · [PyPI](https://pypi.org/project/peekr/) · MIT License

</div>
