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

### `peekr serve` — local web dashboard

Prefer a browser to a terminal? `peekr serve` boots a local-first dashboard that reads the same `traces.db` / `traces.jsonl` — no signup, no remote backend, nothing leaves your machine.

```bash
pip install "peekr[serve]"
peekr serve --port 8000      # → http://127.0.0.1:8000
peekr serve --db traces.db   # explicit storage
peekr serve --jsonl traces.jsonl
```

What you get:

- **Trace list** with filters for user, session, time, cost, errors, eval scores and guardrail findings. Paginated 50 per page.
- **Trace detail** — full span tree, click any span to lazy-load its input / output. Eval scores and guardrail findings render inline so you see what competitors don't show you.
- **Compare view** — two trace IDs side-by-side. Spot the diff between a good run and a bad one without `diff -u`'ing JSON.

The dashboard binds to `127.0.0.1` by default and refuses non-loopback hosts unless `PEEKR_ALLOW_REMOTE=1` is set. Data stays on your machine.

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
| **Feedback + fine-tuning export** | `peekr.feedback(trace_id, rating="good")` |
| **A/B experiments** | `@peekr.experiment(variants=["control", "test"])` |
| **Trace replay** | `peekr replay <trace_id>` |

## Supported clients

| Provider | SDK | Install |
|---|---|---|
| **OpenAI** | `openai` | `pip install "peekr[openai]"` |
| **Anthropic** | `anthropic` | `pip install "peekr[anthropic]"` |
| **AWS Bedrock** | `boto3` | `pip install "peekr[bedrock]"` |

All three auto-instrument with the same two lines — `peekr.instrument()` detects whichever SDKs are installed and patches them. Streaming is supported for all three.

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

## Installation

```bash
pip install peekr                   # base
pip install "peekr[openai]"         # with OpenAI
pip install "peekr[anthropic]"      # with Anthropic
pip install "peekr[bedrock]"        # with AWS Bedrock
pip install "peekr[serve]"          # local web dashboard (Flask)
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
