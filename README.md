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

When your agent gives a wrong answer, you have no idea why. Was the prompt malformed? Did a tool return bad data? Did the LLM hallucinate? Without observability, you're guessing.

Peekr records every LLM call, every tool invocation, every token, and every error — as a tree you can inspect. Two lines to add, no backend required.

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
peekr view traces.jsonl          # tree view
peekr view --io traces.jsonl     # include inputs and outputs
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

## What it helps you debug

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

## Installation

```bash
pip install peekr                   # base
pip install "peekr[openai]"         # with OpenAI
pip install "peekr[anthropic]"      # with Anthropic
pip install "peekr[all]"            # both
```

---

## Options

```python
peekr.instrument(
    console=True,                # print spans live (default: True)
    jsonl_path="traces.jsonl",   # write to file (default: "traces.jsonl")
    jsonl_path=None,             # disable file output
)
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
