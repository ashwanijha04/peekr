# peekr

[![PyPI](https://img.shields.io/pypi/v/peekr)](https://pypi.org/project/peekr/)
[![CI](https://github.com/ashwanijha04/peekr/actions/workflows/ci.yml/badge.svg)](https://github.com/ashwanijha04/peekr/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

**Agents are black boxes. Peekr makes them transparent.**

When your agent gives a wrong answer, you have no idea why. Was the prompt wrong? Did a tool return bad data? Did the LLM hallucinate? Without observability, you're guessing.

Peekr records every LLM call, every tool invocation, every token, and every error — as a tree you can inspect. Two lines to install, no backend required.

```bash
pip install peekr
```

```python
import peekr
peekr.instrument()

# Your existing agent code — zero changes needed
```

---

## What it solves

### "My agent gave the wrong answer"

Open the trace. See the exact prompt that was sent — not what you think was sent, what was *actually* sent. Peekr captures every message, so you can see if your context-builder passed stale data, your system prompt got truncated, or a tool returned something malformed before it hit the LLM.

```
peekr view --io traces.jsonl

Trace 3a9f1c2d  2100ms  4821tok
────────────────────────────────────────────────
agent.run  2100ms
   └─ tool.fetch_user  12ms
         in:  {"args": [42], "kwargs": {}}
         out: null                          ← user not found, but agent didn't check
   └─ openai.chat.completions [gpt-4o]  2088ms  4821tok
         in:  [{"role": "system", "content": "User profile: null..."}]
```

The bug is in `fetch_user`, not the LLM.

---

### "My agent is too slow"

The trace shows exactly where the time went:

```
agent.run  4300ms
   └─ tool.search_web   3800ms   ← 88% of total time
   └─ openai.chat        490ms
```

Without this you'd assume the LLM is slow and start swapping models. The real fix is caching or parallelizing the search tool.

---

### "My API bill is too high"

Every trace records token counts. Run a few traces and look for the pattern:

```
Trace 1:  18,432 tokens
Trace 2:  21,104 tokens
Trace 3:  24,891 tokens   ← growing every turn
```

Tokens growing each run means the agent appends full conversation history on every call. Summarize after 5 turns — typically cuts costs 60–80%.

---

### "It works locally but fails in production"

The trace shows what your tools actually returned, not what you think they returned:

```
tool.fetch_inventory  8ms
   in:  {"sku": "ABC-123"}
   out: []                  ← empty in prod, populated locally
```

The bug is in your data pipeline. The agent logic is fine.

---

## Quickstart

```python
import peekr
peekr.instrument()

import openai

# Every call is now traced automatically
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Summarize this doc"}]
)
```

View the trace:

```bash
peekr view traces.jsonl          # tree view
peekr view --io traces.jsonl     # include inputs and outputs
```

---

## Tracing your own tools

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

Decorated functions nest automatically under whatever called them:

```
agent.run  843ms
   └─ tool.search_web  210ms
   └─ openai.chat [gpt-4o]  633ms  891tok
```

---

## Hiding sensitive data

```python
@trace(capture_io=False)   # latency and status still recorded, args/output not
def fetch_api_key(user_id: int) -> str:
    ...
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
    console=True,                # print spans live as they happen (default: True)
    jsonl_path="traces.jsonl",   # write to file (default: traces.jsonl)
)

peekr.instrument(jsonl_path=None)  # console only
```

---

## Send to your own backend

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

`peekr.instrument()` replaces the OpenAI and Anthropic SDK methods with thin wrappers before your code runs. Python resolves function names at call time, so every subsequent call hits the wrapper — with zero changes to your code.

Parent/child relationships between spans are tracked via Python's `contextvars.ContextVar`, which propagates correctly across `async/await` without manual threading.

---

## Contributing

```bash
git clone https://github.com/ashwanijha04/peekr
cd peekr
pip install -e ".[dev]"
pytest
```

Open an issue before submitting large changes.

---

MIT License
