# peekai

[![PyPI](https://img.shields.io/pypi/v/peekai)](https://pypi.org/project/peekai/)
[![CI](https://github.com/ashwanijha04/peekai/actions/workflows/ci.yml/badge.svg)](https://github.com/ashwanijha04/peekai/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

Zero-config observability for AI agents. Auto-instruments OpenAI and Anthropic SDKs — no code changes needed.

```
pip install peekai
```

---

## Quickstart

```python
import peekai
peekai.instrument()

# Your existing agent code — zero changes
import openai
openai.chat.completions.create(model="gpt-4o", messages=[...])
```

Every LLM call is automatically captured. View your traces:

```bash
peekai view traces.jsonl
```

```
Trace a3f2b1c0  1243ms  891tok
────────────────────────────────────────────────
agent.run  1243ms
   └─ openai.chat.completions [gpt-4o]  821ms  312tok
   └─ tool.search_web  12ms
   └─ openai.chat.completions [gpt-4o]  410ms  579tok
```

---

## Installation

```bash
# Base (no LLM SDK required)
pip install peekai

# With OpenAI
pip install "peekai[openai]"

# With Anthropic
pip install "peekai[anthropic]"

# Both
pip install "peekai[all]"
```

---

## Usage

### Auto-instrument LLM SDKs

```python
import peekai

peekai.instrument()
# That's it. All OpenAI and Anthropic calls are now traced.
```

Options:

```python
peekai.instrument(
    console=True,               # print spans as they happen (default: True)
    jsonl_path="traces.jsonl",  # write to file (default: traces.jsonl)
    jsonl_path=None,            # disable file output
)
```

### Trace your own functions

```python
from peekai import trace

@trace
def search_web(query: str) -> list[str]:
    ...

@trace(name="tool.calculator")
def calculate(expression: str) -> float:
    ...

# Async works too
@trace
async def fetch_data(url: str) -> dict:
    ...
```

Decorated functions automatically become child spans of whatever called them.

### Capture or hide inputs/outputs

```python
@trace                        # captures args and return value by default
def search_web(query): ...

@trace(capture_io=False)      # opt out for sensitive data
def get_api_key(): ...
```

### Manual spans

For cases where a decorator doesn't fit:

```python
from peekai import start_span, end_span

span, token = start_span("my.operation")
span.attributes["custom_key"] = "custom_value"
try:
    do_work()
    span.status = "ok"
except Exception as e:
    span.status = "error"
    span.attributes["error"] = str(e)
    raise
finally:
    end_span(span, token)
```

### Viewing traces

```bash
# Basic tree view
peekai view traces.jsonl

# Show inputs and outputs
peekai view --io traces.jsonl
```

---

## What gets captured

| Field | Description |
|---|---|
| `name` | Span name (function name or custom) |
| `duration_ms` | Wall-clock time |
| `status` | `ok` or `error` |
| `model` | LLM model name (auto) |
| `tokens_input` | Prompt tokens (auto) |
| `tokens_output` | Completion tokens (auto) |
| `tokens_total` | Total tokens (auto) |
| `input` | Serialized function args (truncated) |
| `output` | Serialized return value (truncated) |
| `error` | Exception message if status is `error` |

---

## Custom exporters

```python
from peekai.exporters import add_exporter

class MyExporter:
    def export(self, span):
        # send to your backend
        requests.post("https://my-backend.com/spans", json=span.to_dict())

peekai.instrument()
add_exporter(MyExporter())
```

---

## How it works

`instrument()` monkey-patches the OpenAI and Anthropic SDK methods before your code runs. Python looks up function references at call time, so every subsequent call hits the wrapper instead of the original — with zero changes to your code.

Span context (parent/child relationships) is tracked via Python's `contextvars.ContextVar`, which propagates correctly across `async/await` without any manual passing.

---

## Contributing

```bash
git clone https://github.com/ashwanijha04/peekai
cd peekai
pip install -e ".[dev]"
pytest
```

PRs welcome. Open an issue first for large changes.

---

## License

MIT
