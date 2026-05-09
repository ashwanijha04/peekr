# peekr

[![PyPI](https://img.shields.io/pypi/v/peekr)](https://pypi.org/project/peekr/)
[![CI](https://github.com/ashwanijha04/peekr/actions/workflows/ci.yml/badge.svg)](https://github.com/ashwanijha04/peekr/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

Zero-config observability for AI agents. Auto-instruments OpenAI and Anthropic SDKs — no code changes needed.

```
pip install peekr
```

---

## Quickstart

```python
import peekr
peekr.instrument()

# Your existing agent code — zero changes
import openai
openai.chat.completions.create(model="gpt-4o", messages=[...])
```

Every LLM call is automatically captured. View your traces:

```bash
peekr view traces.jsonl
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
pip install peekr

# With OpenAI
pip install "peekr[openai]"

# With Anthropic
pip install "peekr[anthropic]"

# Both
pip install "peekr[all]"
```

---

## Usage

### Auto-instrument LLM SDKs

```python
import peekr

peekr.instrument()
# That's it. All OpenAI and Anthropic calls are now traced.
```

Options:

```python
peekr.instrument(
    console=True,               # print spans as they happen (default: True)
    jsonl_path="traces.jsonl",  # write to file (default: traces.jsonl)
    jsonl_path=None,            # disable file output
)
```

### Trace your own functions

```python
from peekr import trace

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
from peekr import start_span, end_span

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
peekr view traces.jsonl

# Show inputs and outputs
peekr view --io traces.jsonl
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
from peekr.exporters import add_exporter

class MyExporter:
    def export(self, span):
        # send to your backend
        requests.post("https://my-backend.com/spans", json=span.to_dict())

peekr.instrument()
add_exporter(MyExporter())
```

---

## How it works

`instrument()` monkey-patches the OpenAI and Anthropic SDK methods before your code runs. Python looks up function references at call time, so every subsequent call hits the wrapper instead of the original — with zero changes to your code.

Span context (parent/child relationships) is tracked via Python's `contextvars.ContextVar`, which propagates correctly across `async/await` without any manual passing.

---

## Contributing

```bash
git clone https://github.com/ashwanijha04/peekr
cd peekr
pip install -e ".[dev]"
pytest
```

PRs welcome. Open an issue first for large changes.

---

## License

MIT
