# agenttracer

Zero-config observability for AI agents. Auto-instruments OpenAI and Anthropic SDKs — no code changes needed.

## Install

```bash
pip install agenttracer
```

## Usage

```python
import agent_tracer
agent_tracer.instrument()

# Your existing code — zero changes
import openai
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}]
)
```

Every LLM call is automatically captured. View traces:

```bash
agenttracer view traces.jsonl
```

Output:
```
Trace a3f2b1c0
────────────────────────────────────────
openai.chat.completions [gpt-4o] 842ms 312 tokens
```

## Options

```python
agent_tracer.instrument(
    console=True,          # print spans as they happen (default: True)
    jsonl_path="traces.jsonl",  # write to file (default: traces.jsonl)
)
```

## What gets captured

- Model name
- Input/output token counts
- Latency
- Errors
- Parent-child span relationships across tool calls
