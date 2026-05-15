# @peekr/sdk — TypeScript SDK for peekr

Zero-config observability for AI agents, in TypeScript. Auto-instruments the OpenAI and Anthropic Node SDKs and writes one JSON-Lines span per call to disk. The Python [`peekr`](https://pypi.org/project/peekr/) CLI and HTML dashboard read the same JSONL — the schema is the cross-runtime contract.

```ts
import { instrument, wrap, trace, withSession } from "@peekr/sdk";
import OpenAI from "openai";

instrument({ jsonlPath: "./traces.jsonl" });
const openai = wrap(new OpenAI());

const search = trace(async (query: string) => fetchResults(query));

await withSession(
  { user_id: "alice", tenant_id: "acme", retention_class: "long" },
  async () => {
    await search("climate policy");
    await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: "Summarise the docs above" }],
    });
  },
);
```

Then on any machine with the Python CLI:

```bash
pip install peekr
peekr view --io traces.jsonl
peekr dashboard traces.jsonl -o report.html
```

## What it captures

For every wrapped LLM call and every traced function:

- **Span tree** — parent/child relationships propagate across `async/await` via `AsyncLocalStorage`. Concurrent requests stay isolated.
- **Inputs / outputs** — JSON-stringified, truncated to 1000 chars by default.
- **Tokens** — `tokens_input`, `tokens_output`, `tokens_total` from the LLM SDK response. Streaming is supported (we inject `stream_options.include_usage: true` so OpenAI emits a final usage chunk).
- **Status + error** — `status: "error"` and `attributes.error` on rejection.
- **Session** — `attributes.user_id` and `attributes.session_id` populated when inside `withSession(...)`.
- **Multi-tenant** — `tenant_id` (customer org) and `retention_class` (storage-tier hint) are first-class top-level fields on every span, distinct from `user_id` (the end-user). Set via `withSession({ tenant_id, retention_class })`, `instrument({ tenant_id, retention_class })`, or env vars `PEEKR_TENANT_ID` / `PEEKR_RETENTION_CLASS`.
- **Endpoint** — set it yourself: `span.attributes["endpoint"] = req.path`. The Python dashboard's channel-drift heatmap groups by it.

## Public API

| Function | Purpose |
|---|---|
| `instrument(opts?)` | Register storage exporters. Call once at startup. |
| `wrap(client)` | Wrap an OpenAI or Anthropic client; returns a Proxy. Auto-detects which one. |
| `wrapOpenAI(client)` / `wrapAnthropic(client)` | Explicit wrappers if you prefer not to rely on auto-detect. |
| `trace(fn)` / `trace(opts, fn)` | Wrap any sync/async function so each call becomes a span. |
| `withSpan(name, fn)` | Run `fn(span)` inside a new child-span scope. The span auto-finishes when `fn` returns (or its promise resolves/rejects). |
| `withSession({ user_id, session_id, tenant_id, retention_class }, fn)` | Attach identity + retention to every span created inside `fn`. |
| `HTTPExporter({ endpoint, apiKey })` | Reserved exporter for Peekr Cloud. Stable signature; throws until cloud GA. [Waitlist](https://github.com/ashwanijha04/peekr/discussions). |
| `getCurrentSpan()` | The innermost open span, or `null`. Useful for tagging extra attributes. |
| `createDetachedSpan(name)` / `finishDetachedSpan(span)` | Low-level escape hatch for spans that outlive a callback (used internally for streaming responses). |
| `JSONLExporter(path)` / `ConsoleExporter()` / `addExporter(e)` | Default exporters + custom ones. |

## Why a separate SDK and not "just" Python?

Browsers, Next.js API routes, Vercel/Cloudflare Workers, and Node agent stacks (LangChain.js, Mastra, agent-graph frameworks) all run TypeScript. The Python `peekr` package can only see Python processes — monkey-patching is per-runtime by nature.

**What's in this package:** the irreducibly-per-language piece — the SDK wrapper code that intercepts OpenAI/Anthropic JS clients and writes spans to disk.

**What stays in Python and is shared:** the JSONL schema, the SQLite store, the evaluators (Hallucination, RAGAS-style claim decomposition, CitationAccuracy, custom Rubric), the diagnostic engine, the dashboard generator, and the CLI tools (`view`, `cost`, `replay`, `dashboard`). They read the JSONL this SDK writes, indistinguishably from Python-produced traces.

This means you write ~500 lines of TS bridge code, not ~5000 lines re-implementing the entire toolchain.

## Schema (cross-runtime contract)

Every span written to JSONL has these fields, identical to [peekr's Python schema](https://github.com/ashwanijha04/peekr#span-fields):

```jsonc
{
  "name": "openai.chat.completions",
  "trace_id": "bce56f3f665342138682d869f7e3e985",
  "span_id":  "7bb3d386c02e471697c8e5f378ad1c2f",
  "parent_id": "a3957e63a52f42b3a8292572d6c4b78d",
  "start_time": 1778871223.116,
  "end_time":   1778871223.274,
  "duration_ms": 158.0,
  "status": "ok",
  "tenant_id": "acme",
  "retention_class": "long",
  "attributes": {
    "model": "gpt-4o-mini",
    "input":  "[{\"role\":\"user\",\"content\":\"…\"}]",
    "output": "…",
    "tokens_input":  30, "tokens_output": 88, "tokens_total": 118,
    "user_id":    "alice",
    "session_id": "5cf889a6d1784c13acb033b8d5671762",
    "endpoint":   "/api/qa"
  }
}
```

If you write spans with this shape via any means — TS, Python, or a custom emitter — the Python dashboard renders them.

## Install

```bash
npm install @peekr/sdk
# plus your LLM SDK of choice:
npm install openai
npm install @anthropic-ai/sdk
```

`openai` and `@anthropic-ai/sdk` are **optional peer dependencies** — the SDK works with whichever you install (or both, or neither, if you only use `trace()`).

## Development

```bash
npm install
npm run typecheck   # tsc + tsc -p tsconfig.test.json
npm test            # node --test on the compiled tests
npm run build       # emits dist/ for publishing
```

Tests use the built-in `node:test` runner — no test framework dependency.

## Cross-runtime demo

```bash
npx tsx examples/cross_runtime.ts   # writes ./traces.jsonl from a TS workload
pip install peekr
peekr dashboard traces.jsonl -o dashboard.html
open dashboard.html
```

The dashboard you'll see is identical to the one you'd get from Python-produced traces.

## License

MIT. © Ashwani Jha.
