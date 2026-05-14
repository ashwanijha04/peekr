# Peekr Competitive Feature-Gap Matrix

_Last updated: 2026-05-14 — first pass. Owner: Scope (competitive-intelligence agent). Cadence: weekly, or when a major peekr feature ships._

This document is the canonical "where does peekr stand vs. the LLM-observability / evals / guardrails universe" reference. It is opinionated, sourced, and refreshed by diff (not rebuilt).

**Legend.** ✅ shipped · 🟡 partial / behind a flag / weaker than competitor · ❌ missing · n/a explicit non-goal · ❓ unverified, see *verify* note.

**Peekr state used here.** v0.2.0 is what's on `main` and PyPI today. v0.3.0 (guardrails, hallucination evals, OpenTelemetry export, Gemini auto-instrument) is in **PR #1 / ASH-26, in review**. Rows for v0.3 features are marked `✅ (v0.3 in PR)` so the doc stays honest about what users can install right now versus what's about to land.

---

## 1. Full matrix — full-stack tracing + eval + dashboard competitors

This is the head-to-head against the products users compare peekr to most directly. Competitors: **Langfuse**, **Langsmith**, **Arize Phoenix**, **Braintrust**, **W&B Weave**, **Honeyhive**, **Comet Opik**, **Galileo**.

### 1.1 Capture & tracing

| Feature | peekr | Langfuse | Langsmith | Phoenix | Braintrust | Weave | Honeyhive | Opik | Galileo |
|---|---|---|---|---|---|---|---|---|---|
| OpenAI auto-instrument | ✅ [^p1] | ✅ [^lf1] | ✅ [^ls1] | ✅ [^px1] | ✅ [^bt1] | ✅ [^wv1] | ✅ [^hh1] | ✅ [^op1] | ✅ [^gl1] |
| Anthropic auto-instrument | ✅ [^p1] | ✅ [^lf1] | ✅ [^ls1] | ✅ [^px1] | ✅ [^bt1] | ✅ [^wv1] | ✅ [^hh1] | ✅ [^op1] | ✅ [^gl1] |
| Bedrock auto-instrument | ✅ [^p1] | 🟡 via OTel [^lf2] | ❌ | ✅ [^px1] | 🟡 [^bt1] | 🟡 [^wv1] | ❓ verify | ❓ verify | 🟡 [^gl1] |
| Google Gemini / Vertex auto-instrument | ✅ (v0.3 in PR) [^p2] | ✅ [^lf1] | ✅ [^ls1] | ✅ [^px1] | ✅ [^bt1] | ✅ [^wv1] | ✅ [^hh1] | ✅ [^op1] | ✅ [^gl1] |
| Framework auto-instrument (LangChain / LlamaIndex / CrewAI) | ❌ (ASH-16 open) | ✅ [^lf3] | ✅ native [^ls1] | ✅ [^px2] | ✅ [^bt2] | ✅ [^wv2] | ✅ [^hh2] | ✅ [^op2] | ✅ [^gl2] |
| MCP (Model Context Protocol) tracing | ❌ (ASH-24 open) | 🟡 via OTel [^lf2] | ❓ verify | 🟡 via OpenInference [^px3] | ❓ verify | ❓ verify | ❓ verify | ❓ verify | ❓ verify |
| Streaming capture (tokens + content) | ✅ [^p1] | ✅ [^lf1] | ✅ [^ls1] | ✅ [^px1] | ✅ [^bt1] | ✅ [^wv1] | ✅ [^hh1] | ✅ [^op1] | ✅ [^gl1] |
| Multimodal capture (images, audio) | ❌ (ASH-23 open) | ✅ [^lf4] | ✅ [^ls2] | ✅ [^px4] | ✅ [^bt3] | ✅ [^wv3] | 🟡 [^hh3] | ❓ verify | 🟡 [^gl3] |
| Nested span tree / agent graph | ✅ [^p1] | ✅ [^lf1] | ✅ [^ls1] | ✅ [^px1] | ✅ [^bt1] | ✅ [^wv1] | ✅ [^hh1] | ✅ [^op1] | ✅ [^gl1] |
| Async / contextvar propagation | ✅ [^p3] | ✅ [^lf1] | ✅ [^ls1] | ✅ [^px1] | ✅ [^bt1] | ✅ [^wv1] | ❓ verify | ❓ verify | ❓ verify |
| User-defined `@trace` decorator | ✅ [^p1] | ✅ [^lf5] | ✅ [^ls3] | ✅ [^px5] | ✅ [^bt4] | ✅ [^wv1] | ✅ [^hh1] | ✅ [^op3] | ✅ [^gl1] |
| Sampling (head / tail / by-session) | ❌ (ASH-23 open) | ✅ head+tail [^lf6] | 🟡 sampling rate [^ls4] | ✅ [^px6] | 🟡 [^bt5] | ❓ verify | ❓ verify | ❓ verify | ✅ [^gl4] |
| Custom span attributes / tags | ✅ [^p1] | ✅ [^lf1] | ✅ [^ls1] | ✅ [^px1] | ✅ [^bt1] | ✅ [^wv1] | ✅ [^hh1] | ✅ [^op1] | ✅ [^gl1] |

### 1.2 Storage, export, deployment

| Feature | peekr | Langfuse | Langsmith | Phoenix | Braintrust | Weave | Honeyhive | Opik | Galileo |
|---|---|---|---|---|---|---|---|---|---|
| Zero-backend / no signup | ✅ [^p1] | ❌ self-host needs Postgres+Redis+ClickHouse [^lf7] | ❌ cloud only [^ls1] | 🟡 OSS pip but Phoenix server needed [^px7] | ❌ cloud only [^bt6] | ❌ wandb account [^wv1] | ❌ cloud [^hh1] | 🟡 OSS self-host [^op4] | ❌ cloud [^gl1] |
| Local JSONL file output | ✅ [^p1] | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Local SQLite (queryable) | ✅ [^p4] | ❌ (Postgres) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Hosted SaaS | n/a (deliberate non-goal) | ✅ [^lf7] | ✅ [^ls1] | ✅ Arize cloud [^px8] | ✅ [^bt6] | ✅ [^wv1] | ✅ [^hh1] | ✅ Comet cloud [^op4] | ✅ [^gl1] |
| Self-host OSS | n/a (the library *is* the install) | ✅ [^lf7] | ❌ | ✅ [^px7] | ❌ | ❌ | ❌ | ✅ [^op4] | ❌ |
| OpenTelemetry export | ✅ (v0.3 in PR) [^p2] | ✅ [^lf2] | ✅ [^ls5] | ✅ native [^px9] | 🟡 [^bt7] | 🟡 [^wv4] | ✅ [^hh4] | ✅ [^op5] | ✅ [^gl5] |
| Custom exporter API | ✅ [^p1] | 🟡 via SDK [^lf1] | 🟡 [^ls1] | ✅ [^px9] | 🟡 [^bt1] | 🟡 [^wv1] | 🟡 [^hh1] | ✅ [^op5] | 🟡 [^gl1] |
| OpenInference semantic conventions | ✅ (v0.3 in PR) [^p2] | 🟡 [^lf2] | ❌ proprietary | ✅ leads the spec [^px9] | ❌ | ❌ | 🟡 [^hh4] | ✅ [^op5] | 🟡 [^gl5] |

### 1.3 Cost, latency, alerts

| Feature | peekr | Langfuse | Langsmith | Phoenix | Braintrust | Weave | Honeyhive | Opik | Galileo |
|---|---|---|---|---|---|---|---|---|---|
| Per-call cost computation | ✅ CLI [^p5] | ✅ UI [^lf8] | ✅ UI [^ls6] | ✅ UI [^px10] | ✅ UI [^bt8] | ✅ UI [^wv5] | ✅ UI [^hh5] | ✅ UI [^op6] | ✅ UI [^gl6] |
| Top-N hot-spot ranking | ✅ `peekr cost` [^p5] | 🟡 sort by cost [^lf8] | 🟡 [^ls6] | 🟡 [^px10] | 🟡 [^bt8] | 🟡 [^wv5] | 🟡 [^hh5] | 🟡 [^op6] | 🟡 [^gl6] |
| Per-user / per-session cost rollup | 🟡 (session_id captured, no rollup CLI) | ✅ [^lf8] | ✅ [^ls6] | ✅ [^px10] | ✅ [^bt8] | ✅ [^wv5] | ✅ [^hh5] | ✅ [^op6] | ✅ [^gl6] |
| Cost-spike / anomaly alerts | ✅ [^p6] | ✅ [^lf9] | 🟡 [^ls7] | ✅ [^px11] | 🟡 [^bt8] | ❓ verify | ✅ [^hh6] | ❓ verify | ✅ [^gl7] |
| Cost budgets / hard caps | ❌ | 🟡 [^lf9] | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Slack / PagerDuty / webhook alert sinks | ❌ (ASH-25 open; stderr only today) | ✅ [^lf9] | ✅ [^ls7] | ✅ [^px11] | ✅ [^bt8] | ❓ verify | ✅ [^hh6] | ✅ [^op7] | ✅ [^gl7] |

### 1.4 Evals

| Feature | peekr | Langfuse | Langsmith | Phoenix | Braintrust | Weave | Honeyhive | Opik | Galileo |
|---|---|---|---|---|---|---|---|---|---|
| Custom rubric eval (LLM-as-judge) | ✅ [^p7] | ✅ [^lf10] | ✅ [^ls8] | ✅ [^px12] | ✅ [^bt9] | ✅ [^wv6] | ✅ [^hh7] | ✅ [^op8] | ✅ [^gl8] |
| Faithfulness / hallucination eval | ✅ (v0.3 in PR) [^p2] | ✅ [^lf10] | ✅ [^ls8] | ✅ [^px12] | ✅ [^bt9] | ✅ [^wv6] | ✅ [^hh7] | ✅ [^op8] | ✅ marquee [^gl8] |
| Answer-relevance / context-relevance | ✅ (v0.3 in PR) [^p2] | ✅ [^lf10] | ✅ [^ls8] | ✅ [^px12] | ✅ [^bt9] | ✅ [^wv6] | ✅ [^hh7] | ✅ [^op8] | ✅ [^gl8] |
| Custom evaluator class API | ✅ [^p7] | ✅ [^lf10] | ✅ [^ls8] | ✅ [^px12] | ✅ [^bt9] | ✅ [^wv6] | ✅ [^hh7] | ✅ [^op8] | ✅ [^gl8] |
| Offline eval runner (dataset replay) | ❌ (ASH-18 open) | ✅ [^lf11] | ✅ [^ls9] | ✅ [^px13] | ✅ marquee [^bt10] | ✅ [^wv7] | ✅ [^hh8] | ✅ [^op9] | ✅ marquee [^gl9] |
| Score-trace diff / CI gate | ❌ (ASH-18 open) | ✅ [^lf11] | ✅ [^ls9] | ✅ [^px13] | ✅ [^bt10] | ✅ [^wv7] | ✅ [^hh8] | ✅ [^op9] | ✅ [^gl9] |
| Human annotation queue / labelling UI | ❌ (ASH-28 open) | ✅ marquee [^lf12] | ✅ [^ls10] | ✅ [^px14] | ✅ marquee [^bt11] | ✅ [^wv8] | ✅ [^hh9] | ✅ [^op10] | ✅ [^gl10] |
| Feedback (👍/👎) capture | ✅ [^p8] | ✅ [^lf12] | ✅ [^ls10] | ✅ [^px14] | ✅ [^bt11] | ✅ [^wv8] | ✅ [^hh9] | ✅ [^op10] | ✅ [^gl10] |
| Fine-tuning data export | ✅ OpenAI-FT format [^p8] | ✅ [^lf12] | ✅ [^ls10] | 🟡 [^px14] | ✅ [^bt11] | 🟡 [^wv8] | ❓ verify | ❓ verify | ❓ verify |

### 1.5 Other

| Feature | peekr | Langfuse | Langsmith | Phoenix | Braintrust | Weave | Honeyhive | Opik | Galileo |
|---|---|---|---|---|---|---|---|---|---|
| Trace replay (re-run a saved trace) | ✅ [^p9] | 🟡 [^lf13] | ✅ [^ls11] | 🟡 [^px15] | ✅ [^bt12] | ❓ verify | 🟡 [^hh10] | ❓ verify | ❓ verify |
| A/B variant tagging | ✅ [^p10] | ✅ [^lf14] | ✅ [^ls12] | ✅ [^px16] | ✅ marquee [^bt13] | 🟡 [^wv9] | ✅ [^hh11] | ❓ verify | ✅ [^gl11] |
| Prompt registry / versioning | ❌ (ASH-21 open) | ✅ [^lf15] | ✅ [^ls13] | ✅ [^px17] | ✅ [^bt14] | 🟡 [^wv10] | ✅ [^hh12] | ✅ [^op11] | ❓ verify |
| Local CLI viewer | ✅ tree + IO + cost [^p1] [^p5] | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Web dashboard | ❌ (ASH-17 open) | ✅ [^lf7] | ✅ [^ls1] | ✅ [^px7] | ✅ [^bt6] | ✅ [^wv1] | ✅ [^hh1] | ✅ [^op4] | ✅ [^gl1] |

---

## 2. Gateway / proxy competitors

These are not full observability platforms — they are drop-in API proxies that add observability as a by-product of being in the request path. **Helicone**, **Portkey**, **OpenRouter**, **Vercel AI Gateway**, **LiteLLM proxy**.

| Feature | peekr | Helicone | Portkey | OpenRouter | Vercel AI GW | LiteLLM |
|---|---|---|---|---|---|---|
| Drop-in proxy mode (set `OPENAI_BASE_URL`) | ❌ (ASH-20 open) | ✅ marquee [^he1] | ✅ marquee [^pk1] | ✅ marquee [^or1] | ✅ marquee [^vag1] | ✅ [^ll1] |
| Multi-provider routing / fallback | ❌ (ASH-20 open) | ✅ [^he2] | ✅ marquee [^pk2] | ✅ marquee [^or1] | ✅ [^vag1] | ✅ marquee [^ll2] |
| Response cache | ❌ (ASH-29 open) | ✅ marquee [^he3] | ✅ [^pk3] | ❌ | ✅ [^vag2] | ✅ [^ll3] |
| Semantic / embedding cache | ❌ (ASH-29 open) | ✅ marquee [^he3] | ✅ [^pk3] | ❌ | 🟡 [^vag2] | 🟡 [^ll3] |
| Retry on 429 / rate-limit handling | ❌ | ✅ [^he2] | ✅ [^pk2] | ✅ [^or1] | ✅ [^vag1] | ✅ [^ll2] |
| Per-key cost budgets | ❌ | ✅ [^he4] | ✅ [^pk4] | ✅ [^or1] | ❓ verify | ✅ [^ll4] |
| Library mode (no proxy required) | ✅ [^p1] | 🟡 [^he1] | ✅ [^pk1] | n/a (proxy only) | n/a | ✅ [^ll1] |
| Local-first (no SaaS) | ✅ | 🟡 OSS self-host [^he5] | 🟡 OSS gateway [^pk5] | ❌ | ❌ | ✅ [^ll5] |

---

## 3. OTel-native competitors

**Traceloop / OpenLLMetry** (the library) and **OpenInference** (the spec, driven by Arize).

| Feature | peekr | Traceloop / OpenLLMetry | OpenInference (spec) |
|---|---|---|---|
| OpenInference semantic conventions | ✅ (v0.3 in PR) [^p2] | 🟡 own spec, OpenLLMetry [^tl1] | ✅ canonical [^oi1] |
| OTLP / OpenTelemetry export | ✅ (v0.3 in PR) [^p2] | ✅ marquee [^tl1] | n/a (spec) |
| Multi-framework instrumentation (LangChain, LlamaIndex, Haystack, …) | ❌ (ASH-16 open) | ✅ marquee [^tl2] | n/a |
| Library + standalone collector | ✅ library | ✅ both [^tl1] | n/a |
| Ships to Datadog / Honeycomb / Grafana | ✅ via OTel exporter (v0.3) | ✅ [^tl1] | n/a |

---

## 4. Prompt-management competitors

**PromptLayer**, **Promptfoo**, **Mirascope**, **Helicone Prompts**.

| Feature | peekr | PromptLayer | Promptfoo | Mirascope | Helicone Prompts |
|---|---|---|---|---|---|
| Prompt registry (name + version) | ❌ (ASH-21 open) | ✅ marquee [^pl1] | 🟡 [^pf1] | ✅ [^mc1] | ✅ [^he6] |
| Prompt labels / staging / rollout | ❌ (ASH-21 open) | ✅ [^pl1] | ❌ | 🟡 [^mc1] | ✅ [^he6] |
| Prompt diff / history | ❌ (ASH-21 open) | ✅ [^pl1] | 🟡 [^pf1] | 🟡 [^mc1] | ✅ [^he6] |
| Prompt-test / batch eval against suite | 🟡 v0.3 evaluators + (ASH-18 open for runner) | 🟡 [^pl1] | ✅ marquee [^pf2] | 🟡 [^mc1] | 🟡 [^he6] |
| Trace attribution back to prompt id/version | ❌ (ASH-21 open) | ✅ [^pl1] | n/a | ✅ [^mc1] | ✅ [^he6] |

---

## 5. Guardrails / safety competitors

**Guardrails AI**, **NeMo Guardrails**, **Lakera**, **Patronus**, **Protect AI**.

| Feature | peekr | Guardrails AI | NeMo Guardrails | Lakera | Patronus | Protect AI |
|---|---|---|---|---|---|---|
| PII detection | ✅ (v0.3 in PR) [^p2] | ✅ [^ga1] | ✅ [^nm1] | ✅ [^lk1] | ✅ [^pt1] | ✅ [^pr1] |
| Secrets detection | ✅ (v0.3 in PR) [^p2] | 🟡 [^ga1] | 🟡 [^nm1] | ❓ verify | ❓ verify | ✅ [^pr1] |
| Heuristic prompt-injection guard | ✅ (v0.3 in PR) [^p2] | ✅ [^ga1] | ✅ [^nm1] | ❌ (model-only) | ❌ | ✅ [^pr1] |
| ML-grade prompt-injection model | ❌ (ASH-19 open) | 🟡 [^ga1] | 🟡 [^nm1] | ✅ marquee [^lk1] | ✅ marquee [^pt1] | ✅ marquee [^pr1] |
| Toxicity / moderation | ✅ heuristic (v0.3 in PR) [^p2] | ✅ [^ga1] | ✅ [^nm1] | ✅ [^lk1] | ✅ [^pt1] | ✅ [^pr1] |
| JSON schema / structured-output validation | ✅ (v0.3 in PR) [^p2] | ✅ marquee [^ga2] | ✅ [^nm1] | ❌ | 🟡 [^pt1] | ❌ |
| Regex / allow / deny / max-length checks | ✅ (v0.3 in PR) [^p2] | ✅ [^ga1] | ✅ [^nm1] | ❌ | ❌ | ❌ |
| Dialogue / programmable conversation flow | ❌ (n/a per peekr scope) | 🟡 [^ga1] | ✅ marquee [^nm2] | ❌ | ❌ | ❌ |
| Action: warn / redact / block | ✅ (v0.3 in PR) [^p2] | ✅ [^ga1] | ✅ [^nm1] | ✅ [^lk1] | ✅ [^pt1] | ✅ [^pr1] |
| Composable check chain | ✅ (v0.3 in PR) [^p2] | ✅ [^ga1] | ✅ [^nm1] | 🟡 [^lk1] | 🟡 [^pt1] | ✅ [^pr1] |
| Embedded inside trace span (single tool) | ✅ marquee [^p2] | ❌ separate library | ❌ separate runtime | ❌ external API | ❌ external API | ❌ external API |

---

## 6. Evals-first competitors

**Braintrust**, **Patronus**, **Ragas**, **DeepEval**, **Galileo**.

| Feature | peekr | Braintrust | Patronus | Ragas | DeepEval | Galileo |
|---|---|---|---|---|---|---|
| Custom rubric eval | ✅ [^p7] | ✅ [^bt9] | ✅ [^pt2] | ✅ [^rg1] | ✅ [^de1] | ✅ [^gl8] |
| Faithfulness / hallucination | ✅ (v0.3 in PR) [^p2] | ✅ [^bt9] | ✅ marquee [^pt2] | ✅ marquee [^rg1] | ✅ marquee [^de1] | ✅ marquee [^gl8] |
| Answer-relevance / context-precision / context-recall | ✅ AR + CR (v0.3 in PR) [^p2] | ✅ [^bt9] | ✅ [^pt2] | ✅ marquee [^rg1] | ✅ [^de1] | ✅ [^gl8] |
| BLEU / ROUGE / heuristic NLP metrics | ❌ | 🟡 [^bt9] | ❓ verify | ✅ [^rg1] | ✅ [^de1] | ✅ [^gl8] |
| Multi-turn conversation eval | ❌ | 🟡 [^bt9] | ✅ [^pt2] | 🟡 [^rg1] | ✅ [^de1] | ✅ [^gl8] |
| Adversarial / red-team test suite | ❌ | 🟡 [^bt9] | ✅ marquee [^pt2] | ❌ | ✅ [^de1] | ✅ [^gl8] |
| Bring-your-own dataset | ❌ (ASH-18 open) | ✅ [^bt10] | ✅ [^pt2] | ✅ [^rg1] | ✅ [^de1] | ✅ [^gl9] |
| CI-mode runner (`pytest` plug, exit code) | ❌ (ASH-18 open) | ✅ [^bt10] | ✅ [^pt2] | ✅ [^rg1] | ✅ marquee [^de1] | ✅ [^gl9] |
| Eval lives inside the same trace as production | ✅ inline EvalExporter [^p7] | 🟡 [^bt9] | ❌ separate API | ❌ standalone | ❌ standalone | 🟡 [^gl8] |

---

## 7. Where peekr leads (moat reasoning)

Two rules: a lead is only credible if it shows up as a row above with peekr=✅ and most competitors ≤ 🟡 — and the *reason* must be structural, not just "we got there first."

1. **Two-line install, zero infrastructure, real persistence.** Every full-stack competitor in §1.2 requires a hosted account or a self-hosted stack (Postgres / Redis / ClickHouse). Peekr is `pip install peekr; peekr.instrument()` and you get JSONL + queryable SQLite on disk. The moat is the deliberate decision to make SQLite the storage backend — SQL queryability without operating a database is genuinely hard to copy without invalidating their hosted business model.

2. **`peekr cost` CLI: opinionated top-N hotspot ranking.** Every competitor will show you a cost-by-operation chart in a web UI; nobody else collapses the question "what should I optimize next?" into a sorted CLI table with a 60/40 cost/latency composite score. This is small in surface area but high in adoption — it's the kind of thing that ends up as a screenshot in a blog post.

3. **Guardrails / evals inside the same span tree as production traffic** (v0.3 in PR). Lakera, Patronus, Protect AI are external APIs you POST to. Guardrails AI and NeMo are separate runtimes. Peekr's check results are stored as attributes on the production span itself, so every block/redact event is already linked to the trace that caused it, no joining required. Most competitors carry an integration tax peekr doesn't.

4. **OpenInference-compliant OTel export with no agent / no collector** (v0.3 in PR). Traceloop's pitch is "use our standalone agent." Peekr writes OpenInference-shaped spans in-process and any OTel pipeline (Datadog, Honeycomb, Tempo, Phoenix, even Langfuse-OTel) consumes them. This makes peekr a credible *front-end* for the entire hosted-platform space rather than a competitor to it — strategically valuable: peekr can ride the OTel standard rather than fight it.

5. **`cProfile`-style framing.** None of the competitors land on this analogy. It's a marketing position more than a technical lead, but it's a sticky one with the Python audience peekr already has.

---

## 8. Where peekr trails — ranked by competitive cost

Ranked by how often a missing feature actually kills a deal in the conversations these tools are used for. High-cost gaps go first.

1. **No web dashboard** — every comparison-table review of peekr will ding this, because "trace UI" is the screenshot the entire category sells on. (ASH-17, in progress.)
2. **No framework auto-instrumentation (LangChain / LlamaIndex / CrewAI)** — most "real" agents in 2026 are framework-based. Peekr only sees them as opaque `@trace` boxes. (ASH-16, in progress.)
3. **No gateway / proxy mode** — Helicone, Portkey, Vercel AI Gateway, OpenRouter all market the one-line `OPENAI_BASE_URL` install as their lead. Peekr being Python-only also locks out JS/TS users. (ASH-20, in progress.)
4. **No offline eval runner / CI gate** — Braintrust, DeepEval, Galileo all build their whole positioning on "fail PRs when eval score drops." (ASH-18, in progress.)
5. **No human annotation queue** — Langfuse, Braintrust, Phoenix, Honeyhive, Comet all ship this; it's how the "feedback → dataset → eval" loop closes. (ASH-28, just filed.)
6. **No response / semantic cache** — gateway competitors lead with this. Peekr captures the traces that *prove* you should cache, but doesn't actually cache. (ASH-29, just filed.)
7. **No prompt management** — Langfuse / PromptLayer / Helicone Prompts / Mirascope all ship this. (ASH-21, in progress.)
8. **Heuristic-only prompt-injection / toxicity guardrails** — fine for v0.3, but Lakera / Patronus / Protect AI all lead on model-backed detectors. (ASH-19, in progress.)
9. **No multimodal capture** — image and audio inputs are increasingly the norm; competitors render them in the trace viewer. (ASH-23, in progress.)
10. **No sampling controls** — high-traffic prod users will hit the cost wall fast. (ASH-23, in progress.)
11. **No MCP tracing** — small TAM today, but a free-differentiator window: nobody else has solid coverage yet. (ASH-24, in progress.)
12. **Alerts only emit to stderr** — Slack/PagerDuty/webhook sinks are table-stakes. (ASH-25, in progress.)
13. **Limited provider coverage** — OpenAI/Anthropic/Bedrock today; Gemini v0.3 in PR. OpenRouter / Mistral / Cohere / Groq / Together still missing. (ASH-22, in progress.)

---

## 9. Deliberate non-goals

Listed so we can say "no" with conviction rather than carry guilt about it. We will reject feature requests that fall in this list unless the underlying assumption changes.

- **Multi-tenant SaaS / hosted backend.** Peekr is a library; "no backend required" is the proposition. Hosted offerings are the entire competitive set's revenue model — we don't try to win there.
- **RBAC, SSO, audit logs, SOC2.** All downstream of "hosted backend." Self-host JSONL/SQLite + your existing OTel pipeline is the answer.
- **Org-wide collaboration features** (shared trace links, comments, team workspaces). Local-first first. If anything, this could come *after* the dashboard ships (ASH-17) as a v2.
- **Telemetry / phone-home.** Hard non-goal — peekr is `pip install` and runs entirely on the user's machine. Differentiator vs. anything in this list that calls home.
- **Building an LLM provider.** We export to providers; we don't compete with them.
- **Programmable dialogue flow runtime** (the NeMo Guardrails "Colang" angle). Out of scope. Peekr is observability + checks, not an agent runtime.
- **Drop-in replacement for OpenTelemetry collectors / vendors.** We *export* to them. Not a moat for us to rebuild.

---

## 10. Emerging features nobody has shipped yet (free-differentiator window)

Spotted in conversations, blog posts, GitHub issues across the competitive set but no product has solid coverage yet. These are where peekr can plant a flag relatively cheaply.

- **First-class MCP tracing.** Anthropic-led MCP is becoming the default tool-protocol for agents. Every competitor's MCP "support" today is "we see the resulting HTTP call." Native `mcp.<tool>` spans with server name, tool, arguments, latency would be a clean first. (ASH-24.)
- **Cost / latency / quality regression CI gate that takes a peekr trace as the input.** Braintrust / DeepEval do dataset-mode CI but none ingest production traces directly. Peekr is uniquely positioned because production traces are the storage format. (Adjacent to ASH-18.)
- **Inline guardrail attributes** — guardrail results stored on the production span instead of in a separate findings store. Already shipped in v0.3 PR. Document and amplify.
- **Local-first agent-graph viewer (Phoenix-style) without the Phoenix server.** Once the dashboard ships, an "agent flow" view (who called whom, with which prompt, returning what) on top of SQLite is a differentiator. (Adjacent to ASH-17.)
- **Span-attached prompt fingerprints with no registry.** Even before full prompt management lands (ASH-21), hashing every prompt sent and storing the hash as `prompt.hash` lets users `GROUP BY prompt.hash` in SQLite and discover "this prompt regressed since Tuesday" — cheap, useful, nobody does it.
- **Semantic cache built on the existing SQLite trace store.** Most competitors run cache on Redis. A peekr-shaped cache that reuses traces.db as the index would be a uniquely-peekr feature. (See §11 — new sub-issue.)

---

## 11. Gaps we should file as new sub-issues

Cross-reference with §1–6 and ASH-15 children (ASH-16 through ASH-25 + ASH-26). Two genuine gaps stand out that are real, concrete, and **not** covered by an existing issue:

1. **Human annotation queue / labelling UI** — 8 of 8 full-stack competitors ship this as a marquee feature. ASH-17 (dashboard) is read-only viewing; ASH-18 (offline runner) is post-curation; nothing in the existing tree owns "human looks at trace and assigns a score / tag that the dataset and eval pipeline can use." **→ filed as ASH-28 (high priority).**

2. **Response / semantic cache** — 4 of the 5 gateway competitors (Helicone, Portkey, Vercel AI Gateway, LiteLLM) lead with caching. ASH-20 covers the gateway shell but does not carve out caching as a feature — and the more interesting peekr-shaped angle is "semantic cache backed by traces.db" (see §10), which deserves its own scope. **→ filed as ASH-29 (high priority).**

Everything else in the gap list (§8) is already covered by ASH-16 through ASH-25.

---

## 12. Changelog

| Date | What moved | Detail |
|---|---|---|
| 2026-05-14 | Doc created (first pass) | Initial matrix: 10 competitor clusters across §1–6, 70+ feature rows. Two new gaps surfaced (human annotation queue, semantic cache) and filed as sub-issues. v0.3.0 (guardrails, hallucination, OTel, Gemini) tracked as "in PR" pending ASH-26 merge. |

---

## Sources

Peekr citations point at files in this repo. Competitor citations point at the most authoritative public source (vendor docs, OSS README, or product page) at time of writing. When a claim is uncertain we mark it `❓ verify` in the matrix and the source line ends with a note about what to check on the next pass.

### Peekr

[^p1]: `peekr/__init__.py`, `peekr/patches/{openai,anthropic,bedrock}_patch.py`, `README.md` — auto-instrument OpenAI / Anthropic / Bedrock, `@trace`, JSONL + console exporters, streaming capture, nested spans.
[^p2]: v0.3.0 features in **PR #1 (ASH-26, in review)** — `peekr/guardrails/`, `peekr/eval/hallucination.py`, `peekr/otel.py`, `peekr/patches/gemini_patch.py`, `peekr/session.py` (`grounding` kwarg). Not on `main` yet.
[^p3]: `peekr/context.py` — `ContextVar` for parent/child propagation; async-safe by construction.
[^p4]: `peekr/exporters.py` — `SQLiteExporter` (WAL mode, indexes on trace_id / name / start_time, `.query()` helper).
[^p5]: `peekr/cli.py:_cmd_cost` — cost breakdown + composite-scored top-10 hotspots.
[^p6]: `peekr/alerts.py` — `ErrorRate`, `CostSpike`, `LatencyP95`, `TokenGrowth`; stderr-only delivery today (ASH-25 to add sinks).
[^p7]: `peekr/eval/__init__.py`, `peekr/eval/rubric.py` — `BaseEvaluator`, `Rubric`, `NotEmpty`, `NoError`; `EvalExporter` runs them inline.
[^p8]: `peekr/feedback.py` — `peekr.feedback(trace_id, rating)` + `export_feedback(format="openai-ft")`.
[^p9]: `peekr/replay.py` — `peekr replay <trace_id>` CLI subcommand.
[^p10]: `peekr/experiment.py` — `@experiment(variants=[...])`; variant key written into span attributes for SQL grouping.

### Langfuse

[^lf1]: <https://langfuse.com/docs/integrations/overview> · <https://langfuse.com/docs/sdk/python>
[^lf2]: <https://langfuse.com/docs/opentelemetry/get-started>
[^lf3]: <https://langfuse.com/docs/integrations/langchain/example-python-langchain> · <https://langfuse.com/docs/integrations/llama-index/get-started>
[^lf4]: <https://langfuse.com/docs/tracing-features/multi-modality>
[^lf5]: <https://langfuse.com/docs/sdk/python/decorators>
[^lf6]: <https://langfuse.com/docs/tracing-features/sampling>
[^lf7]: <https://langfuse.com/self-hosting>
[^lf8]: <https://langfuse.com/docs/model-usage-and-cost>
[^lf9]: <https://langfuse.com/docs/scores/overview> (cost-based scoring and alerts via integrations)
[^lf10]: <https://langfuse.com/docs/scores/model-based-evals>
[^lf11]: <https://langfuse.com/docs/datasets/overview>
[^lf12]: <https://langfuse.com/docs/scores/annotation>
[^lf13]: <https://langfuse.com/docs/playground> — Playground re-runs prompts; not a strict trace replay. Verify on next pass.
[^lf14]: <https://langfuse.com/docs/experimentation>
[^lf15]: <https://langfuse.com/docs/prompts/get-started>

### Langsmith

[^ls1]: <https://docs.smith.langchain.com/>
[^ls2]: <https://docs.smith.langchain.com/observability/how_to_guides/multimodal_content>
[^ls3]: <https://docs.smith.langchain.com/observability/how_to_guides/annotate_code>
[^ls4]: <https://docs.smith.langchain.com/observability/how_to_guides/sample_traces>
[^ls5]: <https://docs.smith.langchain.com/observability/how_to_guides/trace_with_opentelemetry>
[^ls6]: <https://docs.smith.langchain.com/observability/how_to_guides/log_llm_trace>
[^ls7]: <https://docs.smith.langchain.com/observability/how_to_guides/monitor> — verify whether webhook alert sinks are GA or beta.
[^ls8]: <https://docs.smith.langchain.com/evaluation>
[^ls9]: <https://docs.smith.langchain.com/evaluation/how_to_guides/evaluate_with_pytest>
[^ls10]: <https://docs.smith.langchain.com/evaluation/how_to_guides/annotation_queues>
[^ls11]: <https://docs.smith.langchain.com/observability/how_to_guides/playground>
[^ls12]: <https://docs.smith.langchain.com/observability/how_to_guides/experiments>
[^ls13]: <https://docs.smith.langchain.com/prompt_engineering/concepts>

### Arize Phoenix

[^px1]: <https://docs.arize.com/phoenix> · <https://github.com/Arize-ai/phoenix>
[^px2]: <https://docs.arize.com/phoenix/tracing/integrations-tracing>
[^px3]: <https://github.com/Arize-ai/openinference> — MCP coverage tracked under OpenInference instrumentations.
[^px4]: <https://docs.arize.com/phoenix/tracing/how-to-tracing/multimodal>
[^px5]: <https://docs.arize.com/phoenix/tracing/how-to-tracing/setup-tracing/instrument-python>
[^px6]: <https://docs.arize.com/phoenix/tracing/how-to-tracing/sampling>
[^px7]: <https://docs.arize.com/phoenix/deployment/docker>
[^px8]: <https://arize.com/>
[^px9]: <https://github.com/Arize-ai/openinference>
[^px10]: <https://docs.arize.com/phoenix/tracing/how-to-tracing/cost-tracking>
[^px11]: <https://docs.arize.com/phoenix/tracing/how-to-tracing/monitoring>
[^px12]: <https://docs.arize.com/phoenix/evaluation/llm-evals>
[^px13]: <https://docs.arize.com/phoenix/datasets-and-experiments>
[^px14]: <https://docs.arize.com/phoenix/evaluation/how-to-evals/human-in-the-loop>
[^px15]: <https://docs.arize.com/phoenix/tracing/how-to-tracing/replay>
[^px16]: <https://docs.arize.com/phoenix/datasets-and-experiments/how-to-experiments>
[^px17]: <https://docs.arize.com/phoenix/prompt-engineering/overview>

### Braintrust

[^bt1]: <https://www.braintrust.dev/docs/start>
[^bt2]: <https://www.braintrust.dev/docs/guides/tracing>
[^bt3]: <https://www.braintrust.dev/docs/guides/multimodal-data>
[^bt4]: <https://www.braintrust.dev/docs/guides/tracing#instrumenting-your-code>
[^bt5]: <https://www.braintrust.dev/docs/guides/tracing#sampling> — verify wording on next pass.
[^bt6]: <https://www.braintrust.dev/>
[^bt7]: <https://www.braintrust.dev/docs/guides/tracing#opentelemetry> — partial OTel ingest.
[^bt8]: <https://www.braintrust.dev/docs/guides/observability>
[^bt9]: <https://www.braintrust.dev/docs/guides/evals>
[^bt10]: <https://www.braintrust.dev/docs/guides/evals/run> · <https://www.braintrust.dev/docs/guides/ci>
[^bt11]: <https://www.braintrust.dev/docs/guides/human-review>
[^bt12]: <https://www.braintrust.dev/docs/guides/playground>
[^bt13]: <https://www.braintrust.dev/docs/guides/evals#comparing-experiments>
[^bt14]: <https://www.braintrust.dev/docs/guides/prompts>

### Weave (Weights & Biases)

[^wv1]: <https://weave-docs.wandb.ai/>
[^wv2]: <https://weave-docs.wandb.ai/guides/integrations>
[^wv3]: <https://weave-docs.wandb.ai/guides/core-types/media>
[^wv4]: <https://weave-docs.wandb.ai/guides/tracking/tracing> — verify OTel-ingest GA.
[^wv5]: <https://weave-docs.wandb.ai/guides/tracking/costs>
[^wv6]: <https://weave-docs.wandb.ai/guides/evaluation/scorers>
[^wv7]: <https://weave-docs.wandb.ai/guides/evaluation/evaluations>
[^wv8]: <https://weave-docs.wandb.ai/guides/tracking/feedback>
[^wv9]: <https://weave-docs.wandb.ai/guides/evaluation/evaluations> (variant comparison via evaluations) — verify dedicated A/B UI on next pass.
[^wv10]: <https://weave-docs.wandb.ai/guides/core-types/prompts> — verify versioning model on next pass.

### Honeyhive

[^hh1]: <https://docs.honeyhive.ai/introduction/quickstart>
[^hh2]: <https://docs.honeyhive.ai/integrations/overview>
[^hh3]: <https://docs.honeyhive.ai/tracing/multimodal> — verify on next pass.
[^hh4]: <https://docs.honeyhive.ai/integrations/opentelemetry>
[^hh5]: <https://docs.honeyhive.ai/observability/cost-and-latency>
[^hh6]: <https://docs.honeyhive.ai/observability/alerts>
[^hh7]: <https://docs.honeyhive.ai/evaluation/evaluators>
[^hh8]: <https://docs.honeyhive.ai/evaluation/datasets>
[^hh9]: <https://docs.honeyhive.ai/evaluation/human-review>
[^hh10]: <https://docs.honeyhive.ai/playground>
[^hh11]: <https://docs.honeyhive.ai/experiments>
[^hh12]: <https://docs.honeyhive.ai/prompts>

### Comet Opik

[^op1]: <https://www.comet.com/docs/opik/tracing/sdk>
[^op2]: <https://www.comet.com/docs/opik/tracing/integrations>
[^op3]: <https://www.comet.com/docs/opik/tracing/log-traces>
[^op4]: <https://github.com/comet-ml/opik> · <https://www.comet.com/site/products/opik/>
[^op5]: <https://www.comet.com/docs/opik/tracing/opentelemetry>
[^op6]: <https://www.comet.com/docs/opik/tracing/cost-tracking>
[^op7]: <https://www.comet.com/docs/opik/production/alerts>
[^op8]: <https://www.comet.com/docs/opik/evaluation/metrics/overview>
[^op9]: <https://www.comet.com/docs/opik/evaluation/datasets>
[^op10]: <https://www.comet.com/docs/opik/evaluation/human-feedback>
[^op11]: <https://www.comet.com/docs/opik/prompt-engineering/prompt-library>

### Galileo

[^gl1]: <https://docs.galileo.ai/galileo/getting-started>
[^gl2]: <https://docs.galileo.ai/galileo/integrations>
[^gl3]: <https://docs.galileo.ai/galileo/multimodal> — verify on next pass.
[^gl4]: <https://docs.galileo.ai/galileo/observability/sampling> — verify on next pass.
[^gl5]: <https://docs.galileo.ai/galileo/observability/opentelemetry>
[^gl6]: <https://docs.galileo.ai/galileo/observability/cost-tracking>
[^gl7]: <https://docs.galileo.ai/galileo/observability/alerts>
[^gl8]: <https://docs.galileo.ai/galileo/evaluation/metrics> · <https://docs.galileo.ai/galileo/evaluation/hallucination>
[^gl9]: <https://docs.galileo.ai/galileo/evaluation/experiments>
[^gl10]: <https://docs.galileo.ai/galileo/evaluation/human-review>
[^gl11]: <https://docs.galileo.ai/galileo/evaluation/experiments>

### Helicone

[^he1]: <https://docs.helicone.ai/getting-started/quick-start>
[^he2]: <https://docs.helicone.ai/features/advanced-usage/fallbacks>
[^he3]: <https://docs.helicone.ai/features/advanced-usage/caching>
[^he4]: <https://docs.helicone.ai/features/advanced-usage/limits>
[^he5]: <https://docs.helicone.ai/getting-started/self-host/overview>
[^he6]: <https://docs.helicone.ai/features/prompts>

### Portkey

[^pk1]: <https://portkey.ai/docs/integrations/llms>
[^pk2]: <https://portkey.ai/docs/product/ai-gateway/fallbacks>
[^pk3]: <https://portkey.ai/docs/product/ai-gateway/cache-simple-and-semantic>
[^pk4]: <https://portkey.ai/docs/product/ai-gateway/budget-limits>
[^pk5]: <https://github.com/Portkey-AI/gateway>

### OpenRouter

[^or1]: <https://openrouter.ai/docs>

### Vercel AI Gateway

[^vag1]: <https://vercel.com/docs/ai-gateway>
[^vag2]: <https://vercel.com/docs/ai-gateway/caching> — verify whether semantic cache is GA or preview.

### LiteLLM

[^ll1]: <https://docs.litellm.ai/docs/proxy/quick_start>
[^ll2]: <https://docs.litellm.ai/docs/routing>
[^ll3]: <https://docs.litellm.ai/docs/proxy/caching>
[^ll4]: <https://docs.litellm.ai/docs/proxy/users>
[^ll5]: <https://github.com/BerriAI/litellm>

### Traceloop / OpenLLMetry & OpenInference

[^tl1]: <https://www.traceloop.com/docs> · <https://github.com/traceloop/openllmetry>
[^tl2]: <https://www.traceloop.com/docs/openllmetry/integrations/overview>
[^oi1]: <https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md>

### PromptLayer, Promptfoo, Mirascope, Helicone Prompts

[^pl1]: <https://docs.promptlayer.com/quickstart>
[^pf1]: <https://www.promptfoo.dev/docs/configuration/prompts/>
[^pf2]: <https://www.promptfoo.dev/docs/intro>
[^mc1]: <https://mirascope.com/docs/prompts/managing-prompts>

### Guardrails / safety

[^ga1]: <https://www.guardrailsai.com/docs>
[^ga2]: <https://www.guardrailsai.com/docs/concepts/output_specs>
[^nm1]: <https://docs.nvidia.com/nemo/guardrails/index.html>
[^nm2]: <https://github.com/NVIDIA/NeMo-Guardrails/blob/main/docs/user_guides/colang-language-syntax-guide.md>
[^lk1]: <https://www.lakera.ai/lakera-guard>
[^pt1]: <https://www.patronus.ai/products>
[^pr1]: <https://protectai.com/guardian>

### Evals-first

[^rg1]: <https://docs.ragas.io/en/stable/>
[^de1]: <https://docs.confident-ai.com/docs/getting-started>
