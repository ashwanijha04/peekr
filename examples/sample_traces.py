"""Generate a realistic sample traces.jsonl for the peekr dashboard.

Each span has the same shape peekr writes at runtime (see `peekr/span.py`),
so the dashboard renders it indistinguishably from real production data.

Scenarios covered:
  - Faithful RAG answers           → Hallucination ~ 0.95–1.00
  - Faithful but tangential        → Hallucination ~ 0.80–0.95
  - Partially hallucinated         → Hallucination ~ 0.40–0.70
  - Heavily hallucinated           → Hallucination ~ 0.00–0.20
  - No-context / non-RAG calls     → Hallucination = 1.0 (nothing to judge)
  - LLM errors / timeouts          → status=error, no eval scores

Drift: the script emits spans in two phases.
  Phase A (first 60%) — a "healthy" model averaging ~0.92 groundedness.
  Phase B (last 40%)  — a "regressed" model averaging ~0.55 groundedness.
The dashboard's drift card should show this as a strong negative delta.

Usage:
    python examples/sample_traces.py                          # → ./sample_traces.jsonl
    python examples/sample_traces.py -o my_traces.jsonl       # custom path
    python examples/sample_traces.py --n 120                  # more spans
    peekr dashboard sample_traces.jsonl -o dashboard.html
"""

from __future__ import annotations

import argparse
import json
import random
import time
import uuid

# ---------------------------------------------------------------------------
# Scenario corpus
# ---------------------------------------------------------------------------
# Each item: (context, prompt, faithful_answer, partial_answer, hallucinated_answer)
# - faithful: every factual claim entailed by context
# - partial:  some claims supported, one wrong detail or one invented detail
# - hallucinated: most factual claims wrong or invented
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "topic": "Eiffel Tower",
        "context": "The Eiffel Tower was completed in 1889 for the World's Fair held in Paris. It was designed by the engineering firm of Gustave Eiffel.",
        "prompt": "When was the Eiffel Tower completed and who designed it?",
        "faithful": "The Eiffel Tower was completed in 1889 and was designed by Gustave Eiffel's firm.",
        "partial":  "The Eiffel Tower was completed in 1889 and was designed by Gustave Eiffel for the 1900 Paris Olympics.",
        "halluc":   "The Eiffel Tower was completed in 1923 and was designed by Frank Lloyd Wright for the Worlds Fair in London.",
        "halluc_claims": [
            {"text": "The Eiffel Tower was completed in 1923", "verdict": "contradicted"},
            {"text": "It was designed by Frank Lloyd Wright",   "verdict": "contradicted"},
            {"text": "The Worlds Fair was in London",          "verdict": "contradicted"},
        ],
        "partial_claims": [
            {"text": "The Eiffel Tower was completed in 1889",        "verdict": "supported"},
            {"text": "It was designed by Gustave Eiffel",            "verdict": "supported"},
            {"text": "The Paris Olympics were held in 1900",         "verdict": "unsupported"},
        ],
        "faithful_claims": [
            {"text": "The Eiffel Tower was completed in 1889", "verdict": "supported"},
            {"text": "It was designed by Gustave Eiffel's firm", "verdict": "supported"},
        ],
    },
    {
        "topic": "Python language",
        "context": "Python is a high-level programming language created by Guido van Rossum and first released in 1991.",
        "prompt": "Who created Python and when was it released?",
        "faithful": "Python was created by Guido van Rossum and first released in 1991.",
        "partial":  "Python was created by Guido van Rossum in 1991 at Google.",
        "halluc":   "Python was created by Linus Torvalds in 2003 as a fork of Perl.",
        "halluc_claims": [
            {"text": "Python was created by Linus Torvalds", "verdict": "contradicted"},
            {"text": "Python was created in 2003",          "verdict": "contradicted"},
            {"text": "Python is a fork of Perl",            "verdict": "unsupported"},
        ],
        "partial_claims": [
            {"text": "Python was created by Guido van Rossum", "verdict": "supported"},
            {"text": "Python was first released in 1991",      "verdict": "supported"},
            {"text": "Python was created at Google",           "verdict": "unsupported"},
        ],
        "faithful_claims": [
            {"text": "Python was created by Guido van Rossum", "verdict": "supported"},
            {"text": "Python was first released in 1991",      "verdict": "supported"},
        ],
    },
    {
        "topic": "Mount Everest",
        "context": "Mount Everest, on the border of Nepal and Tibet, is the highest mountain above sea level at 8,848.86 metres. It was first summited by Edmund Hillary and Tenzing Norgay in 1953.",
        "prompt": "Who first climbed Everest and how tall is it?",
        "faithful": "Edmund Hillary and Tenzing Norgay first summited Mount Everest in 1953. It is 8,848.86 metres tall.",
        "partial":  "Edmund Hillary first summited Everest in 1953. It is 9,144 metres tall.",
        "halluc":   "Reinhold Messner first climbed Everest in 1978. It is 7,500 metres tall and located in India.",
        "halluc_claims": [
            {"text": "Reinhold Messner first climbed Everest",  "verdict": "contradicted"},
            {"text": "Everest was first climbed in 1978",       "verdict": "contradicted"},
            {"text": "Everest is 7,500 metres tall",            "verdict": "contradicted"},
            {"text": "Everest is located in India",             "verdict": "contradicted"},
        ],
        "partial_claims": [
            {"text": "Edmund Hillary first summited Everest", "verdict": "supported"},
            {"text": "Everest was summited in 1953",          "verdict": "supported"},
            {"text": "Everest is 9,144 metres tall",          "verdict": "contradicted"},
        ],
        "faithful_claims": [
            {"text": "Edmund Hillary and Tenzing Norgay first summited Everest", "verdict": "supported"},
            {"text": "The summit was reached in 1953",                            "verdict": "supported"},
            {"text": "Everest is 8,848.86 metres tall",                           "verdict": "supported"},
        ],
    },
    {
        "topic": "Australia capital",
        "context": "Canberra is the capital city of Australia. It was selected as a compromise between Sydney and Melbourne and became the capital in 1927.",
        "prompt": "What is the capital of Australia?",
        "faithful": "The capital of Australia is Canberra; it has been the capital since 1927.",
        "partial":  "The capital of Australia is Canberra, chosen in 1908 to balance Sydney and Brisbane.",
        "halluc":   "The capital of Australia is Sydney; it has been the capital since the country's founding in 1901.",
        "halluc_claims": [
            {"text": "The capital of Australia is Sydney",         "verdict": "contradicted"},
            {"text": "Sydney has been the capital since 1901",     "verdict": "contradicted"},
        ],
        "partial_claims": [
            {"text": "The capital of Australia is Canberra",   "verdict": "supported"},
            {"text": "Canberra was chosen in 1908",            "verdict": "contradicted"},
            {"text": "Brisbane was a candidate for capital",   "verdict": "contradicted"},
        ],
        "faithful_claims": [
            {"text": "The capital of Australia is Canberra", "verdict": "supported"},
            {"text": "Canberra became the capital in 1927",  "verdict": "supported"},
        ],
    },
    {
        "topic": "Photosynthesis",
        "context": "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen, primarily in the chloroplasts of leaves.",
        "prompt": "What is photosynthesis?",
        "faithful": "Photosynthesis is the process plants use to turn sunlight, water, and CO2 into glucose and oxygen, mostly in chloroplasts.",
        "partial":  "Photosynthesis is the process by which plants convert sunlight, water, and CO2 into glucose and oxygen — it happens entirely in the roots.",
        "halluc":   "Photosynthesis is the process by which animals digest sunlight to produce nitrogen and hydrogen in their stomachs.",
        "halluc_claims": [
            {"text": "Photosynthesis is a process performed by animals",                 "verdict": "contradicted"},
            {"text": "Photosynthesis produces nitrogen and hydrogen",                    "verdict": "contradicted"},
            {"text": "Photosynthesis happens in the stomach",                            "verdict": "contradicted"},
        ],
        "partial_claims": [
            {"text": "Plants convert sunlight, water, and CO2 into glucose and oxygen", "verdict": "supported"},
            {"text": "Photosynthesis happens in the roots",                              "verdict": "contradicted"},
        ],
        "faithful_claims": [
            {"text": "Plants convert sunlight, water, and CO2 into glucose and oxygen", "verdict": "supported"},
            {"text": "Photosynthesis happens in chloroplasts",                            "verdict": "supported"},
        ],
    },
    {
        "topic": "Speed of light",
        "context": "The speed of light in a vacuum is approximately 299,792,458 metres per second. This constant is denoted by the letter c and is fundamental to Einstein's theory of special relativity.",
        "prompt": "How fast is light?",
        "faithful": "Light travels at roughly 299,792,458 m/s in a vacuum. This is the constant c in special relativity.",
        "partial":  "Light travels at roughly 299,792,458 m/s in a vacuum. Einstein discovered it in 1922.",
        "halluc":   "Light travels at 186,000 km/s in air and was first measured by Isaac Newton in 1666.",
        "halluc_claims": [
            {"text": "Light travels at 186,000 km/s in air",       "verdict": "contradicted"},
            {"text": "Isaac Newton first measured the speed of light", "verdict": "unsupported"},
            {"text": "It was measured in 1666",                    "verdict": "unsupported"},
        ],
        "partial_claims": [
            {"text": "Light travels at 299,792,458 m/s in vacuum", "verdict": "supported"},
            {"text": "Einstein discovered the speed of light in 1922", "verdict": "contradicted"},
        ],
        "faithful_claims": [
            {"text": "Light travels at 299,792,458 m/s in a vacuum", "verdict": "supported"},
            {"text": "It is the constant c in special relativity",  "verdict": "supported"},
        ],
    },
    {
        "topic": "DNA structure",
        "context": "The double helix structure of DNA was described by James Watson and Francis Crick in 1953, building on X-ray crystallography by Rosalind Franklin.",
        "prompt": "Who discovered the structure of DNA?",
        "faithful": "Watson and Crick described the double helix in 1953, using Rosalind Franklin's X-ray crystallography.",
        "partial":  "Watson and Crick described the DNA double helix in 1953, with Linus Pauling's X-ray work.",
        "halluc":   "Charles Darwin discovered DNA in 1859 by using an electron microscope.",
        "halluc_claims": [
            {"text": "Charles Darwin discovered DNA",         "verdict": "contradicted"},
            {"text": "DNA was discovered in 1859",            "verdict": "contradicted"},
            {"text": "Darwin used an electron microscope",    "verdict": "contradicted"},
        ],
        "partial_claims": [
            {"text": "Watson and Crick described the DNA double helix", "verdict": "supported"},
            {"text": "They described it in 1953",                        "verdict": "supported"},
            {"text": "Linus Pauling contributed X-ray work",             "verdict": "contradicted"},
        ],
        "faithful_claims": [
            {"text": "Watson and Crick described the double helix",       "verdict": "supported"},
            {"text": "Rosalind Franklin produced the X-ray crystallography", "verdict": "supported"},
            {"text": "The work was published in 1953",                       "verdict": "supported"},
        ],
    },
    {
        "topic": "World War II",
        "context": "World War II began in 1939 when Germany invaded Poland and ended in 1945 with the surrender of Japan after the atomic bombings of Hiroshima and Nagasaki.",
        "prompt": "When did World War II start and end?",
        "faithful": "WWII began in 1939 with Germany's invasion of Poland and ended in 1945 after Japan surrendered.",
        "partial":  "WWII began in 1939 and ended in 1944 with the Treaty of Versailles.",
        "halluc":   "WWII began in 1914 when Italy invaded France and ended in 1953.",
        "halluc_claims": [
            {"text": "WWII began in 1914",            "verdict": "contradicted"},
            {"text": "Italy invaded France",           "verdict": "contradicted"},
            {"text": "WWII ended in 1953",             "verdict": "contradicted"},
        ],
        "partial_claims": [
            {"text": "WWII began in 1939",                  "verdict": "supported"},
            {"text": "WWII ended in 1944",                  "verdict": "contradicted"},
            {"text": "It ended with the Treaty of Versailles", "verdict": "contradicted"},
        ],
        "faithful_claims": [
            {"text": "WWII began in 1939",                          "verdict": "supported"},
            {"text": "It started with Germany invading Poland",     "verdict": "supported"},
            {"text": "WWII ended in 1945",                          "verdict": "supported"},
        ],
    },
    {
        "topic": "Mona Lisa",
        "context": "The Mona Lisa was painted by Leonardo da Vinci between 1503 and 1519 and is on display at the Louvre in Paris.",
        "prompt": "Who painted the Mona Lisa and where is it?",
        "faithful": "Leonardo da Vinci painted the Mona Lisa (1503–1519); it hangs in the Louvre in Paris.",
        "partial":  "Leonardo da Vinci painted the Mona Lisa in 1505; it hangs in the Uffizi in Florence.",
        "halluc":   "Michelangelo painted the Mona Lisa in 1620 and it is displayed at the Vatican Museum.",
        "halluc_claims": [
            {"text": "Michelangelo painted the Mona Lisa",       "verdict": "contradicted"},
            {"text": "The Mona Lisa was painted in 1620",        "verdict": "contradicted"},
            {"text": "The Mona Lisa is at the Vatican Museum",   "verdict": "contradicted"},
        ],
        "partial_claims": [
            {"text": "Leonardo da Vinci painted the Mona Lisa", "verdict": "supported"},
            {"text": "It was painted in 1505",                  "verdict": "supported"},
            {"text": "It hangs in the Uffizi in Florence",      "verdict": "contradicted"},
        ],
        "faithful_claims": [
            {"text": "Leonardo da Vinci painted the Mona Lisa", "verdict": "supported"},
            {"text": "It was painted between 1503 and 1519",    "verdict": "supported"},
            {"text": "It is at the Louvre",                     "verdict": "supported"},
        ],
    },
    {
        "topic": "Mars",
        "context": "Mars is the fourth planet from the Sun. Its two small moons are Phobos and Deimos.",
        "prompt": "How many moons does Mars have?",
        "faithful": "Mars has two moons: Phobos and Deimos.",
        "partial":  "Mars has two moons: Phobos and Titan.",
        "halluc":   "Mars has 12 moons, the largest of which is Europa.",
        "halluc_claims": [
            {"text": "Mars has 12 moons",                "verdict": "contradicted"},
            {"text": "Mars's largest moon is Europa",     "verdict": "contradicted"},
        ],
        "partial_claims": [
            {"text": "Mars has two moons",  "verdict": "supported"},
            {"text": "One of them is Phobos", "verdict": "supported"},
            {"text": "One of them is Titan",  "verdict": "contradicted"},
        ],
        "faithful_claims": [
            {"text": "Mars has two moons",   "verdict": "supported"},
            {"text": "They are Phobos and Deimos", "verdict": "supported"},
        ],
    },
]

MODELS = ["gpt-4o", "gpt-4o-mini", "claude-haiku-4-5-20251001", "claude-opus-4-5"]

# Tenants / endpoints — populated onto each span so the dashboard's
# per-channel drift panel and the channel_drift report have data to slice by.
TENANTS = ["acme", "globex", "initech", "umbrella"]
ENDPOINTS = ["/api/qa", "/api/summarize", "/api/rag-search", "/api/agent"]

# Per-model bias: some models hallucinate more than others in the synthetic data,
# so the "drift by channel" panel shows real differences. Tuple is (faithful, partial, halluc) prob.
MODEL_BIAS = {
    "gpt-4o":                       (0.80, 0.15, 0.05),
    "gpt-4o-mini":                  (0.65, 0.20, 0.15),
    "claude-haiku-4-5-20251001":    (0.55, 0.25, 0.20),
    "claude-opus-4-5":              (0.85, 0.12, 0.03),
}

# Approximate score ranges for each scenario class
SCORE_RANGES = {
    "faithful":  (0.92, 1.00),
    "partial":   (0.45, 0.75),
    "halluc":    (0.00, 0.20),
    "nocontext": (1.00, 1.00),
}


def _hex(n: int = 16) -> str:
    return uuid.uuid4().hex[:n]


def _build_messages(context: str, prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": context},
        {"role": "user",   "content": prompt},
    ]


def _make_span(
    *,
    trace_id: str,
    name: str,
    start: float,
    duration_ms: float,
    attributes: dict,
    status: str = "ok",
    parent_id: str | None = None,
) -> dict:
    return {
        "name": name,
        "trace_id": trace_id,
        "span_id": _hex(),
        "parent_id": parent_id,
        "start_time": start,
        "end_time": start + duration_ms / 1000.0,
        "duration_ms": duration_ms,
        "attributes": attributes,
        "status": status,
    }


def _make_scenario_span(
    sc: dict,
    kind: str,
    *,
    start: float,
    model: str,
    detailed: bool,
    tenant: str,
    endpoint: str,
) -> dict:
    """Create an LLM span for a single scenario at the requested truthfulness kind."""
    messages = _build_messages(sc["context"], sc["prompt"])
    answer = sc[{"faithful": "faithful", "partial": "partial", "halluc": "halluc"}[kind]]
    lo, hi = SCORE_RANGES[kind]
    score = round(random.uniform(lo, hi), 3)

    attrs: dict = {
        "model": model,
        "input": json.dumps(messages),
        "output": answer,
        "tokens_input": random.randint(50, 150),
        "tokens_output": random.randint(20, 80),
        "user_id":   tenant,        # treat user_id as tenant identifier
        "endpoint":  endpoint,
    }
    attrs["tokens_total"] = attrs["tokens_input"] + attrs["tokens_output"]
    citation_score = 1.0 if kind == "faithful" else (0.5 if kind == "partial" else 0.0)
    attrs["eval_scores"] = {
        "Hallucination":     score,
        "Rubric":            round(random.uniform(0.7, 0.95), 3),
        "CitationAccuracy":  citation_score,
        "NotEmpty":          1.0,
        "NoError":           1.0,
    }

    if detailed:
        claims = sc[f"{kind}_claims"]
        total = len(claims)
        supported = sum(1 for c in claims if c["verdict"] == "supported")
        contradicted = sum(1 for c in claims if c["verdict"] == "contradicted")
        unsupported = sum(1 for c in claims if c["verdict"] == "unsupported")
        attrs["hallucination_details"] = {
            "claims": claims,
            "supported": supported,
            "contradicted": contradicted,
            "unsupported": unsupported,
            "total": total,
            "score": supported / total if total else 1.0,
        }
        # When detailed mode is on, the score on eval_scores should reflect claim ratio.
        attrs["eval_scores"]["Hallucination"] = attrs["hallucination_details"]["score"]

    return _make_span(
        trace_id=_hex(),
        name="openai.chat.completions" if model.startswith("gpt") else "anthropic.messages",
        start=start,
        duration_ms=random.uniform(400, 2100),
        attributes=attrs,
    )


def _make_nocontext_span(*, start: float, model: str, tenant: str, endpoint: str) -> dict:
    """A chat call with no real grounding context — Hallucination evaluator returns 1.0 by design."""
    attrs = {
        "model": model,
        "input": json.dumps([{"role": "user", "content": "What's a good metaphor for being patient?"}]),
        "output": "Being patient is like waiting for a kettle to boil — staring at it doesn't help.",
        "tokens_input": random.randint(20, 60),
        "tokens_output": random.randint(20, 60),
        "user_id":  tenant,
        "endpoint": endpoint,
    }
    attrs["tokens_total"] = attrs["tokens_input"] + attrs["tokens_output"]
    attrs["eval_scores"] = {
        "Hallucination":    1.0,   # no context → not evaluable, stays 1.0
        "Rubric":           round(random.uniform(0.8, 0.95), 3),
        "CitationAccuracy": 1.0,
        "NotEmpty":         1.0,
        "NoError":          1.0,
    }
    return _make_span(
        trace_id=_hex(),
        name="openai.chat.completions" if model.startswith("gpt") else "anthropic.messages",
        start=start,
        duration_ms=random.uniform(300, 900),
        attributes=attrs,
    )


def _make_citation_halluc_span(*, start: float, model: str, tenant: str, endpoint: str) -> dict:
    """An answer with one real fact + one invented citation — exactly what CitationAccuracy is for."""
    context = "BERT was introduced by Devlin et al. in 2018 at Google."
    answer = ("BERT was introduced by Devlin et al. in 2018. See arXiv:9999.12345 "
              "and 'Sparse Attention Mechanisms' (Vaswani et al. 2025) for follow-ups.")
    messages = [
        {"role": "system", "content": context},
        {"role": "user",   "content": "Who introduced BERT and what's a follow-up?"},
    ]
    attrs = {
        "model": model,
        "input": json.dumps(messages),
        "output": answer,
        "tokens_input": random.randint(70, 130),
        "tokens_output": random.randint(40, 90),
        "user_id":  tenant,
        "endpoint": endpoint,
    }
    attrs["tokens_total"] = attrs["tokens_input"] + attrs["tokens_output"]
    # Real Hallucination judge would flag this around 0.4 (one supported claim,
    # two unsupported), and CitationAccuracy flags 0.0 (two invented citations).
    attrs["eval_scores"] = {
        "Hallucination":    round(random.uniform(0.35, 0.55), 3),
        "Rubric":           round(random.uniform(0.7, 0.85), 3),
        "CitationAccuracy": 0.0,
        "NotEmpty":         1.0,
        "NoError":          1.0,
    }
    attrs["citation_details"] = {
        "total": 2, "grounded": 0, "invented": 2,
        "items": [
            {"kind": "arxiv",       "text": "arXiv:9999.12345",     "grounded": False},
            {"kind": "author_year", "text": "Vaswani et al. 2025",  "grounded": False},
        ],
    }
    return _make_span(
        trace_id=_hex(),
        name="openai.chat.completions" if model.startswith("gpt") else "anthropic.messages",
        start=start,
        duration_ms=random.uniform(500, 1500),
        attributes=attrs,
    )


def _make_error_span(*, start: float, model: str, tenant: str, endpoint: str) -> dict:
    attrs = {
        "model": model,
        "input": json.dumps([{"role": "user", "content": "Summarize the report."}]),
        "error": "openai.RateLimitError: 429 — too many requests",
        "user_id":  tenant,
        "endpoint": endpoint,
    }
    return _make_span(
        trace_id=_hex(),
        name="openai.chat.completions" if model.startswith("gpt") else "anthropic.messages",
        start=start,
        duration_ms=random.uniform(50, 200),
        attributes=attrs,
        status="error",
    )


def generate(n: int, output: str, seed: int = 42) -> str:
    random.seed(seed)
    t0 = time.time() - n * 30   # spread spans across the recent past
    spans: list[dict] = []

    # Phase split: first 60% "healthy" — heavily skewed to faithful answers.
    # Last 40% "regressed" — many more partials and full hallucinations (visible drift).
    phase_a_end = int(n * 0.6)

    detailed_indices = set(random.sample(range(n), max(1, n // 4)))
    # Sprinkle ~8% citation-hallucinating spans throughout, weighted to the regressed phase.
    citation_indices = set(random.sample(range(n), max(1, n // 12)))

    # Pin tenant→endpoint affinity so each tenant looks like a real product
    # (different traffic mix per customer).
    tenant_endpoint_affinity = {
        "acme":     ["/api/qa", "/api/qa", "/api/summarize"],
        "globex":   ["/api/rag-search", "/api/rag-search", "/api/agent"],
        "initech":  ["/api/agent", "/api/summarize", "/api/agent"],
        "umbrella": ["/api/qa", "/api/summarize", "/api/rag-search"],
    }

    for i in range(n):
        start = t0 + i * random.uniform(25, 35)
        tenant = random.choice(TENANTS)
        endpoint = random.choice(tenant_endpoint_affinity[tenant])
        # Bias model choice slightly by tenant — globex is mostly on haiku, acme on opus, etc.
        if tenant == "globex":
            model = random.choices(MODELS, weights=[1, 2, 4, 1])[0]
        elif tenant == "acme":
            model = random.choices(MODELS, weights=[2, 1, 1, 4])[0]
        else:
            model = random.choice(MODELS)
        detailed = i in detailed_indices

        # Citation-halluc carve-out (overrides normal selection)
        if i in citation_indices:
            spans.append(_make_citation_halluc_span(
                start=start, model=model, tenant=tenant, endpoint=endpoint,
            ))
            continue

        roll = random.random()

        if i < phase_a_end:
            # Healthy phase
            if roll < 0.10:
                spans.append(_make_nocontext_span(start=start, model=model, tenant=tenant, endpoint=endpoint))
                continue
            if roll < 0.13:
                spans.append(_make_error_span(start=start, model=model, tenant=tenant, endpoint=endpoint))
                continue
            # Use model-biased mix
            faithful_p, partial_p, _ = MODEL_BIAS[model]
            r2 = random.random()
            if r2 < faithful_p:
                kind = "faithful"
            elif r2 < faithful_p + partial_p:
                kind = "partial"
            else:
                kind = "halluc"
        else:
            # Regressed phase: shift the curve significantly worse for every model
            if roll < 0.06:
                spans.append(_make_nocontext_span(start=start, model=model, tenant=tenant, endpoint=endpoint))
                continue
            if roll < 0.10:
                spans.append(_make_error_span(start=start, model=model, tenant=tenant, endpoint=endpoint))
                continue
            r2 = random.random()
            if r2 < 0.35:
                kind = "faithful"
            elif r2 < 0.65:
                kind = "partial"
            else:
                kind = "halluc"

        sc = random.choice(SCENARIOS)
        spans.append(_make_scenario_span(
            sc, kind, start=start, model=model, detailed=detailed,
            tenant=tenant, endpoint=endpoint,
        ))

    with open(output, "w") as f:
        for s in spans:
            f.write(json.dumps(s) + "\n")
    return output


def main():
    ap = argparse.ArgumentParser(description="Generate sample peekr traces for the dashboard demo.")
    ap.add_argument("-o", "--output", default="sample_traces.jsonl", help="Output JSONL path")
    ap.add_argument("--n", type=int, default=80, help="Number of LLM spans to generate")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    path = generate(args.n, args.output, seed=args.seed)

    # Quick stats
    with open(path) as f:
        spans = [json.loads(line) for line in f if line.strip()]
    scored = [s for s in spans if (s["attributes"].get("eval_scores") or {}).get("Hallucination") is not None]
    detailed = [s for s in spans if s["attributes"].get("hallucination_details")]
    errors = [s for s in spans if s.get("status") == "error"]
    avg_h = sum((s["attributes"]["eval_scores"]["Hallucination"]) for s in scored) / max(len(scored), 1)

    print(f"Wrote {len(spans)} spans → {path}")
    print(f"  • scored:    {len(scored)}")
    print(f"  • detailed:  {len(detailed)}")
    print(f"  • errors:    {len(errors)}")
    print(f"  • avg Hallucination score: {avg_h:.3f}")
    print()
    print("Next:  peekr dashboard sample_traces.jsonl -o dashboard.html  &&  open dashboard.html")


if __name__ == "__main__":
    main()
