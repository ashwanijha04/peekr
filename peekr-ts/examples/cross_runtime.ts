/**
 * End-to-end demo: produce a peekr-compatible traces.jsonl from a TS
 * process, then read it with the Python `peekr` CLI / dashboard.
 *
 * No real API keys needed — we mock OpenAI and Anthropic clients with the
 * same shape so the wrappers exercise their real code paths. Running:
 *
 *   npx tsx peekr-ts/examples/cross_runtime.ts
 *   pip install peekr
 *   peekr view --io traces.jsonl
 *   peekr dashboard traces.jsonl -o dashboard.html
 *   open dashboard.html
 *
 * is the same verification flow we'd run against a real app — proves the
 * JSONL schema is wire-compatible across the two SDKs.
 */

import {
  instrument,
  wrap,
  trace,
  withSession,
  withSpan,
} from "../src/index.js";

instrument({ jsonlPath: "./traces.jsonl", reset: true });

// ---------------------------------------------------------------------------
// Mock OpenAI client (no `openai` package needed to verify the schema).
// ---------------------------------------------------------------------------
function mockOpenAI() {
  return {
    chat: {
      completions: {
        async create(params: { messages?: unknown[]; model?: string }) {
          await new Promise((r) => setTimeout(r, 30 + Math.random() * 200));
          const q = JSON.stringify(params.messages ?? []);
          // Pretend we hallucinate occasionally.
          const willHallucinate = Math.random() < 0.4;
          const content = willHallucinate
            ? "The Eiffel Tower was completed in 1923 by Frank Lloyd Wright."
            : "The Eiffel Tower was completed in 1889 by Gustave Eiffel's firm.";
          return {
            id: "resp_" + Math.random().toString(36).slice(2, 10),
            choices: [{ message: { content }, finish_reason: "stop" }],
            usage: {
              prompt_tokens: 30 + q.length % 50,
              completion_tokens: 20 + content.length % 30,
              total_tokens: 50 + (q.length + content.length) % 70,
            },
          };
        },
      },
    },
  };
}

function mockAnthropic() {
  return {
    messages: {
      async create(params: { messages?: unknown[]; model?: string }) {
        await new Promise((r) => setTimeout(r, 30 + Math.random() * 250));
        const text = "Yes — the Eiffel Tower was completed in 1889 in Paris.";
        return {
          id: "msg_" + Math.random().toString(36).slice(2, 10),
          content: [{ type: "text", text }],
          usage: {
            input_tokens: 25 + (params.messages?.length ?? 0) * 5,
            output_tokens: 18,
          },
        };
      },
    },
  };
}

const openai = wrap(mockOpenAI());
const anthropic = wrap(mockAnthropic());

// ---------------------------------------------------------------------------
// Simulate a multi-tenant RAG-style workload
// ---------------------------------------------------------------------------

const TENANTS = ["acme", "globex", "initech", "umbrella"];
const ENDPOINTS = ["/api/qa", "/api/summarize", "/api/agent"];

const searchTool = trace({ name: "tool.search" }, async (query: string) => {
  await new Promise((r) => setTimeout(r, 50));
  return [
    `Result 1 for ${query}`,
    `Result 2 for ${query}`,
  ];
});

async function handleRequest(tenant: string, endpoint: string, q: string) {
  await withSession({ user_id: tenant }, async () => {
    await withSpan(`http.POST ${endpoint}`, async (span) => {
      span.attributes["endpoint"] = endpoint;

      const docs = await searchTool(q);

      // Half the requests hit OpenAI, half Anthropic.
      if (Math.random() < 0.5) {
        await openai.chat.completions.create({
          model: "gpt-4o-mini",
          messages: [
            { role: "system", content: docs.join("\n") },
            { role: "user", content: q },
          ],
        });
      } else {
        await anthropic.messages.create({
          model: "claude-haiku-4-5-20251001",
          system: docs.join("\n"),
          messages: [{ role: "user", content: q }],
        });
      }
    });
  });
}

// Drive a small workload
async function main() {
  const questions = [
    "When was the Eiffel Tower completed?",
    "Who designed the Eiffel Tower?",
    "Summarise the construction history of the Eiffel Tower.",
  ];

  const tasks: Promise<void>[] = [];
  for (let i = 0; i < 20; i++) {
    const tenant = TENANTS[i % TENANTS.length]!;
    const endpoint = ENDPOINTS[i % ENDPOINTS.length]!;
    const q = questions[i % questions.length]!;
    tasks.push(handleRequest(tenant, endpoint, q));
  }
  await Promise.all(tasks);

  console.log(`Wrote ${tasks.length} simulated request traces to ./traces.jsonl`);
  console.log("Next:");
  console.log("  pip install peekr");
  console.log("  peekr view --io traces.jsonl");
  console.log("  peekr dashboard traces.jsonl -o dashboard.html");
  console.log("  open dashboard.html");
}

await main();
