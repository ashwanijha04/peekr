import { test, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { wrapOpenAI } from "../../src/wrappers/openai.js";
import { wrap } from "../../src/wrap.js";
import {
  addExporter,
  clearExporters,
} from "../../src/context.js";
import type { Exporter, Span } from "../../src/span.js";

class Sink implements Exporter {
  readonly spans: Span[] = [];
  export(s: Span): void {
    this.spans.push(s);
  }
}

/**
 * Minimal mock that mimics the shape of the official `openai` client.
 * We don't need the real package to test the Proxy logic — the wrapper
 * only depends on the `client.chat.completions.create(params)` shape.
 */
function mockOpenAI(opts: {
  responseFactory: (params: any) => Promise<any> | any;
}) {
  return {
    chat: {
      completions: {
        create(params: any) {
          return opts.responseFactory(params);
        },
      },
    },
  };
}

beforeEach(() => clearExporters());

test("wrapOpenAI captures a non-streaming chat.completions.create call", async () => {
  const sink = new Sink();
  addExporter(sink);

  const raw = mockOpenAI({
    responseFactory: async () => ({
      id: "resp_1",
      choices: [{ message: { content: "Hello world" } }],
      usage: { prompt_tokens: 12, completion_tokens: 3, total_tokens: 15 },
    }),
  });

  const client = wrapOpenAI(raw);
  const r = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [{ role: "user", content: "hi" }],
  });

  assert.equal(r.choices[0].message.content, "Hello world");
  assert.equal(sink.spans.length, 1);
  const s = sink.spans[0]!;
  assert.equal(s.name, "openai.chat.completions");
  assert.equal(s.attributes["model"], "gpt-4o-mini");
  assert.equal(s.attributes["output"], "Hello world");
  assert.equal(s.attributes["tokens_input"], 12);
  assert.equal(s.attributes["tokens_output"], 3);
  assert.equal(s.attributes["tokens_total"], 15);
  assert.equal(s.status, "ok");
});

test("wrapOpenAI records status='error' when the underlying call rejects", async () => {
  const sink = new Sink();
  addExporter(sink);

  const raw = mockOpenAI({
    responseFactory: async () => {
      throw new Error("rate limited");
    },
  });
  const client = wrapOpenAI(raw);
  await assert.rejects(
    client.chat.completions.create({ model: "gpt-4o", messages: [] }),
    /rate limited/,
  );
  assert.equal(sink.spans.length, 1);
  assert.equal(sink.spans[0]!.status, "error");
  assert.equal(sink.spans[0]!.attributes["error"], "rate limited");
});

test("wrapOpenAI is idempotent — wrap(wrap(client)) === wrap(client) behaviour", () => {
  const raw = mockOpenAI({ responseFactory: () => Promise.resolve({}) });
  const once = wrapOpenAI(raw);
  const twice = wrapOpenAI(once);
  // The marker property identifies the inner wrap, and the second wrap
  // returns the already-wrapped client unchanged.
  assert.equal((twice as { __peekr_wrapped?: boolean }).__peekr_wrapped, true);
});

test("wrapOpenAI captures streaming usage from the final chunk", async () => {
  const sink = new Sink();
  addExporter(sink);

  async function* fakeStream() {
    yield { choices: [{ delta: { content: "Hel" } }] };
    yield { choices: [{ delta: { content: "lo" } }] };
    // OpenAI emits a final chunk with usage when include_usage: true.
    yield {
      choices: [],
      usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 },
    };
  }

  let receivedParams: any = null;
  const raw = mockOpenAI({
    responseFactory: (params) => {
      receivedParams = params;
      return Promise.resolve(fakeStream());
    },
  });
  const client = wrapOpenAI(raw);
  const stream = await client.chat.completions.create({
    model: "gpt-4o",
    messages: [{ role: "user", content: "hi" }],
    stream: true,
  });

  // Wrapper auto-injects include_usage so we get token counts in the final chunk.
  assert.equal(receivedParams.stream_options?.include_usage, true);

  const collected: any[] = [];
  for await (const chunk of stream as AsyncIterable<any>) collected.push(chunk);

  assert.equal(collected.length, 3);
  assert.equal(sink.spans.length, 1);
  const s = sink.spans[0]!;
  assert.equal(s.attributes["tokens_total"], 7);
  assert.equal(s.attributes["tokens_input"], 5);
  assert.equal(s.attributes["tokens_output"], 2);
  assert.equal(s.status, "ok");
});

test("generic wrap() detects OpenAI by shape", () => {
  const raw = mockOpenAI({ responseFactory: () => Promise.resolve({}) });
  const wrapped = wrap(raw);
  assert.equal((wrapped as { __peekr_wrapped?: boolean }).__peekr_wrapped, true);
});
