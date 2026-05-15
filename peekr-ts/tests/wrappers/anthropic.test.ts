import { test, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { wrapAnthropic } from "../../src/wrappers/anthropic.js";
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

function mockAnthropic(opts: {
  responseFactory: (params: any) => Promise<any> | any;
}) {
  return {
    messages: {
      create(params: any) {
        return opts.responseFactory(params);
      },
    },
  };
}

beforeEach(() => clearExporters());

test("wrapAnthropic captures a non-streaming messages.create call", async () => {
  const sink = new Sink();
  addExporter(sink);

  const raw = mockAnthropic({
    responseFactory: async () => ({
      id: "msg_1",
      content: [{ type: "text", text: "Hello, world." }],
      usage: { input_tokens: 8, output_tokens: 5 },
    }),
  });

  const client = wrapAnthropic(raw);
  const r = await client.messages.create({
    model: "claude-haiku-4-5-20251001",
    messages: [{ role: "user", content: "hi" }],
  });

  assert.equal(r.content[0].text, "Hello, world.");
  assert.equal(sink.spans.length, 1);
  const s = sink.spans[0]!;
  assert.equal(s.name, "anthropic.messages");
  assert.equal(s.attributes["model"], "claude-haiku-4-5-20251001");
  assert.equal(s.attributes["output"], "Hello, world.");
  assert.equal(s.attributes["tokens_input"], 8);
  assert.equal(s.attributes["tokens_output"], 5);
  assert.equal(s.attributes["tokens_total"], 13);
  assert.equal(s.status, "ok");
});

test("generic wrap() detects Anthropic by shape", () => {
  const raw = mockAnthropic({ responseFactory: () => Promise.resolve({}) });
  const wrapped = wrap(raw);
  assert.equal((wrapped as { __peekr_wrapped?: boolean }).__peekr_wrapped, true);
});

test("wrapAnthropic records error status when the call throws", async () => {
  const sink = new Sink();
  addExporter(sink);

  const raw = mockAnthropic({
    responseFactory: () => {
      throw new Error("anthropic-down");
    },
  });
  const client = wrapAnthropic(raw);
  assert.throws(() => client.messages.create({ model: "claude", messages: [] }), /anthropic-down/);
  assert.equal(sink.spans.length, 1);
  assert.equal(sink.spans[0]!.status, "error");
  assert.equal(sink.spans[0]!.attributes["error"], "anthropic-down");
});
