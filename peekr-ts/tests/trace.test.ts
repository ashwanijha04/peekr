import { test, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { trace } from "../src/trace.js";
import {
  addExporter,
  clearExporters,
} from "../src/context.js";
import type { Exporter, Span } from "../src/span.js";

class Sink implements Exporter {
  readonly spans: Span[] = [];
  export(s: Span): void {
    this.spans.push(s);
  }
}

beforeEach(() => clearExporters());

test("trace wraps a sync function and captures input/output", () => {
  const sink = new Sink();
  addExporter(sink);

  const add = trace({ name: "math.add" }, (a: number, b: number) => a + b);
  assert.equal(add(2, 3), 5);
  assert.equal(sink.spans.length, 1);
  assert.equal(sink.spans[0]!.name, "math.add");
  assert.equal(sink.spans[0]!.status, "ok");
  assert.equal(sink.spans[0]!.attributes["input"], "[2,3]");
  assert.equal(sink.spans[0]!.attributes["output"], "5");
});

test("trace wraps an async function and captures the resolved value", async () => {
  const sink = new Sink();
  addExporter(sink);

  const fetchUser = trace(async (id: number) => ({ id, name: "ash" }));
  const u = await fetchUser(7);
  assert.deepEqual(u, { id: 7, name: "ash" });
  assert.equal(sink.spans.length, 1);
  assert.equal(sink.spans[0]!.status, "ok");
  assert.equal(sink.spans[0]!.attributes["output"], '{"id":7,"name":"ash"}');
});

test("trace records error status and rethrows for sync throws", () => {
  const sink = new Sink();
  addExporter(sink);

  const boom = trace(() => {
    throw new Error("nope");
  });
  assert.throws(() => boom(), /nope/);
  assert.equal(sink.spans.length, 1);
  assert.equal(sink.spans[0]!.status, "error");
  assert.equal(sink.spans[0]!.attributes["error"], "nope");
});

test("trace records error status and rejects for async throws", async () => {
  const sink = new Sink();
  addExporter(sink);

  const boom = trace(async () => {
    throw new Error("async-nope");
  });
  await assert.rejects(boom(), /async-nope/);
  assert.equal(sink.spans.length, 1);
  assert.equal(sink.spans[0]!.status, "error");
  assert.equal(sink.spans[0]!.attributes["error"], "async-nope");
});

test("captureIO: false skips input/output recording", () => {
  const sink = new Sink();
  addExporter(sink);
  const f = trace({ captureIO: false, name: "secret" }, (token: string) => `${token}-ok`);
  f("sk-supersecret");
  assert.equal(sink.spans[0]!.attributes["input"], undefined);
  assert.equal(sink.spans[0]!.attributes["output"], undefined);
});

test("nested traced functions form a parent/child tree", () => {
  const sink = new Sink();
  addExporter(sink);

  const inner = trace({ name: "inner" }, (x: number) => x + 1);
  const outer = trace({ name: "outer" }, (x: number) => inner(x) * 2);

  assert.equal(outer(10), 22);
  // Inner finishes before outer.
  assert.equal(sink.spans.length, 2);
  assert.equal(sink.spans[0]!.name, "inner");
  assert.equal(sink.spans[1]!.name, "outer");
  assert.equal(sink.spans[0]!.parent_id, sink.spans[1]!.span_id);
  assert.equal(sink.spans[0]!.trace_id, sink.spans[1]!.trace_id);
});
