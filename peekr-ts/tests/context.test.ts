import { test, beforeEach } from "node:test";
import assert from "node:assert/strict";
import {
  withSpan,
  getCurrentSpan,
  withSession,
  addExporter,
  clearExporters,
  createDetachedSpan,
  finishDetachedSpan,
} from "../src/context.js";
import type { Exporter, Span } from "../src/span.js";

class CollectingExporter implements Exporter {
  readonly spans: Span[] = [];
  export(span: Span): void {
    this.spans.push(span);
  }
}

beforeEach(() => clearExporters());

test("withSpan with no parent generates a fresh trace_id and runs the callback", () => {
  const exp = new CollectingExporter();
  addExporter(exp);
  const out = withSpan("op.a", (span) => {
    assert.equal(span.parent_id, null);
    assert.equal(span.trace_id.length, 32);
    return 42;
  });
  assert.equal(out, 42);
  assert.equal(exp.spans.length, 1);
  assert.equal(exp.spans[0]!.status, "ok");
});

test("nested withSpan inherits trace_id and sets parent_id", () => {
  const exp = new CollectingExporter();
  addExporter(exp);
  withSpan("outer", () => {
    withSpan("inner", () => undefined);
  });
  assert.equal(exp.spans.length, 2);
  const [inner, outer] = exp.spans;
  assert.equal(inner!.trace_id, outer!.trace_id);
  assert.equal(inner!.parent_id, outer!.span_id);
  assert.equal(inner!.name, "inner");
  assert.equal(outer!.name, "outer");
});

test("getCurrentSpan returns the innermost open span and unwinds on end", () => {
  assert.equal(getCurrentSpan(), null);
  withSpan("outer", () => {
    assert.equal(getCurrentSpan()?.name, "outer");
    withSpan("inner", () => {
      assert.equal(getCurrentSpan()?.name, "inner");
    });
    assert.equal(getCurrentSpan()?.name, "outer");
  });
  assert.equal(getCurrentSpan(), null);
});

test("span tree propagates correctly across async/await", async () => {
  const exp = new CollectingExporter();
  addExporter(exp);

  async function tool(): Promise<void> {
    await withSpan("tool.fetch", async () => {
      await new Promise((r) => setTimeout(r, 5));
    });
  }

  async function agent(): Promise<void> {
    await withSpan("agent.run", async () => {
      await tool();
      await tool();
    });
  }

  await agent();

  assert.equal(exp.spans.length, 3);
  const [t1, t2, ag] = exp.spans;
  assert.equal(ag!.name, "agent.run");
  assert.equal(t1!.name, "tool.fetch");
  assert.equal(t2!.name, "tool.fetch");
  assert.equal(t1!.parent_id, ag!.span_id);
  assert.equal(t2!.parent_id, ag!.span_id);
  assert.equal(t1!.trace_id, ag!.trace_id);
  assert.equal(t2!.trace_id, ag!.trace_id);
});

test("withSpan records error status and rethrows on sync throw", () => {
  const exp = new CollectingExporter();
  addExporter(exp);
  assert.throws(() => withSpan("boom", () => { throw new Error("nope"); }), /nope/);
  assert.equal(exp.spans.length, 1);
  assert.equal(exp.spans[0]!.status, "error");
  assert.equal(exp.spans[0]!.attributes["error"], "nope");
});

test("withSpan records error status and rejects on async throw", async () => {
  const exp = new CollectingExporter();
  addExporter(exp);
  await assert.rejects(
    withSpan("boom", async () => { throw new Error("async-nope"); }),
    /async-nope/,
  );
  assert.equal(exp.spans.length, 1);
  assert.equal(exp.spans[0]!.status, "error");
});

test("withSession attaches user_id and session_id to spans inside it", () => {
  const exp = new CollectingExporter();
  addExporter(exp);

  withSession({ user_id: "u_123", session_id: "s_abc" }, () => {
    withSpan("op", () => undefined);
  });

  assert.equal(exp.spans[0]!.attributes["user_id"], "u_123");
  assert.equal(exp.spans[0]!.attributes["session_id"], "s_abc");
});

test("multiple concurrent async tasks have isolated span trees", async () => {
  const exp = new CollectingExporter();
  addExporter(exp);

  async function workload(tenant: string) {
    return withSession({ user_id: tenant }, async () => {
      await withSpan("agent.run", async () => {
        await new Promise((r) => setTimeout(r, Math.random() * 10));
      });
    });
  }

  await Promise.all([workload("acme"), workload("globex"), workload("initech")]);

  assert.equal(exp.spans.length, 3);
  const tenants = exp.spans.map((s) => s.attributes["user_id"]).sort();
  assert.deepEqual(tenants, ["acme", "globex", "initech"]);
  // Each agent.run has a distinct trace_id (separate top-level operations).
  const traceIds = new Set(exp.spans.map((s) => s.trace_id));
  assert.equal(traceIds.size, 3);
});

test("createDetachedSpan inherits parent from current store and exports on finish", () => {
  const exp = new CollectingExporter();
  addExporter(exp);
  withSpan("outer", () => {
    const detached = createDetachedSpan("streaming.op");
    detached.attributes["model"] = "gpt-4o";
    finishDetachedSpan(detached);
  });
  // 'streaming.op' finishes first (before 'outer' closes)
  assert.equal(exp.spans.length, 2);
  const [streaming, outer] = exp.spans;
  assert.equal(streaming!.name, "streaming.op");
  assert.equal(streaming!.parent_id, outer!.span_id);
  assert.equal(streaming!.trace_id, outer!.trace_id);
  assert.equal(streaming!.attributes["model"], "gpt-4o");
});
