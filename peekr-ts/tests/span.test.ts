import { test } from "node:test";
import assert from "node:assert/strict";
import { Span, hexId, unixSeconds } from "../src/span.js";

test("hexId returns 32-char lowercase hex (matches Python uuid.uuid4().hex)", () => {
  const id = hexId();
  assert.equal(id.length, 32);
  assert.match(id, /^[0-9a-f]{32}$/);
  // Two consecutive calls produce different ids
  assert.notEqual(hexId(), hexId());
});

test("unixSeconds returns a unix-seconds float with millisecond precision", () => {
  const t = unixSeconds();
  const now = Date.now() / 1000;
  assert.ok(Math.abs(t - now) < 1, "should be within 1s of now");
  // Has at least 3 decimal places (millisecond precision)
  assert.ok(t.toString().includes("."), "should be a float");
});

test("Span starts in 'ok' state with sensible defaults", () => {
  const s = new Span({ name: "test.op", trace_id: "abc" });
  assert.equal(s.name, "test.op");
  assert.equal(s.trace_id, "abc");
  assert.equal(s.parent_id, null);
  assert.equal(s.status, "ok");
  assert.equal(s.end_time, null);
  assert.equal(s.duration_ms, null);
});

test("Span.finish sets end_time and computes duration_ms", async () => {
  const s = new Span({ name: "test.op", trace_id: "abc" });
  await new Promise((r) => setTimeout(r, 20));
  s.finish();
  assert.ok(s.end_time !== null);
  assert.ok((s.duration_ms ?? 0) >= 15);
});

test("Span.toRecord matches the Python schema (field names + types)", () => {
  const s = new Span({ name: "openai.chat.completions", trace_id: "t1", parent_id: "p1" });
  s.attributes["model"] = "gpt-4o";
  s.attributes["tokens_total"] = 312;
  s.finish();
  const r = s.toRecord();
  // Required field names — these are the cross-runtime contract.
  assert.deepEqual(Object.keys(r).sort(), [
    "attributes",
    "duration_ms",
    "end_time",
    "name",
    "parent_id",
    "span_id",
    "start_time",
    "status",
    "trace_id",
  ]);
  assert.equal(r.name, "openai.chat.completions");
  assert.equal(r.trace_id, "t1");
  assert.equal(r.parent_id, "p1");
  assert.equal(r.status, "ok");
  assert.equal(typeof r.start_time, "number");
  assert.equal(typeof r.end_time, "number");
  assert.equal(typeof r.duration_ms, "number");
  assert.equal(r.attributes["model"], "gpt-4o");
  assert.equal(r.attributes["tokens_total"], 312);
});
