import { test, beforeEach } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { JSONLExporter } from "../src/exporters.js";
import { withSpan, addExporter, clearExporters } from "../src/context.js";

beforeEach(() => clearExporters());

test("JSONLExporter writes one line per span with the Python-compatible schema", () => {
  const dir = mkdtempSync(join(tmpdir(), "peekr-ts-"));
  const path = join(dir, "traces.jsonl");
  try {
    addExporter(new JSONLExporter(path));
    withSpan("outer", (outerSpan) => {
      outerSpan.attributes["model"] = "gpt-4o";
      withSpan("inner", (innerSpan) => {
        innerSpan.attributes["tokens_total"] = 42;
      });
    });

    const lines = readFileSync(path, "utf8").trim().split("\n");
    assert.equal(lines.length, 2);
    const innerRecord = JSON.parse(lines[0]!);
    const outerRecord = JSON.parse(lines[1]!);

    for (const k of [
      "name",
      "trace_id",
      "span_id",
      "parent_id",
      "start_time",
      "end_time",
      "attributes",
      "status",
      "duration_ms",
    ]) {
      assert.ok(k in innerRecord, `inner missing key: ${k}`);
      assert.ok(k in outerRecord, `outer missing key: ${k}`);
    }

    assert.equal(innerRecord.parent_id, outerRecord.span_id);
    assert.equal(innerRecord.trace_id, outerRecord.trace_id);
    assert.equal(innerRecord.attributes.tokens_total, 42);
    assert.equal(outerRecord.attributes.model, "gpt-4o");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test("Failed exporter does not break others or propagate exceptions", () => {
  const dir = mkdtempSync(join(tmpdir(), "peekr-ts-"));
  const path = join(dir, "traces.jsonl");
  try {
    addExporter({
      export() {
        throw new Error("kaboom");
      },
    });
    addExporter(new JSONLExporter(path));
    withSpan("op", () => undefined);
    const lines = readFileSync(path, "utf8").trim().split("\n");
    assert.equal(lines.length, 1);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
