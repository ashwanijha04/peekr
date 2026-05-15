/**
 * Exporters — write completed spans somewhere durable.
 *
 * The JSONL format is the cross-runtime contract: the Python `peekr` CLI
 * and HTML dashboard consume the same files produced here.
 */

import { appendFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import type { Exporter, Span } from "./span.js";

/**
 * Append one span per line to a JSONL file.
 *
 * Uses synchronous append on purpose — span volume is low (LLM calls are
 * the hot path), and sync writes guarantee the file is on disk even if
 * the process crashes immediately after. Matches Python `JSONLExporter`.
 */
export class JSONLExporter implements Exporter {
  constructor(public readonly path: string) {
    try {
      mkdirSync(dirname(path), { recursive: true });
    } catch {
      /* directory may already exist or be the cwd */
    }
  }

  export(span: Span): void {
    const line = JSON.stringify(span.toRecord()) + "\n";
    appendFileSync(this.path, line, "utf8");
  }
}

/**
 * Pretty-print completed spans to the console as they finish.
 * Useful for local development. Matches Python `ConsoleExporter`.
 */
export class ConsoleExporter implements Exporter {
  export(span: Span): void {
    const r = span.toRecord();
    const ms = r.duration_ms != null ? `${r.duration_ms.toFixed(0)}ms` : "?";
    const attrs = r.attributes as Record<string, unknown>;
    const model = attrs["model"] ? ` [${String(attrs["model"])}]` : "";
    const tokens = attrs["tokens_total"] ? ` ${String(attrs["tokens_total"])}tok` : "";
    const status = r.status === "error" ? " \x1b[31mERROR\x1b[0m" : "";
    process.stderr.write(`\x1b[1m${r.name}\x1b[0m${model}  ${ms}${tokens}${status}\n`);
  }
}
