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

export interface HTTPExporterOptions {
  endpoint: string;
  apiKey: string;
  batchSize?: number;
  flushIntervalSeconds?: number;
  timeoutSeconds?: number;
}

/**
 * Ship spans to a Peekr Cloud (or self-hosted) ingestion endpoint.
 *
 * Reserved public surface — implementation lands with Peekr Cloud GA.
 * The constructor signature is stable as of v0.3 so call sites won't
 * change when the body is filled in.
 *
 * Until then `.export()` throws so a misconfigured pipeline fails loudly
 * rather than silently dropping spans.
 *
 * Get on the waitlist:
 * https://github.com/ashwanijha04/peekr/discussions
 */
export class HTTPExporter implements Exporter {
  readonly endpoint: string;
  readonly apiKey: string;
  readonly batchSize: number;
  readonly flushIntervalSeconds: number;
  readonly timeoutSeconds: number;

  constructor(opts: HTTPExporterOptions) {
    if (!opts.endpoint) throw new Error("HTTPExporter: endpoint is required");
    if (!opts.apiKey) throw new Error("HTTPExporter: apiKey is required");
    this.endpoint = opts.endpoint.replace(/\/+$/, "");
    this.apiKey = opts.apiKey;
    this.batchSize = opts.batchSize ?? 100;
    this.flushIntervalSeconds = opts.flushIntervalSeconds ?? 5.0;
    this.timeoutSeconds = opts.timeoutSeconds ?? 10.0;
  }

  export(_span: Span): void {
    throw new Error(
      "HTTPExporter ships with Peekr Cloud (Phase 1). " +
      "Use JSONLExporter today; see " +
      "https://github.com/ashwanijha04/peekr/discussions for the waitlist.",
    );
  }
}
