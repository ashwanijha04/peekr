/**
 * Top-level entry point — mirrors `peekr.instrument()` in Python.
 *
 * Sets up storage exporters. Unlike the Python version, this SDK does NOT
 * auto-patch SDK packages at the module level (ESM imports are immutable);
 * instead, users opt in by passing their client through `wrap()`.
 */

import { JSONLExporter, ConsoleExporter } from "./exporters.js";
import { addExporter, clearExporters, setProcessDefaults } from "./context.js";
import type { Exporter } from "./span.js";

export interface InstrumentOptions {
  /** Print each completed span to stderr. Default: true. */
  console?: boolean;
  /** "jsonl" | "none". SQLite is read-only here; use the Python exporter for SQLite. Default: "jsonl". */
  storage?: "jsonl" | "none";
  /** JSONL output path. Default: "./traces.jsonl". */
  jsonlPath?: string;
  /** Optional list of custom exporters to register on top of the defaults. */
  exporters?: Exporter[];
  /** If true, reset the exporter registry before adding new exporters. Default: false. */
  reset?: boolean;
  /**
   * Process-wide default tenant (B2B customer org). Overridden per-request
   * by withSession({ tenant_id }). Falls back to env PEEKR_TENANT_ID.
   */
  tenant_id?: string;
  /**
   * Process-wide default retention class. Falls back to env
   * PEEKR_RETENTION_CLASS.
   */
  retention_class?: string;
}

let _instrumented = false;

export function instrument(options: InstrumentOptions = {}): void {
  const {
    console: useConsole = true,
    storage = "jsonl",
    jsonlPath = "./traces.jsonl",
    exporters = [],
    reset = false,
    tenant_id,
    retention_class,
  } = options;

  if (reset) clearExporters();

  setProcessDefaults({
    tenant_id: tenant_id ?? null,
    retention_class: retention_class ?? null,
  });

  if (useConsole) addExporter(new ConsoleExporter());
  if (storage === "jsonl") addExporter(new JSONLExporter(jsonlPath));
  for (const e of exporters) addExporter(e);

  _instrumented = true;
}

/** For tests: whether instrument() has been called this process. */
export function isInstrumented(): boolean {
  return _instrumented;
}

/** For tests: reset state. */
export function _resetForTests(): void {
  _instrumented = false;
  clearExporters();
}
