/**
 * Span data model.
 *
 * Schema is wire-compatible with peekr-py (`peekr/span.py`). The JSONL files
 * written by this SDK are consumed unchanged by the Python `peekr` CLI and
 * the HTML dashboard. If you change a field name here, change it in the
 * Python Span dataclass too.
 *
 * Times are unix-seconds floats with millisecond precision, matching
 * Python's `time.time()` so timestamps from both runtimes interleave
 * correctly when sharing a traces file.
 */

import { randomUUID } from "node:crypto";

export type SpanStatus = "ok" | "error";

export interface SpanRecord {
  name: string;
  trace_id: string;
  span_id: string;
  parent_id: string | null;
  start_time: number;
  end_time: number | null;
  attributes: Record<string, unknown>;
  status: SpanStatus;
  duration_ms: number | null;
  // First-class so the hosted backend can route + index without JSON
  // extraction. Wire-compatible with peekr-py (`peekr/span.py`).
  tenant_id: string | null;
  retention_class: string | null;
}

export class Span {
  readonly name: string;
  readonly trace_id: string;
  readonly span_id: string;
  readonly parent_id: string | null;
  readonly start_time: number;
  end_time: number | null = null;
  attributes: Record<string, unknown> = {};
  status: SpanStatus = "ok";
  tenant_id: string | null = null;
  retention_class: string | null = null;

  constructor(params: {
    name: string;
    trace_id: string;
    parent_id?: string | null;
    span_id?: string;
    start_time?: number;
    tenant_id?: string | null;
    retention_class?: string | null;
  }) {
    this.name = params.name;
    this.trace_id = params.trace_id;
    this.parent_id = params.parent_id ?? null;
    this.span_id = params.span_id ?? hexId();
    this.start_time = params.start_time ?? unixSeconds();
    this.tenant_id = params.tenant_id ?? null;
    this.retention_class = params.retention_class ?? null;
  }

  finish(): void {
    if (this.end_time === null) {
      this.end_time = unixSeconds();
    }
  }

  get duration_ms(): number | null {
    if (this.end_time === null) return null;
    return (this.end_time - this.start_time) * 1000;
  }

  toRecord(): SpanRecord {
    return {
      name: this.name,
      trace_id: this.trace_id,
      span_id: this.span_id,
      parent_id: this.parent_id,
      start_time: this.start_time,
      end_time: this.end_time,
      attributes: this.attributes,
      status: this.status,
      duration_ms: this.duration_ms,
      tenant_id: this.tenant_id,
      retention_class: this.retention_class,
    };
  }
}

/** Hex span/trace id — matches Python's `uuid.uuid4().hex` (32 chars, no dashes). */
export function hexId(): string {
  return randomUUID().replaceAll("-", "");
}

/** Unix seconds float (millisecond precision), matches Python's `time.time()`. */
export function unixSeconds(): number {
  return Date.now() / 1000;
}

/** Exporter is anything with an `.export(span)` method, sync or async. */
export interface Exporter {
  export(span: Span): void | Promise<void>;
}
