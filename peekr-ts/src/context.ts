/**
 * Span hierarchy propagation.
 *
 * Mirrors `peekr/context.py`: Python uses `contextvars.ContextVar`; we use
 * Node's `AsyncLocalStorage`. Both propagate through `async/await` chains
 * without manual threading.
 *
 * The primary primitive is `withSpan(name, fn)` — it uses `als.run()` so the
 * span context is correctly scoped to the callback. We deliberately avoid
 * `als.enterWith()` because of a long-standing pitfall: a child's
 * `enterWith()` leaks back into the parent's continuation after `await`,
 * producing wrong parent_id links. `als.run()` is scoped and correct.
 *
 * For cases where the span outlives the callback (notably streaming SDK
 * responses) we expose `createDetachedSpan(name)`: it inherits parent/trace
 * from the current ALS store at creation but does not establish a new
 * scope, so nothing nests under it. Callers must call `finishDetachedSpan()`
 * when the work is done.
 */

import { AsyncLocalStorage } from "node:async_hooks";
import { Span, hexId } from "./span.js";
import type { Exporter } from "./span.js";

interface Frame {
  span: Span;
  parent: Frame | null;
}

const als = new AsyncLocalStorage<Frame>();

export interface SessionFields {
  /** End-user identifier (B2C). Stored in span.attributes.user_id. */
  user_id?: string;
  /** Conversation/session correlation id. */
  session_id?: string;
  /**
   * Customer org identifier (B2B). First-class on the span — distinct from
   * user_id. Required for multi-tenant routing in Peekr Cloud.
   */
  tenant_id?: string;
  /**
   * Storage tier hint (e.g. "default" | "short" | "long" | "pii").
   * OSS stores it; Peekr Cloud enforces it.
   */
  retention_class?: string;
}
const sessionAls = new AsyncLocalStorage<SessionFields>();

// Process-wide defaults set by instrument(). Lower priority than withSession().
let _defaultTenantId: string | null = null;
let _defaultRetentionClass: string | null = null;

export function setProcessDefaults(opts: {
  tenant_id?: string | null;
  retention_class?: string | null;
}): void {
  if (opts.tenant_id !== undefined) _defaultTenantId = opts.tenant_id;
  if (opts.retention_class !== undefined) _defaultRetentionClass = opts.retention_class;
}

export function getCurrentSpan(): Span | null {
  return als.getStore()?.span ?? null;
}

export function getSession(): SessionFields | null {
  return sessionAls.getStore() ?? null;
}

/** Start a session. All spans created inside the callback inherit the fields. */
export function withSession<T>(fields: SessionFields, fn: () => T): T {
  const merged = { ...(sessionAls.getStore() ?? {}), ...fields };
  if (!merged.session_id) merged.session_id = hexId();
  return sessionAls.run(merged, fn);
}

function resolveTenantId(sess: SessionFields | null): string | null {
  return (
    sess?.tenant_id ??
    _defaultTenantId ??
    process.env.PEEKR_TENANT_ID ??
    null
  );
}

function resolveRetentionClass(sess: SessionFields | null): string | null {
  return (
    sess?.retention_class ??
    _defaultRetentionClass ??
    process.env.PEEKR_RETENTION_CLASS ??
    null
  );
}

function buildChildSpan(name: string): { span: Span; parentFrame: Frame | null } {
  const parentFrame = als.getStore() ?? null;
  const trace_id = parentFrame?.span.trace_id ?? hexId();
  const parent_id = parentFrame?.span.span_id ?? null;
  const sess = getSession();
  const span = new Span({
    name,
    trace_id,
    parent_id,
    tenant_id: resolveTenantId(sess),
    retention_class: resolveRetentionClass(sess),
  });

  if (sess?.session_id) span.attributes["session_id"] = sess.session_id;
  if (sess?.user_id) span.attributes["user_id"] = sess.user_id;

  return { span, parentFrame };
}

/**
 * Run `fn(span)` inside a new child-span scope. The span is automatically
 * finished and exported whether `fn` returns synchronously, returns a
 * Promise that resolves, or throws/rejects. The return value of `fn` is
 * returned through unchanged.
 *
 * Use this for almost everything. For long-lived spans that outlive the
 * call (e.g. wrapping an async iterator returned from a streaming API),
 * use `createDetachedSpan` + `finishDetachedSpan` instead.
 */
export function withSpan<T>(name: string, fn: (span: Span) => T): T {
  const { span, parentFrame } = buildChildSpan(name);
  const newFrame: Frame = { span, parent: parentFrame };

  return als.run(newFrame, () => {
    let result: T;
    try {
      result = fn(span);
    } catch (err) {
      span.status = "error";
      span.attributes["error"] = err instanceof Error ? err.message : String(err);
      span.finish();
      exportSpan(span);
      throw err;
    }
    if (result instanceof Promise) {
      return result.then(
        (value: unknown) => {
          span.finish();
          exportSpan(span);
          return value;
        },
        (err: unknown) => {
          span.status = "error";
          span.attributes["error"] = err instanceof Error ? err.message : String(err);
          span.finish();
          exportSpan(span);
          throw err;
        },
      ) as T;
    }
    span.finish();
    exportSpan(span);
    return result;
  });
}

/**
 * Create a span that inherits parent/trace from the current ALS store but
 * does NOT push a new scope (so nothing implicit nests under it).
 *
 * Used by the streaming paths in the OpenAI and Anthropic wrappers, where
 * the span outlives the wrapper function: the wrapper returns the wrapped
 * iterable, the user iterates it, and the wrapper's iteration handler
 * calls `finishDetachedSpan(span)` when iteration completes.
 */
export function createDetachedSpan(name: string): Span {
  return buildChildSpan(name).span;
}

export function finishDetachedSpan(span: Span): void {
  span.finish();
  exportSpan(span);
}

// -----------------------------------------------------------------------
// Exporter registry — populated by `instrument()` / `addExporter()`
// -----------------------------------------------------------------------

const _exporters: Exporter[] = [];

export function addExporter(exp: Exporter): void {
  _exporters.push(exp);
}

export function clearExporters(): void {
  _exporters.length = 0;
}

export function exportSpan(span: Span): void {
  for (const exp of _exporters) {
    try {
      const result = exp.export(span);
      if (result instanceof Promise) {
        result.catch(() => undefined);
      }
    } catch {
      /* exporters must never throw out of here */
    }
  }
}
