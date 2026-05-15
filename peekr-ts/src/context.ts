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
  user_id?: string;
  session_id?: string;
}
const sessionAls = new AsyncLocalStorage<SessionFields>();

export function getCurrentSpan(): Span | null {
  return als.getStore()?.span ?? null;
}

export function getSession(): SessionFields | null {
  return sessionAls.getStore() ?? null;
}

/** Start a session. All spans created inside the callback inherit user_id / session_id. */
export function withSession<T>(fields: SessionFields, fn: () => T): T {
  const merged = { ...(sessionAls.getStore() ?? {}), ...fields };
  if (!merged.session_id) merged.session_id = hexId();
  return sessionAls.run(merged, fn);
}

function buildChildSpan(name: string): { span: Span; parentFrame: Frame | null } {
  const parentFrame = als.getStore() ?? null;
  const trace_id = parentFrame?.span.trace_id ?? hexId();
  const parent_id = parentFrame?.span.span_id ?? null;
  const span = new Span({ name, trace_id, parent_id });

  const sess = getSession();
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
