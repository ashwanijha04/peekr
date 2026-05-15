/**
 * `trace()` wraps any sync or async function so each invocation creates a
 * child span. Mirrors Python's `@peekr.trace` decorator.
 *
 *   const search = trace(async (query: string) => { ... });
 *   const calc   = trace({ name: "tool.calculator" }, (expr: string) => eval(expr));
 *
 * Captured automatically:
 *   - duration_ms
 *   - status (ok/error) + error message
 *   - input  (JSON-serialised positional args, truncated)
 *   - output (JSON-serialised return value, truncated)
 *
 * Opt out of IO capture with `{ captureIO: false }` for functions handling
 * secrets or large payloads.
 */

import { withSpan } from "./context.js";

export interface TraceOptions {
  name?: string;
  captureIO?: boolean;
}

const TRUNCATE = 1000;

function truncate(s: string): string {
  return s.length > TRUNCATE ? s.slice(0, TRUNCATE) + "…" : s;
}

function safeStringify(v: unknown): string {
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}

export function trace<F extends (...args: any[]) => any>(fn: F): F;
export function trace<F extends (...args: any[]) => any>(opts: TraceOptions, fn: F): F;
export function trace<F extends (...args: any[]) => any>(
  optsOrFn: TraceOptions | F,
  maybeFn?: F,
): F {
  const fn = (typeof optsOrFn === "function" ? optsOrFn : maybeFn) as F;
  if (!fn) throw new TypeError("trace(...) requires a function");
  const opts: TraceOptions =
    typeof optsOrFn === "function" ? {} : (optsOrFn as TraceOptions);
  const captureIO = opts.captureIO !== false;
  const spanName = opts.name ?? fn.name ?? "anonymous";

  const wrapped = function (this: unknown, ...args: unknown[]): unknown {
    return withSpan(spanName, (span) => {
      if (captureIO) span.attributes["input"] = truncate(safeStringify(args));
      const result = fn.apply(this, args);
      if (result instanceof Promise) {
        return result.then((value: unknown) => {
          if (captureIO) span.attributes["output"] = truncate(safeStringify(value));
          return value;
        });
      }
      if (captureIO) span.attributes["output"] = truncate(safeStringify(result));
      return result;
    });
  } as unknown as F;

  return wrapped;
}
