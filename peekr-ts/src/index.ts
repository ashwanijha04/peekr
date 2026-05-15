/**
 * peekr — TypeScript SDK
 *
 * Zero-config observability for AI agents. Auto-instruments the OpenAI and
 * Anthropic Node.js SDKs and writes one JSON-Lines span per call to disk.
 * The Python `peekr` CLI and HTML dashboard read the same file unchanged —
 * the JSONL schema is the cross-runtime contract.
 *
 *   import { instrument, wrap, trace } from "@peekr/sdk";
 *   import OpenAI from "openai";
 *
 *   instrument({ jsonlPath: "./traces.jsonl" });
 *   const openai = wrap(new OpenAI());
 *
 *   const fetchUser = trace(async (id: number) => db.get(id));
 *
 *   // Every openai.chat.completions.create call and every fetchUser call
 *   // appears as a span in ./traces.jsonl. Run:
 *   //   pip install peekr && peekr dashboard traces.jsonl
 *   // to get the full HTML observability dashboard.
 */

export { instrument, isInstrumented, _resetForTests } from "./instrument.js";
export type { InstrumentOptions } from "./instrument.js";

export { wrap } from "./wrap.js";
export { wrapOpenAI } from "./wrappers/openai.js";
export { wrapAnthropic } from "./wrappers/anthropic.js";

export { trace } from "./trace.js";
export type { TraceOptions } from "./trace.js";

export {
  withSpan,
  createDetachedSpan,
  finishDetachedSpan,
  getCurrentSpan,
  withSession,
  getSession,
  addExporter,
  clearExporters,
  exportSpan,
} from "./context.js";
export type { SessionFields } from "./context.js";

export { JSONLExporter, ConsoleExporter } from "./exporters.js";

export { Span, hexId, unixSeconds } from "./span.js";
export type { Exporter, SpanRecord, SpanStatus } from "./span.js";
