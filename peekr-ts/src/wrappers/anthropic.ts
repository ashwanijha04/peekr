/**
 * Anthropic client wrapper.
 *
 *   import Anthropic from "@anthropic-ai/sdk";
 *   import { wrapAnthropic } from "@peekr/sdk";
 *   const anthropic = wrapAnthropic(new Anthropic());
 *
 * Intercepts `client.messages.create` and records:
 *   - span name: "anthropic.messages"
 *   - attributes.model
 *   - attributes.input  (system prompt + messages, truncated)
 *   - attributes.output (first text block of the response, truncated)
 *   - attributes.tokens_input / tokens_output / tokens_total
 *   - status / error
 *
 * Streaming is handled when the user passes `stream: true` — the wrapper
 * captures the final usage event and closes the span when the stream ends.
 */

import { withSpan, createDetachedSpan, finishDetachedSpan } from "../context.js";
import type { Span } from "../span.js";

const TRUNCATE = 1000;

function truncate(s: string): string {
  return s.length > TRUNCATE ? s.slice(0, TRUNCATE) + "…" : s;
}

function captureUsage(span: Span, usage: unknown): void {
  if (!usage || typeof usage !== "object") return;
  const u = usage as Record<string, unknown>;
  const inTok = typeof u["input_tokens"] === "number" ? u["input_tokens"] : undefined;
  const outTok = typeof u["output_tokens"] === "number" ? u["output_tokens"] : undefined;
  if (inTok !== undefined) span.attributes["tokens_input"] = inTok;
  if (outTok !== undefined) span.attributes["tokens_output"] = outTok;
  if (inTok !== undefined && outTok !== undefined) {
    span.attributes["tokens_total"] = inTok + outTok;
  }
}

function captureResponseText(span: Span, content: unknown): void {
  if (!Array.isArray(content) || content.length === 0) return;
  const first = content[0] as { type?: string; text?: unknown };
  if (first?.type === "text" && typeof first.text === "string" && first.text.length > 0) {
    span.attributes["output"] = truncate(first.text);
  }
}

interface MessagesCreateParams {
  model?: string;
  messages?: unknown[];
  system?: string | unknown;
  stream?: boolean;
  [k: string]: unknown;
}

function captureInput(span: Span, params: MessagesCreateParams): void {
  span.attributes["model"] = params.model ?? "unknown";
  const inputForCapture: Record<string, unknown> = {};
  if (params.system !== undefined) inputForCapture["system"] = params.system;
  if (Array.isArray(params.messages)) inputForCapture["messages"] = params.messages;
  span.attributes["input"] = truncate(JSON.stringify(inputForCapture));
}

function wrapStreamingResponse(stream: unknown, span: Span): unknown {
  if (stream === null || typeof stream !== "object") {
    finishDetachedSpan(span);
    return stream;
  }
  const orig = stream as AsyncIterable<unknown> & Record<string, unknown>;
  if (typeof orig[Symbol.asyncIterator] !== "function") {
    finishDetachedSpan(span);
    return stream;
  }

  async function* generator(): AsyncGenerator<unknown> {
    try {
      for await (const event of orig) {
        const e = event as Record<string, unknown>;
        const msg = e?.["message"] as Record<string, unknown> | undefined;
        if (msg?.["usage"]) captureUsage(span, msg["usage"]);
        const delta = e?.["delta"] as Record<string, unknown> | undefined;
        const usage = e?.["usage"] ?? delta?.["usage"];
        if (usage) captureUsage(span, usage);
        yield event;
      }
    } catch (err) {
      span.status = "error";
      span.attributes["error"] = err instanceof Error ? err.message : String(err);
      throw err;
    } finally {
      finishDetachedSpan(span);
    }
  }

  const gen = generator();
  return new Proxy(orig, {
    get(target, prop, receiver) {
      if (prop === Symbol.asyncIterator) return () => gen;
      return Reflect.get(target, prop, receiver);
    },
  });
}

function wrapMessagesCreate(
  originalCreate: (...args: unknown[]) => unknown,
  boundTo: unknown,
): (...args: unknown[]) => unknown {
  return function (this: unknown, ...args: unknown[]): unknown {
    const params = (args[0] ?? {}) as MessagesCreateParams;
    const isStreaming = params.stream === true;

    if (isStreaming) {
      const span = createDetachedSpan("anthropic.messages");
      captureInput(span, params);
      let result: unknown;
      try {
        result = originalCreate.apply(boundTo, args);
      } catch (err) {
        span.status = "error";
        span.attributes["error"] = err instanceof Error ? err.message : String(err);
        finishDetachedSpan(span);
        throw err;
      }
      if (result instanceof Promise) {
        return result.then(
          (value) => wrapStreamingResponse(value, span),
          (err: unknown) => {
            span.status = "error";
            span.attributes["error"] = err instanceof Error ? err.message : String(err);
            finishDetachedSpan(span);
            throw err;
          },
        );
      }
      return wrapStreamingResponse(result, span);
    }

    return withSpan("anthropic.messages", (span) => {
      captureInput(span, params);
      const result = originalCreate.apply(boundTo, args);
      if (result instanceof Promise) {
        return result.then((value: unknown) => {
          if (value && typeof value === "object") {
            const v = value as Record<string, unknown>;
            captureUsage(span, v["usage"]);
            captureResponseText(span, v["content"]);
          }
          return value;
        });
      }
      if (result && typeof result === "object") {
        const v = result as Record<string, unknown>;
        captureUsage(span, v["usage"]);
        captureResponseText(span, v["content"]);
      }
      return result;
    });
  };
}

function wrapMessages(messages: unknown, boundTo: unknown): unknown {
  if (messages === null || typeof messages !== "object") return messages;
  return new Proxy(messages as object, {
    get(target, prop, receiver) {
      const value = Reflect.get(target, prop, receiver);
      if (prop === "create" && typeof value === "function") {
        return wrapMessagesCreate(value as (...a: unknown[]) => unknown, boundTo);
      }
      return value;
    },
  });
}

export function wrapAnthropic<T extends object>(client: T): T {
  if ((client as { __peekr_wrapped?: boolean }).__peekr_wrapped) {
    return client;
  }
  return new Proxy(client, {
    get(target, prop, receiver) {
      if (prop === "__peekr_wrapped") return true;
      if (prop === "messages") {
        return wrapMessages(Reflect.get(target, prop, receiver), target);
      }
      return Reflect.get(target, prop, receiver);
    },
  }) as T;
}
