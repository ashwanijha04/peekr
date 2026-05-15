/**
 * OpenAI client wrapper.
 *
 *   import OpenAI from "openai";
 *   import { wrapOpenAI } from "@peekr/sdk";
 *   const openai = wrapOpenAI(new OpenAI());
 *
 * Returns a Proxy around the client that intercepts
 * `client.chat.completions.create` and records:
 *   - span name: "openai.chat.completions"
 *   - attributes.model
 *   - attributes.input  (JSON-stringified messages, truncated)
 *   - attributes.output (assistant content, truncated)
 *   - attributes.tokens_input / tokens_output / tokens_total
 *   - status / error
 *
 * Streaming is detected via `stream: true`. We inject
 * `stream_options.include_usage: true` so the final chunk carries token
 * usage, and wrap the async iterable to capture it.
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
  if (typeof u["prompt_tokens"] === "number") span.attributes["tokens_input"] = u["prompt_tokens"];
  if (typeof u["completion_tokens"] === "number") span.attributes["tokens_output"] = u["completion_tokens"];
  if (typeof u["total_tokens"] === "number") span.attributes["tokens_total"] = u["total_tokens"];
}

function captureChoiceContent(span: Span, choices: unknown): void {
  if (!Array.isArray(choices) || choices.length === 0) return;
  const first = choices[0] as { message?: { content?: unknown } };
  const content = first?.message?.content;
  if (typeof content === "string" && content.length > 0) {
    span.attributes["output"] = truncate(content);
  }
}

interface ChatCreateParams {
  model?: string;
  messages?: unknown[];
  stream?: boolean;
  stream_options?: { include_usage?: boolean } & Record<string, unknown>;
  [k: string]: unknown;
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
      for await (const chunk of orig) {
        const c = chunk as Record<string, unknown>;
        if (c?.["usage"]) captureUsage(span, c["usage"]);
        yield chunk;
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

function wrapChatCreate(
  originalCreate: (...args: unknown[]) => unknown,
  boundTo: unknown,
): (...args: unknown[]) => unknown {
  return function (this: unknown, ...args: unknown[]): unknown {
    const params = (args[0] ?? {}) as ChatCreateParams;
    const isStreaming = params.stream === true;

    if (isStreaming) {
      // Streaming: span outlives this call — use a detached span. The
      // iteration handler in `wrapStreamingResponse` finishes/exports it.
      const span = createDetachedSpan("openai.chat.completions");
      span.attributes["model"] = params.model ?? "unknown";
      if (Array.isArray(params.messages)) {
        span.attributes["input"] = truncate(JSON.stringify(params.messages));
      }
      const finalArgs = [
        { ...params, stream_options: { ...(params.stream_options ?? {}), include_usage: true } },
        ...args.slice(1),
      ];
      let result: unknown;
      try {
        result = originalCreate.apply(boundTo, finalArgs);
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

    // Non-streaming: scoped span via withSpan, so any peekr calls made by
    // hooks/callbacks during this await correctly nest underneath.
    return withSpan("openai.chat.completions", (span) => {
      span.attributes["model"] = params.model ?? "unknown";
      if (Array.isArray(params.messages)) {
        span.attributes["input"] = truncate(JSON.stringify(params.messages));
      }
      const result = originalCreate.apply(boundTo, args);
      if (result instanceof Promise) {
        return result.then((value: unknown) => {
          if (value && typeof value === "object") {
            const v = value as Record<string, unknown>;
            captureUsage(span, v["usage"]);
            captureChoiceContent(span, v["choices"]);
          }
          return value;
        });
      }
      return result;
    });
  };
}

function wrapChat(chat: unknown): unknown {
  if (chat === null || typeof chat !== "object") return chat;
  return new Proxy(chat as object, {
    get(target, prop, receiver) {
      if (prop === "completions") {
        return wrapCompletions(Reflect.get(target, prop, receiver), target);
      }
      return Reflect.get(target, prop, receiver);
    },
  });
}

function wrapCompletions(completions: unknown, boundTo: unknown): unknown {
  if (completions === null || typeof completions !== "object") return completions;
  return new Proxy(completions as object, {
    get(target, prop, receiver) {
      const value = Reflect.get(target, prop, receiver);
      if (prop === "create" && typeof value === "function") {
        return wrapChatCreate(value as (...a: unknown[]) => unknown, boundTo);
      }
      return value;
    },
  });
}

export function wrapOpenAI<T extends object>(client: T): T {
  if ((client as { __peekr_wrapped?: boolean }).__peekr_wrapped) {
    return client;
  }
  return new Proxy(client, {
    get(target, prop, receiver) {
      if (prop === "__peekr_wrapped") return true;
      if (prop === "chat") {
        return wrapChat(Reflect.get(target, prop, receiver));
      }
      return Reflect.get(target, prop, receiver);
    },
  }) as T;
}
