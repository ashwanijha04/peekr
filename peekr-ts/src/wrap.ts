/**
 * `wrap(client)` — generic entry point. Detects which SDK the client comes
 * from and dispatches to the right wrapper. Falls back to a no-op (returns
 * the client unchanged) if we don't recognise it, so users always get back
 * something usable.
 *
 *   import { wrap } from "@peekr/sdk";
 *   import OpenAI from "openai";
 *   import Anthropic from "@anthropic-ai/sdk";
 *   const openai = wrap(new OpenAI());
 *   const claude = wrap(new Anthropic());
 */

import { wrapOpenAI } from "./wrappers/openai.js";
import { wrapAnthropic } from "./wrappers/anthropic.js";

function looksLikeOpenAI(client: unknown): boolean {
  if (client === null || typeof client !== "object") return false;
  const c = client as Record<string, unknown>;
  const chat = c["chat"];
  if (chat === null || typeof chat !== "object") return false;
  const completions = (chat as Record<string, unknown>)["completions"];
  if (completions === null || typeof completions !== "object") return false;
  return typeof (completions as Record<string, unknown>)["create"] === "function";
}

function looksLikeAnthropic(client: unknown): boolean {
  if (client === null || typeof client !== "object") return false;
  const c = client as Record<string, unknown>;
  const messages = c["messages"];
  if (messages === null || typeof messages !== "object") return false;
  return typeof (messages as Record<string, unknown>)["create"] === "function";
}

export function wrap<T extends object>(client: T): T {
  if (looksLikeOpenAI(client)) return wrapOpenAI(client);
  if (looksLikeAnthropic(client)) return wrapAnthropic(client);
  // Unknown shape — return unchanged.
  return client;
}
