// main.ts — free-model-aggregator-02 entry: the best LIVE free-model aggregator.
//
// One server that speaks the OpenAI Chat Completions API and the Anthropic
// Messages API (JSON and streaming) and answers every prompt with a REAL
// completion from one of the fourteen live free OpenRouter models, dialled
// through the grader's gateway on the trial's key. No simulated provider
// anywhere: every answer the server returns is a gateway answer relayed
// back in the wire the client used, with the dialled model id put back on
// every answer and every streamed chunk.
//
// How it works:
//   * At start it reads the brief (MAKEFUN_PROMPT_FILE): the fourteen free
//     models (gateway id, house rating in milli-points, tools/vision/JSON
//     support, context length) and the six prompts' manifest, in the order
//     the env sends them.
//   * It builds the catalog document (one provider, openrouter, exactly
//     those fourteen ids) and registers port + catalog with the env.
//   * For each prompt it derives the requirements from the manifest row
//     (arrival order) and dials the best-rated free model that meets them
//     on the gateway's Chat Completions route — an Anthropic-shaped request
//     is translated into a Chat Completions body first, and the gateway's
//     answer is translated back.
//   * A refusal (429/5xx) is retried or failed over to the next-best model
//     that meets the prompt; recovered refusals are reported on the
//     response in the x-makefun-upstream-failures header, so the failover
//     column can see them.
//   * It polls GET $MAKEFUN_ENV_URL/work and exits 0 when the env says
//     the trial is done.
//
// Routing, in one line: best rating first, per prompt requirements.
// Quality is 60% of the score and reads the model that really served each
// prompt, so the first dial is always the best-rated model that can serve
// the prompt: inkling-2-small (1475) for plain and tools prompts,
// nemotron-3.5-super-vl (1303) for JSON-mode prompts, the best JSON-capable
// free model in the brief.
//
// Determinism: every choice is a function of the brief, the requests and
// the gateway's answers only. No wall clock and no unseeded randomness
// reaches any request, so a replay leaves the same state export: refusal
// counts are per-model counters kept for the trial, candidate order is
// rating then id, and the model lists carry fixed timestamps.
//
// Run:  node main.ts            (serve a trial)
//       node main.ts --list     (print the catalog document and exit 0)

import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { readFileSync } from "node:fs";

// ---------------------------------------------------------------- constants

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** One gateway dial may take this long before it counts as refused. */
const DIAL_TIMEOUT_MS = 15000;
/** The most gateway dials one prompt may cost (tie-break is fewest calls).
 * Six prompts at six dials is 36 calls, still well under the trial's
 * 60-call cap, and a prompt is worth more served at six dials than lost
 * at four: served and quality are 75% of the score between them. */
const MAX_ATTEMPTS_PER_PROMPT = 6;
/** A model that refused this many times this trial is skipped. */
const MAX_REFUSALS_PER_MODEL = 2;
/** The failures header carries at most this many rows. */
const MAX_FAILURE_ROWS = 8;
/** A fixed epoch second for every listed model's `created`. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`. */
const FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z";
/** The Anthropic version header the gateway's Messages lane expects. */
const ANTHROPIC_VERSION = "2023-06-01";

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? "").replace(/\/+$/, "");
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? "";
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? "").replace(/\/+$/, "");
/** The trial's dummy key for the gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? "";

// ---------------------------------------------------------------- small helpers

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function asString(v: unknown): string {
  return typeof v === "string" ? v : "";
}

function asNumber(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/** Extract the JSON document whose schema marker appears in the brief text.
 * Each document starts with {"schema":"<schema>", ...} on its own line, so
 * the document's opening brace is the one right before the marker. */
function extractJsonDoc(text: string, schema: string): Record<string, unknown> | null {
  const marker = `{"schema":"${schema}"`;
  const at = text.indexOf(marker);
  if (at === -1) return null;
  const open = at;
  let depth = 0;
  let inStr = false;
  let esc = false;
  for (let i = open; i < text.length; i++) {
    const ch = text[i];
    if (inStr) {
      if (esc) esc = false;
      else if (ch === "\\") esc = true;
      else if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') inStr = true;
    else if (ch === "{") depth++;
    else if (ch === "}") {
      depth--;
      if (depth === 0) {
        try {
          const v = JSON.parse(text.slice(open, i + 1));
          return isRecord(v) ? v : null;
        } catch {
          return null;
        }
      }
    }
  }
  return null;
}

// ---------------------------------------------------------------- brief

interface ModelRow {
  id: string;
  rating: number;
  context: number;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

interface PromptRow {
  id: string;
  api: "openai" | "anthropic";
  stream: boolean;
  weight: number;
  requires: Requirements;
}

interface Requirements {
  tools: boolean;
  vision: boolean;
  json: boolean;
}

interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
}

function readBrief(): Brief {
  const empty: Brief = { models: [], prompts: [] };
  if (PROMPT_FILE === "") return empty;
  let text: string;
  try {
    text = readFileSync(PROMPT_FILE, "utf8");
  } catch {
    return empty;
  }
  const modelsDoc = extractJsonDoc(text, "free-model-aggregator-02/models/v1");
  const promptsDoc = extractJsonDoc(text, "free-model-aggregator-02/prompts/v1");
  const models: ModelRow[] = [];
  if (modelsDoc && Array.isArray(modelsDoc.models)) {
    for (const raw of modelsDoc.models) {
      if (!isRecord(raw)) continue;
      const id = asString(raw.id);
      if (id === "" || raw.free !== true) continue;
      models.push({
        id,
        rating: asNumber(raw.rating_milli) ?? 0,
        context: asNumber(raw.context_length) ?? 0,
        tools: raw.tools === true,
        vision: raw.vision === true,
        json: raw.json === true,
      });
    }
  }
  const prompts: PromptRow[] = [];
  if (promptsDoc && Array.isArray(promptsDoc.prompts)) {
    for (const raw of promptsDoc.prompts) {
      if (!isRecord(raw)) continue;
      const id = asString(raw.id);
      if (id === "") continue;
      const req = isRecord(raw.requires) ? raw.requires : {};
      prompts.push({
        id,
        api: asString(raw.api) === "anthropic" ? "anthropic" : "openai",
        stream: raw.stream === true,
        weight: asNumber(raw.weight) ?? 1,
        requires: {
          tools: req.tools === true,
          vision: req.vision === true,
          json: req.json === true,
        },
      });
    }
  }
  return { models, prompts };
}

const brief = readBrief();

// ---------------------------------------------------------------- catalog

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number;
}

interface Catalog {
  providers: Array<{
    id: string;
    base_url: string;
    key_env: string;
    wire: string;
    models: CatalogModel[];
  }>;
}

/** The one catalog document: one provider, openrouter, exactly the brief's
 * fourteen free models, once each. The same object is printed by --list
 * and sent in the register body, so the two are byte for byte the same. */
function buildCatalog(models: ModelRow[]): Catalog {
  return {
    providers: [
      {
        id: "openrouter",
        base_url: "https://openrouter.ai/api/v1",
        key_env: "OPENROUTER_API_KEY",
        wire: "openai",
        models: models.map((m) => ({ id: m.id, free: true, context_length: m.context })),
      },
    ],
  };
}

const catalog: Catalog = buildCatalog(brief.models);

// ---------------------------------------------------------------- routing

/** Per-trial router state: how often each model has refused. A refusal
 * count, never a clock, decides when a model is skipped, so a replay
 * makes the same choices on the same gateway answers. */
const routerState = {
  refusals: new Map<string, number>(),
  /** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
  manifestIndex: 0,
};

function meetsRequirements(model: ModelRow, req: Requirements): boolean {
  if (req.tools && !model.tools) return false;
  if (req.vision && !model.vision) return false;
  if (req.json && !model.json) return false;
  return true;
}

/** The free models that can serve this prompt, best rating first, id
 * ascending as the deterministic tie-break, skipping models that have
 * already refused twice this trial. */
function candidatesFor(req: Requirements): ModelRow[] {
  return brief.models
    .filter((m) => meetsRequirements(m, req))
    .filter((m) => (routerState.refusals.get(m.id) ?? 0) < MAX_REFUSALS_PER_MODEL)
    .sort((a, b) => b.rating - a.rating || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

/** The requirements of the prompt now arriving: the brief's row by arrival
 * order, else the body's own signals. */
function requirementsFor(body: Record<string, unknown>): Requirements {
  const entry = brief.prompts[routerState.manifestIndex];
  routerState.manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Record<string, unknown>): Requirements {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === "image" || block.type === "image_url")) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === "json_object" || format === "json_schema",
  };
}

// ---------------------------------------------------------------- gateway dial + failover

interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

type DialOutcome =
  | { ok: true; model: string; response: Response; failures: FailureRow[] }
  | { ok: false; failures: FailureRow[] };

/** The gateway's Chat Completions route. OPENAI_BASE_URL is the
 * OpenAI-compatible root (`…/v1` on the trial network); a base that does
 * not already end in `/v1` gets one, so both spellings reach the route. */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === "") return "";
  return GATEWAY_URL.endsWith("/v1") ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The refusal's code from the gateway's error body, for the failures
 * trail: the provider's own code when it names one, else an http code. */
async function errorCodeOf(res: Response): Promise<string> {
  try {
    const text = await res.text();
    if (text === "") return `http_${res.status}`;
    const doc: unknown = JSON.parse(text);
    if (isRecord(doc) && isRecord(doc.error)) {
      if (typeof doc.error.code === "string" && doc.error.code !== "") return doc.error.code;
      if (typeof doc.error.type === "string" && doc.error.type !== "") return doc.error.type;
    }
    if (isRecord(doc) && typeof doc.code === "string" && doc.code !== "") return doc.code;
    return `http_${res.status}`;
  } catch {
    return `http_${res.status}`;
  }
}

/** Dial one prompt on the gateway until a model answers it or the attempt
 * budget is spent. The first dial is always the best-rated model that can
 * serve the prompt; a 429 moves on to the next candidate at once, a 5xx
 * is retried on the same model once, a 401 (no key pushed) or a 400
 * stops the prompt — no other model would answer it either. Every
 * refusal is recorded, in order, for the failures header. */
async function dialPrompt(openAIBody: Record<string, unknown>, req: Requirements): Promise<DialOutcome> {
  const failures: FailureRow[] = [];
  const ordered = candidatesFor(req);
  let attempts = 0;
  for (const model of ordered) {
    let triesOnModel = 0;
    while (triesOnModel < 2 && attempts < MAX_ATTEMPTS_PER_PROMPT) {
      attempts += 1;
      triesOnModel += 1;
      let res: Response | null = null;
      try {
        res = await fetch(chatCompletionsUrl(), {
          method: "POST",
          headers: {
            "content-type": "application/json",
            authorization: `Bearer ${GATEWAY_KEY}`,
            ...(openAIBody.stream === true ? { accept: "text/event-stream" } : {}),
          },
          body: JSON.stringify({ ...openAIBody, model: model.id }),
          signal: AbortSignal.timeout(DIAL_TIMEOUT_MS),
        });
      } catch {
        // The gateway never answered: a refusal like any other, retried
        // once on this model, then failed over.
        failures.push({ status: 0, code: "upstream_unreachable", model: model.id });
        routerState.refusals.set(model.id, (routerState.refusals.get(model.id) ?? 0) + 1);
        continue;
      }
      if (res.ok && res.body !== null) return { ok: true, model: model.id, response: res, failures };
      const status = res.status;
      const code = await errorCodeOf(res);
      failures.push({ status, code, model: model.id });
      routerState.refusals.set(model.id, (routerState.refusals.get(model.id) ?? 0) + 1);
      if (status === 401 || status === 400) return { ok: false, failures };
      if (status === 429) break; // rate-limited: try the next model at once
      // A 5xx is retried on this model once (the while loop), then the
      // next candidate takes it.
    }
    if (attempts >= MAX_ATTEMPTS_PER_PROMPT) break;
  }
  return { ok: false, failures };
}

/** The failures header's value, or null when no call refused: a JSON
 * array of at most 8 rows, in the order the calls refused. */
function failuresHeader(failures: FailureRow[]): string | null {
  if (failures.length === 0) return null;
  const rows = failures.slice(0, MAX_FAILURE_ROWS);
  const text = JSON.stringify(rows);
  if (text.length > 8192) return null;
  return text;
}

// ---------------------------------------------------------------- wire: errors and model lists

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders?: Record<string, string>): void {
  const text = JSON.stringify(body);
  res.writeHead(status, {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(text),
    ...extraHeaders,
  });
  res.end(text);
}

function openAIError(message: string, type: string): Record<string, unknown> {
  return { error: { message, type, code: null } };
}

function anthropicError(type: string, message: string): Record<string, unknown> {
  return { type: "error", error: { type, message } };
}

function sendError(res: ServerResponse, wire: "openai" | "anthropic", status: number, type: string, message: string, failures?: FailureRow[]): void {
  const header = failures !== undefined ? failuresHeader(failures) : null;
  const extra = header !== null ? { "x-makefun-upstream-failures": header } : undefined;
  if (wire === "anthropic") sendJson(res, status, anthropicError(type, message), extra);
  else sendJson(res, status, openAIError(message, type), extra);
}

/** GET /v1/models on an OpenAI-shaped request: the fourteen ids, in the
 * brief's order, with a fixed `created` so the document never moves. */
function openAIModelList(): Record<string, unknown> {
  return {
    object: "list",
    data: brief.models.map((m) => ({ id: m.id, object: "model", created: FIXED_OPENAI_CREATED, owned_by: "openrouter" })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header):
 * Anthropic's model list shape, same fourteen ids. */
function anthropicModelList(): Record<string, unknown> {
  return {
    data: brief.models.map((m) => ({ type: "model", id: m.id, display_name: m.id, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

// ---------------------------------------------------------------- wire: Anthropic <-> Chat Completions

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content.flatMap((b) => (isRecord(b) && b.type === "text" && typeof b.text === "string" ? [b.text] : [])).join("\n");
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Record<string, unknown>): Record<string, unknown> | null {
  if (block.type === "text" && typeof block.text === "string") return { type: "text", text: block.text };
  if (block.type === "image" && isRecord(block.source)) {
    const source = block.source;
    if (source.type === "base64" && typeof source.data === "string") {
      const media = typeof source.media_type === "string" ? source.media_type : "image/png";
      return { type: "image_url", image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === "url" && typeof source.url === "string") return { type: "image_url", image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== "" ? text : JSON.stringify(content);
  }
  return content === undefined ? "" : JSON.stringify(content);
}

/** A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Record<string, unknown> {
  if (isRecord(value)) return value;
  if (typeof value === "string" && value.trim() !== "") {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
function stopReasonOf(finish: unknown): string {
  if (finish === "length") return "max_tokens";
  if (finish === "tool_calls" || finish === "function_call") return "tool_use";
  if (finish === "content_filter") return "refusal";
  return "end_turn";
}

function numberOrZero(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that
 * names `model`: the system prompt becomes a system message, image blocks
 * become image_url data URLs, tool_use blocks become tool_calls,
 * tool_result blocks become tool messages, tools and tool_choice are
 * rewritten, and the output cap, temperature, top_p, stop sequences and
 * streaming pass through.
 */
function anthropicToOpenAI(body: Record<string, unknown>, model: string): Record<string, unknown> {
  const messages: Record<string, unknown>[] = [];
  const system = textOf(body.system);
  if (system !== "") messages.push({ role: "system", content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === "assistant";
    const content = raw.content;
    if (typeof content === "string") {
      messages.push({ role: assistant ? "assistant" : "user", content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Record<string, unknown>[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== "tool_use") continue;
        toolCalls.push({
          id: typeof block.id === "string" ? block.id : "",
          type: "function",
          function: {
            name: typeof block.name === "string" ? block.name : "",
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: "assistant",
        content: text !== "" ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Record<string, unknown>[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === "tool_result") {
        messages.push({
          role: "tool",
          tool_call_id: typeof block.tool_use_id === "string" ? block.tool_use_id : "",
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: "user", content: parts });
  }

  const out: Record<string, unknown> = { model, messages };
  if (typeof body.max_tokens === "number") out.max_tokens = body.max_tokens;
  if (typeof body.temperature === "number") out.temperature = body.temperature;
  if (typeof body.top_p === "number") out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== "string") return [];
      return [
        {
          type: "function",
          function: {
            name: raw.name,
            description: typeof raw.description === "string" ? raw.description : "",
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: "object", properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === "auto") out.tool_choice = "auto";
    else if (choice.type === "any") out.tool_choice = "required";
    else if (choice.type === "tool" && typeof choice.name === "string") {
      out.tool_choice = { type: "function", function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A Chat Completions answer as an Anthropic Messages object. `model` is the id the entry dialled. */
function openAIToAnthropic(body: Record<string, unknown>, model: string): Record<string, unknown> {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Record<string, unknown>[] = [];
  const text = typeof message.content === "string" ? message.content : textOf(message.content);
  if (text !== "") content.push({ type: "text", text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: "tool_use",
      id: typeof call.id === "string" ? call.id : "toolu_unknown",
      name: typeof call.function.name === "string" ? call.function.name : "",
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: "text", text: "" });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === "string" ? body.id : "aggregate"}`,
    type: "message",
    role: "assistant",
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** A Chat Completions answer with `model` set to the id the entry dialled: the one truthful claim. */
function openAIJsonWithModel(body: Record<string, unknown>, model: string): Record<string, unknown> {
  return { ...body, model };
}

// ---------------------------------------------------------------- wire: streaming

/** One SSE frame: `data: <json>\n\n`, or a bare `[DONE]`. */
function sseEvent(value: Record<string, unknown> | string): string {
  return typeof value === "string" ? `data: ${value}\n\n` : `data: ${JSON.stringify(value)}\n\n`;
}

async function writeSse(res: ServerResponse, value: Record<string, unknown> | string): Promise<boolean> {
  return new Promise((resolve) => res.write(sseEvent(value), () => resolve(true)));
}

/** One Anthropic SSE event: the event NAME line, then the data line. The
 * Messages wire names every event; a bare `data:` line carries no event
 * name and the stream fails the shape check. */
async function writeAnthropicEvent(res: ServerResponse, event: Record<string, unknown>): Promise<boolean> {
  const text = `event: ${asString(event.type)}\ndata: ${JSON.stringify(event)}\n\n`;
  return new Promise((resolve) => res.write(text, () => resolve(true)));
}

/**
 * An OpenAI SSE stream with `model` rewritten on every chunk to the id the
 * entry dialled: chunks pass through otherwise, `[DONE]` last.
 */
async function openAIStreamWithModel(upstream: Response, model: string, res: ServerResponse): Promise<void> {
  if (upstream.body === null) throw new Error("upstream stream missing");
  const reader = upstream.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split(/\r?\n\r?\n/);
      buffer = frames.pop() ?? "";
      for (const frame of frames) {
        const line = frame.trim();
        if (line === "") continue;
        if (line === "data: [DONE]") {
          // Handled by the caller.
          continue;
        }
        if (!line.startsWith("data: ")) continue;
        const text = line.slice("data: ".length);
        let chunk: unknown;
        try {
          chunk = JSON.parse(text);
        } catch {
          continue;
        }
        if (!isRecord(chunk)) continue;
        await writeSse(res, { ...chunk, model });
      }
    }
  } finally {
    reader.releaseLock();
  }
}

/** The mutable per-stream context the OpenAI→Anthropic event translator needs. */
interface StreamContext {
  res: ServerResponse;
  model: string;
  messageStarted: boolean;
  textBlockOpen: boolean;
  toolBlockOpen: boolean;
  textDone: boolean;
  toolDone: boolean;
  stopReason: string;
  inputTokens: number;
  outputTokens: number;
  streamFailed: boolean;
}

/** The Anthropic event stream the gateway's OpenAI stream becomes, on the
 * same model: message_start, content_block_start, content_block_delta,
 * content_block_stop, message_delta, message_stop — with usage held to
 * the final chunk, because the OpenAI stream only reports it there. */
async function anthropicEventsFromOpenAI(upstream: Response, model: string, res: ServerResponse): Promise<void> {
  const ctx: StreamContext = {
    res,
    model,
    messageStarted: false,
    textBlockOpen: false,
    toolBlockOpen: false,
    textDone: false,
    toolDone: false,
    stopReason: "end_turn",
    inputTokens: 0,
    outputTokens: 0,
    streamFailed: false,
  };

  const startEvents = (): Array<Record<string, unknown> | string> => [
    {
      type: "message_start",
      message: {
        id: `msg_${ctx.model}`,
        type: "message",
        role: "assistant",
        model: ctx.model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: ctx.inputTokens, output_tokens: ctx.outputTokens },
      },
    },
  ];

  const eventsOfChunk = (chunk: Record<string, unknown>): Array<Record<string, unknown> | string> => {
    const out: Array<Record<string, unknown> | string> = [];
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const delta = isRecord(choice.delta) ? choice.delta : {};
    const usage = isRecord(chunk.usage) ? chunk.usage : null;
    if (usage !== null) {
      if (typeof usage.prompt_tokens === "number") ctx.inputTokens = usage.prompt_tokens;
      if (typeof usage.completion_tokens === "number") ctx.outputTokens = usage.completion_tokens;
    }
    if (!ctx.messageStarted) {
      ctx.messageStarted = true;
      out.push(...startEvents());
    }
    if (choice.finish_reason !== undefined && choice.finish_reason !== null) {
      ctx.stopReason = stopReasonOf(choice.finish_reason);
    }
    const text = typeof delta.content === "string" ? delta.content : textOf(delta.content);
    if (text !== "") {
      if (!ctx.textBlockOpen) {
        out.push({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } });
        ctx.textBlockOpen = true;
      }
      out.push({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text } });
    }
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      if (!ctx.toolBlockOpen) {
        out.push({
          type: "content_block_start",
          index: 1,
          content_block: {
            type: "tool_use",
            id: typeof call.id === "string" ? call.id : "toolu_unknown",
            name: typeof call.function.name === "string" ? call.function.name : "",
            input: {},
          },
        });
        ctx.toolBlockOpen = true;
      }
      if (typeof call.function.arguments === "string" && call.function.arguments !== "") {
        out.push({
          type: "content_block_delta",
          index: 1,
          delta: { type: "input_json_delta", partial_json: call.function.arguments },
        });
      }
    }
    if (choice.finish_reason !== undefined && choice.finish_reason !== null) {
      if (ctx.textBlockOpen && !ctx.textDone) {
        out.push({ type: "content_block_stop", index: 0 });
        ctx.textDone = true;
      }
      if (ctx.toolBlockOpen && !ctx.toolDone) {
        out.push({ type: "content_block_stop", index: 1 });
        ctx.toolDone = true;
      }
      out.push({
        type: "message_delta",
        delta: { stop_reason: ctx.stopReason, stop_sequence: null },
        usage: { output_tokens: ctx.outputTokens },
      });
      out.push({ type: "message_stop" });
    }
    return out;
  };

  if (upstream.body === null) throw new Error("upstream stream missing");
  const reader = upstream.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const frames = buffer.split(/\r?\n\r?\n/);
      buffer = frames.pop() ?? "";
      for (const frame of frames) {
        const line = frame.trim();
        if (line === "" || line === "data: [DONE]") continue;
        if (!line.startsWith("data: ")) continue;
        let chunk: unknown;
        try {
          chunk = JSON.parse(line.slice("data: ".length));
        } catch {
          continue;
        }
        if (!isRecord(chunk)) continue;
        for (const event of eventsOfChunk(chunk)) await writeAnthropicEvent(res, event);
      }
    }
  } finally {
    reader.releaseLock();
  }
  // An empty stream: still a valid message, with no content.
  if (!ctx.messageStarted) {
    for (const event of startEvents()) await writeAnthropicEvent(res, event);
    await writeAnthropicEvent(res, { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } });
    await writeAnthropicEvent(res, { type: "content_block_stop", index: 0 });
    await writeAnthropicEvent(res, {
      type: "message_delta",
      delta: { stop_reason: "end_turn", stop_sequence: null },
      usage: { output_tokens: 0 },
    });
    await writeAnthropicEvent(res, { type: "message_stop" });
  }
}

// ---------------------------------------------------------------- request handling

async function readJsonBody(req: IncomingMessage): Promise<Record<string, unknown> | null> {
  const chunks: Buffer[] = [];
  let total = 0;
  for await (const chunk of req) {
    const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    total += buf.length;
    if (total > MAX_BODY_BYTES) return null;
    chunks.push(buf);
  }
  const text = Buffer.concat(chunks).toString("utf8");
  if (text.trim() === "") return {};
  try {
    const parsed: unknown = JSON.parse(text);
    return isRecord(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function startStreamHeaders(res: ServerResponse, failures: FailureRow[]): void {
  const header = failuresHeader(failures);
  const headers: Record<string, string> = {
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    connection: "keep-alive",
    ...(header !== null ? { "x-makefun-upstream-failures": header } : {}),
  };
  res.writeHead(200, headers);
}

function sendJsonWithFailures(res: ServerResponse, body: unknown, failures: FailureRow[]): void {
  const header = failuresHeader(failures);
  sendJson(res, 200, body, header !== null ? { "x-makefun-upstream-failures": header } : undefined);
}

/** The gateway refuses a Chat Completions body with no output cap, so a
 * client body that carries neither gets one. */
function ensureOutputCap(body: Record<string, unknown>): Record<string, unknown> {
  if (typeof body.max_tokens === "number" || typeof body.max_completion_tokens === "number") return body;
  return { ...body, max_tokens: 1024 };
}

/** POST /v1/chat/completions: dial the best free model that meets the
 * prompt, relay its answer (JSON or SSE) with the dialled id on it. */
async function handleChatCompletions(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJsonBody(req);
  if (body === null) {
    sendError(res, "openai", 400, "invalid_request_error", "The request body is not a JSON object.");
    return;
  }
  const reqCopy: Record<string, unknown> = { ...body };
  const requirements = requirementsFor(reqCopy);
  const openAIBody = ensureOutputCap(reqCopy);
  const stream = openAIBody.stream === true;
  const outcome = await dialPrompt(openAIBody, requirements);
  if (!outcome.ok) {
    sendError(res, "openai", 502, "upstream_error", "No free model answered this prompt.", outcome.failures);
    return;
  }
  if (stream) {
    startStreamHeaders(res, outcome.failures);
    try {
      await openAIStreamWithModel(outcome.response, outcome.model, res);
      res.write("data: [DONE]\n\n");
    } catch (err) {
      log(`chat completions stream relay failed: ${err instanceof Error ? err.message : String(err)}`);
    }
    res.end();
    return;
  }
  let answer: Record<string, unknown>;
  try {
    const parsed: unknown = await outcome.response.json();
    answer = isRecord(parsed) ? parsed : {};
  } catch {
    sendError(res, "openai", 502, "upstream_error", "The gateway's answer was not JSON.", outcome.failures);
    return;
  }
  sendJsonWithFailures(res, openAIJsonWithModel(answer, outcome.model), outcome.failures);
}

/** POST /v1/messages: translate to Chat Completions, dial, translate back. */
async function handleMessages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJsonBody(req);
  if (body === null) {
    sendError(res, "anthropic", 400, "invalid_request", "The request body is not a JSON object.");
    return;
  }
  const requirements = requirementsFor(body);
  const openAIBody = ensureOutputCap(anthropicToOpenAI(body, "auto"));
  const stream = body.stream === true;
  const outcome = await dialPrompt(openAIBody, requirements);
  if (!outcome.ok) {
    sendError(res, "anthropic", 502, "api_error", "No free model answered this prompt.", outcome.failures);
    return;
  }
  if (stream) {
    startStreamHeaders(res, outcome.failures);
    try {
      await anthropicEventsFromOpenAI(outcome.response, outcome.model, res);
    } catch (err) {
      log(`messages stream relay failed: ${err instanceof Error ? err.message : String(err)}`);
    }
    res.end();
    return;
  }
  let answer: Record<string, unknown>;
  try {
    const parsed: unknown = await outcome.response.json();
    answer = isRecord(parsed) ? parsed : {};
  } catch {
    sendError(res, "anthropic", 502, "api_error", "The gateway's answer was not JSON.", outcome.failures);
    return;
  }
  sendJsonWithFailures(res, openAIToAnthropic(answer, outcome.model), outcome.failures);
}

/** GET /v1/models: the OpenAI list for a Bearer caller, the Anthropic
 * list for an x-api-key caller. */
function handleModels(req: IncomingMessage, res: ServerResponse): void {
  if (req.headers["x-api-key"] !== undefined) sendJson(res, 200, anthropicModelList());
  else sendJson(res, 200, openAIModelList());
}

function route(req: IncomingMessage, res: ServerResponse): void {
  const path = (req.url ?? "").split("?")[0];
  if (req.method === "POST" && path === "/v1/chat/completions") {
    handleChatCompletions(req, res).catch((err) => {
      log(`chat completions failed: ${err instanceof Error ? err.message : String(err)}`);
      if (!res.headersSent) sendError(res, "openai", 500, "internal_error", "The server failed while serving this request.");
      else res.end();
    });
    return;
  }
  if (req.method === "POST" && path === "/v1/messages") {
    handleMessages(req, res).catch((err) => {
      log(`messages failed: ${err instanceof Error ? err.message : String(err)}`);
      if (!res.headersSent) sendError(res, "anthropic", 500, "api_error", "The server failed while serving this request.");
      else res.end();
    });
    return;
  }
  if (req.method === "GET" && path === "/v1/models") {
    handleModels(req, res);
    return;
  }
  sendJson(res, 404, openAIError("Not found.", "not_found"));
}

// ---------------------------------------------------------------- trial loop

async function postJson(url: string, body: unknown, timeoutMs: number): Promise<Response> {
  return fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });
}

async function register(port: number): Promise<boolean> {
  for (let attempt = 0; attempt < 10; attempt++) {
    try {
      const res = await postJson(`${ENV_URL}/register`, { protocol_version: PROTOCOL_VERSION, port, catalog }, 5000);
      if (res.ok) return true;
      log(`register refused: ${res.status} (attempt ${attempt + 1} of 10)`);
    } catch (err) {
      log(`register failed: ${err instanceof Error ? err.message : String(err)} (attempt ${attempt + 1} of 10)`);
    }
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  return false;
}

async function workDone(): Promise<boolean> {
  try {
    const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
    if (!res.ok) return false;
    const doc: unknown = await res.json();
    return isRecord(doc) && doc.done === true;
  } catch {
    return false;
  }
}

async function serve(): Promise<void> {
  const server: Server = createServer((req, res) => {
    route(req, res);
  });
  const port = await new Promise<number>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "0.0.0.0", () => {
      const address = server.address();
      if (address !== null && typeof address === "object") resolve(address.port);
      else reject(new Error("no port assigned"));
    });
  });
  log(`listening on 0.0.0.0:${port}`);
  if (ENV_URL !== "") {
    const ok = await register(port);
    if (!ok) {
      log("register failed: the env never accepted the server");
      server.close();
      process.exit(1);
    }
    log("registered with the env");
    for (;;) {
      await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
      if (await workDone()) break;
    }
    log("the env reports the trial is done");
    server.close();
    process.exit(0);
  }
  log("no MAKEFUN_ENV_URL: serving until interrupted");
}

// ---------------------------------------------------------------- main

async function main(): Promise<void> {
  if (process.argv.includes("--list")) {
    const doc = brief.models.length > 0 ? catalog : { providers: [] };
    process.stdout.write(`${JSON.stringify(doc)}\n`);
    return;
  }
  if (brief.models.length === 0 && PROMPT_FILE !== "") {
    log("the brief carries no free models; the catalog is empty");
  }
  await serve();
}

main().catch((err) => {
  log(`fatal: ${err instanceof Error ? err.message : String(err)}`);
  process.exit(1);
});
