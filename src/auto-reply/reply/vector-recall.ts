import type { OpenClawConfig } from "../../config/config.js";
import { logVerbose } from "../../globals.js";

export type VectorRecallMode = "heuristic" | "always";

export type VectorRecallSettings = {
  enabled: boolean;
  mode: VectorRecallMode;
  limit: number;
  maxChars: number;
  /** Max chars per snippet inserted into prompt. */
  maxSnippetChars: number;
  /** System prompt label for injected recall. */
  systemLabel: string;

  qdrantUrl: string;
  qdrantCollection: string;
  /** Environment variable name holding Qdrant API key. */
  qdrantApiKeyEnv: string;

  ollamaUrl: string;
  ollamaEmbedModel: string;
};

export function resolveVectorRecallSettings(cfg: OpenClawConfig): VectorRecallSettings {
  const raw = cfg.agents?.defaults?.vectorRecall ?? {};
  const enabled = Boolean(raw.enabled ?? false);
  const mode: VectorRecallMode = raw.mode === "always" ? "always" : "heuristic";
  const limit = clampInt(raw.limit ?? 6, 1, 20);
  const maxChars = clampInt(raw.maxChars ?? 2200, 200, 20000);
  const maxSnippetChars = clampInt(raw.maxSnippetChars ?? 500, 100, 2000);
  const systemLabel = String(raw.systemLabel ?? "Vector recall (Qdrant)").trim();

  const qdrantUrl = String(raw.qdrantUrl ?? process.env.QDRANT_URL ?? "http://127.0.0.1:6333");
  const qdrantCollection = String(
    raw.qdrantCollection ?? process.env.QDRANT_COLLECTION ?? "agent_memories",
  );
  const qdrantApiKeyEnv = String(raw.qdrantApiKeyEnv ?? "QDRANT_API_KEY");

  const ollamaUrl = String(raw.ollamaUrl ?? process.env.OLLAMA_URL ?? "http://127.0.0.1:11434");
  const ollamaEmbedModel = String(
    raw.ollamaEmbedModel ?? process.env.OLLAMA_EMBED_MODEL ?? "mxbai-embed-large:latest",
  );

  return {
    enabled,
    mode,
    limit,
    maxChars,
    maxSnippetChars,
    systemLabel,
    qdrantUrl,
    qdrantCollection,
    qdrantApiKeyEnv,
    ollamaUrl,
    ollamaEmbedModel,
  };
}

function clampInt(v: unknown, min: number, max: number): number {
  const n = typeof v === "number" ? v : Number(v);
  if (!Number.isFinite(n)) {
    return min;
  }
  return Math.max(min, Math.min(max, Math.trunc(n)));
}

export function shouldRunVectorRecall(params: {
  settings: VectorRecallSettings;
  messageText: string;
  isHeartbeat: boolean;
}): boolean {
  const { settings, messageText, isHeartbeat } = params;
  if (!settings.enabled) {
    return false;
  }
  if (isHeartbeat) {
    return false;
  }
  if (settings.mode === "always") {
    return true;
  }
  return looksLikeMemoryQuestion(messageText);
}

// Intentionally simple: fast + explainable.
export function looksLikeMemoryQuestion(text: string): boolean {
  const t = text.trim().toLowerCase();
  if (!t) {
    return false;
  }

  const needles = [
    "did we",
    "did i",
    "did you",
    "what did we",
    "what was",
    "what were",
    "remind me",
    "remember",
    "where did we",
    "when did we",
    "who is",
    "who was",
    "what is the name",
    "what's the name",
    "last time",
    "previously",
    "earlier you",
    "you said",
    "we said",
    "we decided",
    "what did you do",
    "status of",
    "where is",
  ];
  if (needles.some((n) => t.includes(n))) {
    return true;
  }

  // Question marks often correlate, but we avoid triggering on every question.
  if (t.endsWith("?") && (t.includes("remember") || t.includes("did we") || t.includes("where"))) {
    return true;
  }

  return false;
}

async function ollamaEmbed(params: {
  settings: VectorRecallSettings;
  input: string;
}): Promise<number[] | null> {
  const { settings, input } = params;
  const url = settings.ollamaUrl.replace(/\/$/, "");

  // Prefer /api/embed; fall back to /api/embeddings
  try {
    const r = await fetch(url + "/api/embed", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: settings.ollamaEmbedModel, input }),
    });
    if (r.status === 404) {
      throw new Error("embed_404");
    }
    if (!r.ok) {
      throw new Error(`ollama embed failed status=${r.status}`);
    }
    const j: unknown = await r.json();
    const vec = (() => {
      if (!j || typeof j !== "object") {
        return null;
      }
      const embeddings = (j as { embeddings?: unknown }).embeddings;
      if (!Array.isArray(embeddings) || !Array.isArray(embeddings[0])) {
        return null;
      }
      return embeddings[0] as number[];
    })();
    if (!vec) {
      return null;
    }
    return vec;
  } catch (err) {
    if (!(err instanceof Error) || err.message !== "embed_404") {
      // continue to fallback
    }
  }

  try {
    const r = await fetch(url + "/api/embeddings", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: settings.ollamaEmbedModel, prompt: input }),
    });
    if (!r.ok) {
      throw new Error(`ollama embeddings failed status=${r.status}`);
    }
    const j: unknown = await r.json();
    const vec = (() => {
      if (!j || typeof j !== "object") {
        return null;
      }
      const embedding = (j as { embedding?: unknown }).embedding;
      if (!Array.isArray(embedding)) {
        return null;
      }
      return embedding as number[];
    })();
    return vec;
  } catch {
    return null;
  }
}

type QdrantHit = {
  score?: number;
  payload?: Record<string, unknown>;
};

async function qdrantSearch(params: {
  settings: VectorRecallSettings;
  vector: number[];
  limit: number;
}): Promise<QdrantHit[]> {
  const { settings, vector, limit } = params;
  const url = settings.qdrantUrl.replace(/\/$/, "");
  const apiKey = process.env[settings.qdrantApiKeyEnv] ?? "";

  const r = await fetch(
    `${url}/collections/${encodeURIComponent(settings.qdrantCollection)}/points/search`,
    {
      method: "POST",
      headers: {
        "content-type": "application/json",
        ...(apiKey ? { "api-key": apiKey } : {}),
      },
      body: JSON.stringify({
        vector,
        limit,
        with_payload: true,
        with_vectors: false,
      }),
    },
  );

  if (!r.ok) {
    const body = await r.text().catch(() => "");
    throw new Error(`qdrant search failed status=${r.status} body=${body.slice(0, 200)}`);
  }

  const j: unknown = await r.json();
  const hits = (() => {
    if (!j || typeof j !== "object") {
      return [];
    }
    const result = (j as { result?: unknown }).result;
    return Array.isArray(result) ? result : [];
  })();

  return hits.map((h) => {
    if (!h || typeof h !== "object") {
      return { score: undefined, payload: undefined };
    }
    const score = (h as { score?: unknown }).score;
    const payload = (h as { payload?: unknown }).payload;
    return {
      score: typeof score === "number" ? score : undefined,
      payload:
        payload && typeof payload === "object" ? (payload as Record<string, unknown>) : undefined,
    };
  });
}

function clip(s: string, max: number): string {
  if (s.length <= max) {
    return s;
  }
  return s.slice(0, max - 1) + "…";
}

export async function buildVectorRecallSystemPrompt(params: {
  cfg: OpenClawConfig;
  messageText: string;
  isHeartbeat: boolean;
}): Promise<string | null> {
  const settings = resolveVectorRecallSettings(params.cfg);
  if (
    !shouldRunVectorRecall({
      settings,
      messageText: params.messageText,
      isHeartbeat: params.isHeartbeat,
    })
  ) {
    return null;
  }

  const query = clip(params.messageText.trim(), 800);
  if (!query) {
    return null;
  }

  try {
    const vec = await ollamaEmbed({ settings, input: query });
    if (!vec) {
      return null;
    }
    const hits = await qdrantSearch({ settings, vector: vec, limit: settings.limit });

    const lines: string[] = [];
    for (const h of hits) {
      const canon = typeof h.payload?.canon === "string" ? h.payload.canon : "";
      const ts = typeof h.payload?.ts === "number" ? h.payload.ts : undefined;
      const role = typeof h.payload?.role === "string" ? h.payload.role : undefined;
      const snippet = clip(
        canon || JSON.stringify(h.payload ?? {}).slice(0, 800),
        settings.maxSnippetChars,
      );
      if (!snippet.trim()) {
        continue;
      }
      const metaBits = [
        role ? `role=${role}` : null,
        typeof ts === "number" ? `ts=${ts}` : null,
        typeof h.score === "number" ? `score=${h.score.toFixed(3)}` : null,
      ].filter(Boolean);
      lines.push(`- ${snippet}${metaBits.length ? ` (${metaBits.join(" ")})` : ""}`);
    }

    if (!lines.length) {
      return null;
    }

    const block = `${settings.systemLabel}:\n${lines.join("\n")}`;
    return clip(block, settings.maxChars);
  } catch (err) {
    logVerbose(`vector recall failed: ${String(err)}`);
    return null;
  }
}
