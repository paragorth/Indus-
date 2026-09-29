// Ai4Qi build-audit: writes a complete audit protocol for a theme, on the hosted site.
// The page sends the theme and the library material it found (published audits, standards,
// proposed audits); this function calls Claude with the owner's key, which never reaches the page.
// Signed-in users only. A theme built before is served from built_audits without calling Claude
// (unless the user asks for another version). A per-person daily cap stops abuse; an optional
// site-wide cap (off by default) can send people to the "build it in Claude" link instead.
//
// Secrets (Supabase dashboard > Edge Functions > Secrets): ANTHROPIC_API_KEY.
// SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are provided by Supabase automatically.
// Optional: BUILD_DAILY_LIMIT (default 20), BUILD_MONTHLY_LIMIT (new builds per 30 days for the
// whole site; default 0 = no cap, the site stays free for everyone), REUSE_DAYS (how long a saved build is reused, default 180),
// BUILD_MODEL (default claude-sonnet-5; claude-haiku-4-5 costs about half, weaker protocols), BUILD_EFFORT (default low),
// ALLOWED_ORIGIN (default *; set it to the site address, e.g. https://ai4qi.org).

import { createClient } from "npm:@supabase/supabase-js@2";
import Anthropic from "npm:@anthropic-ai/sdk";

const MODEL = Deno.env.get("BUILD_MODEL") ?? "claude-sonnet-5";
const LIMIT = Number(Deno.env.get("BUILD_DAILY_LIMIT") ?? "20");
const MONTHLY = Number(Deno.env.get("BUILD_MONTHLY_LIMIT") ?? "0");   // 0 = no site-wide cap
const REUSE_DAYS = Number(Deno.env.get("REUSE_DAYS") ?? "180");
// How hard the model thinks before writing. "low" keeps a build to well under a minute; raise to
// "medium" or "high" (BUILD_EFFORT secret) if protocols need more depth.
const EFFORT = (Deno.env.get("BUILD_EFFORT") ?? "low") as "low" | "medium" | "high";
const ORIGIN = Deno.env.get("ALLOWED_ORIGIN") ?? "*";
const cors = {
  "Access-Control-Allow-Origin": ORIGIN,
  "Access-Control-Allow-Headers": "authorization, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};
const reply = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { ...cors, "Content-Type": "application/json" } });

const SYSTEM =
  "You write clinical audit protocols for Ai4Qi, a professional clinical audit library for UK and Irish clinicians. " +
  "Follow the instructions in the user message exactly and reply with only the JSON object it asks for. " +
  "Only write clinical audit or quality improvement protocols; for any other request reply with {\"error\": \"not an audit topic\"}.";

function firstJson(text: string): unknown {
  try { return JSON.parse(text); } catch { /* fall through */ }
  const a = text.indexOf("{"), b = text.lastIndexOf("}");
  if (a === -1 || b <= a) throw new Error("no json");
  return JSON.parse(text.slice(a, b + 1));
}

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: cors });
  if (req.method !== "POST") return reply(405, { error: "method" });

  const url = Deno.env.get("SUPABASE_URL")!, service = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
  const key = Deno.env.get("ANTHROPIC_API_KEY");
  if (!key) return reply(503, { error: "not configured" });

  const jwt = (req.headers.get("Authorization") ?? "").replace(/^Bearer\s+/i, "");
  const admin = createClient(url, service, { auth: { persistSession: false } });
  const { data: who } = await admin.auth.getUser(jwt);
  const user = who?.user;
  if (!user) return reply(401, { error: "sign in" });

  let body: { topic?: unknown; prompt?: unknown; key?: unknown; fresh?: unknown };
  try { body = await req.json(); } catch { return reply(400, { error: "bad json" }); }
  const topic = String(body.topic ?? "").trim().slice(0, 300);
  const prompt = String(body.prompt ?? "");
  const topicKey = String(body.key ?? "");
  const fresh = body.fresh === true;
  if (!topic || !/^B-[a-z0-9-]{1,29}$/.test(topicKey) || prompt.length < 200 || prompt.length > 60000) {
    return reply(400, { error: "bad request" });
  }

  // 1. Reuse a saved protocol for the same theme: no Claude call, no charge.
  if (!fresh) {
    const after = new Date(Date.now() - REUSE_DAYS * 86400000).toISOString();
    const { data: saved } = await admin.from("built_audits").select("protocol")
      .eq("topic_key", topicKey).eq("reused", false).gte("created_at", after)
      .order("created_at", { ascending: false }).limit(1);
    if (saved && saved.length && saved[0].protocol) {
      await admin.from("built_audits").insert({ user_id: user.id, topic, topic_key: topicKey, reused: true });
      return reply(200, saved[0].protocol);
    }
  }

  // 2. Caps on new builds: per user per day, and for the whole site per 30 days.
  const day = new Date(Date.now() - 86400000).toISOString();
  const { count } = await admin.from("built_audits").select("id", { count: "exact", head: true })
    .eq("user_id", user.id).eq("reused", false).gte("created_at", day);
  if ((count ?? 0) >= LIMIT) return reply(429, { error: "daily limit" });
  const month = new Date(Date.now() - 30 * 86400000).toISOString();
  const { count: siteCount } = await admin.from("built_audits").select("id", { count: "exact", head: true })
    .eq("reused", false).gte("created_at", month);
  if (MONTHLY > 0 && (siteCount ?? 0) >= MONTHLY) return reply(429, { error: "monthly limit" });

  // 3. Write it. The protocol is streamed to the page as it is written (one JSON object per line:
  // {"t": text so far added} ... then {"done": protocol} or {"error": code}), so the page can show the
  // sections filling in instead of a blank wait. Saved to built_audits once complete.
  const anthropic = new Anthropic({ apiKey: key });
  const enc = new TextEncoder();
  const out = new ReadableStream({
    async start(ctl) {
      const send = (o: unknown) => ctl.enqueue(enc.encode(JSON.stringify(o) + "\n"));
      try {
        const stream = anthropic.messages.stream({
          model: MODEL, max_tokens: 16000, system: SYSTEM,
          output_config: { effort: EFFORT },
          messages: [{ role: "user", content: prompt }],
        });
        for await (const ev of stream) {
          if (ev.type === "content_block_delta" && ev.delta.type === "text_delta") send({ t: ev.delta.text });
        }
        const msg = await stream.finalMessage();
        if (msg.stop_reason === "refusal") { send({ error: "not an audit topic" }); return; }
        const text = msg.content.filter((c) => c.type === "text").map((c) => (c as { text: string }).text).join("");
        let protocol: Record<string, unknown>;
        try { protocol = firstJson(text) as Record<string, unknown>; } catch { send({ error: "invalid json" }); return; }
        if (!protocol || typeof protocol !== "object" || !protocol.question) { send({ error: "not an audit topic" }); return; }
        await admin.from("built_audits").insert({ user_id: user.id, topic, topic_key: topicKey, protocol });
        send({ done: protocol });
      } catch (e) {
        send({ error: e instanceof Anthropic.RateLimitError ? "busy" : "upstream" });
      } finally {
        ctl.close();
      }
    },
  });
  return new Response(out, { status: 200, headers: { ...cors, "Content-Type": "application/x-ndjson", "Cache-Control": "no-store" } });
});
