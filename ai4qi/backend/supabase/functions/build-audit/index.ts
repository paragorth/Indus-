// Ai4Qi build-audit: writes a complete audit protocol for a theme, on the hosted site.
// The page sends the theme and the library material it found (published audits, standards,
// proposed audits); this function calls Claude with the owner's key, which never reaches the page.
// Signed-in users only, with a daily cap per user. Each protocol is saved to built_audits.
//
// Secrets (Supabase dashboard > Edge Functions > Secrets): ANTHROPIC_API_KEY.
// SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are provided by Supabase automatically.
// Optional: BUILD_DAILY_LIMIT (default 20), BUILD_MODEL (default claude-sonnet-5),
// ALLOWED_ORIGIN (default *; set it to the site address, e.g. https://ai4qi.org).

import { createClient } from "npm:@supabase/supabase-js@2";

const MODEL = Deno.env.get("BUILD_MODEL") ?? "claude-sonnet-5";
const LIMIT = Number(Deno.env.get("BUILD_DAILY_LIMIT") ?? "20");
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

  let body: { topic?: unknown; prompt?: unknown };
  try { body = await req.json(); } catch { return reply(400, { error: "bad json" }); }
  const topic = String(body.topic ?? "").trim().slice(0, 300);
  const prompt = String(body.prompt ?? "");
  if (!topic || prompt.length < 200 || prompt.length > 60000) return reply(400, { error: "bad request" });

  const since = new Date(Date.now() - 86400000).toISOString();
  const { count } = await admin.from("built_audits").select("id", { count: "exact", head: true })
    .eq("user_id", user.id).gte("created_at", since);
  if ((count ?? 0) >= LIMIT) return reply(429, { error: "daily limit" });

  const r = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json" },
    body: JSON.stringify({ model: MODEL, max_tokens: 8000, system: SYSTEM, messages: [{ role: "user", content: prompt }] }),
  });
  if (r.status === 429) return reply(429, { error: "busy" });
  if (!r.ok) return reply(502, { error: "upstream" });
  const out = await r.json();
  const text = (out.content ?? []).filter((c: { type: string }) => c.type === "text").map((c: { text: string }) => c.text).join("");
  let protocol: Record<string, unknown>;
  try { protocol = firstJson(text) as Record<string, unknown>; } catch { return reply(502, { error: "invalid json" }); }
  if (!protocol || typeof protocol !== "object" || !protocol.question) return reply(422, { error: "not an audit topic" });

  await admin.from("built_audits").insert({ user_id: user.id, topic, protocol });
  return reply(200, protocol);
});
