// Ai4Qi build-audit: writes a complete audit protocol for a theme, on the hosted site.
// The page sends the theme and the library material it found (published audits, standards,
// proposed audits); this function calls Claude with the owner's key, which never reaches the page.
// Signed-in users only. A theme built before is served from built_audits without calling Claude
// (unless the user asks for another version). A per-person daily cap stops abuse; an optional
// site-wide cap (off by default) can send people to the "build it in Claude" link instead.
//
// The answer is a stream sent back at once, one JSON object per line:
//   {"hb": 1}            heartbeat, straight away and every 5 seconds (keeps the gateway and the page waiting)
//   {"t": "..."}         the next piece of the protocol text as Claude writes it
//   {"done": {...}}      the finished protocol
//   {"error": "..."}     "sign in", "no access", "bad request", "daily limit", "monthly limit", "busy",
//                        "not an audit topic", "invalid json", "upstream", "not configured"
// All checks and the Claude call run after the response has started, so nothing can hold up the
// headers. Errors are written to the function logs with console.error.
//
// Secrets (Supabase dashboard > Edge Functions > Secrets): ANTHROPIC_API_KEY.
// SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are provided by Supabase automatically.
// Optional: BUILD_DAILY_LIMIT (default 20), BUILD_MONTHLY_LIMIT (new builds per 30 days for the
// whole site; default 0 = no cap, the site stays free for everyone), REUSE_DAYS (how long a saved build is reused, default 180),
// BUILD_MODEL (default claude-sonnet-5; claude-haiku-4-5 costs about half, weaker protocols),
// BUILD_EFFORT (default low: fastest; medium or high think longer before writing),
// ALLOWED_ORIGIN (default *; set it to the site address, e.g. https://ai4qi.com).
// Deploy with "Verify JWT" off: the project's new JWT signing keys fail the legacy check, so the
// function checks the user itself (auth.getUser) before calling Claude.

import { createClient } from "npm:@supabase/supabase-js@2";

const MODEL = Deno.env.get("BUILD_MODEL") ?? "claude-sonnet-5";
const EFFORT = Deno.env.get("BUILD_EFFORT") ?? "low";
const LIMIT = Number(Deno.env.get("BUILD_DAILY_LIMIT") ?? "20");
const MONTHLY = Number(Deno.env.get("BUILD_MONTHLY_LIMIT") ?? "0");   // 0 = no site-wide cap
const REUSE_DAYS = Number(Deno.env.get("REUSE_DAYS") ?? "180");
const ORIGIN = Deno.env.get("ALLOWED_ORIGIN") ?? "*";
const cors = {
  "Access-Control-Allow-Origin": ORIGIN,
  "Access-Control-Allow-Headers": "authorization, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

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

type Send = (o: Record<string, unknown>) => Promise<void>;

// Claude's streamed reply (server-sent events), read with fetch: text pieces go to the page as they
// arrive. Returns the whole text and the stop reason.
async function writeWithClaude(key: string, prompt: string, send: Send): Promise<{ text: string; stop: string }> {
  const r = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: { "x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json" },
    body: JSON.stringify({
      model: MODEL, max_tokens: 16000, stream: true, system: SYSTEM,
      output_config: { effort: EFFORT },
      messages: [{ role: "user", content: prompt }],
    }),
  });
  if (!r.ok || !r.body) {
    const detail = await r.text().catch(() => "");
    console.error("anthropic http", r.status, detail.slice(0, 500));
    throw new Error(r.status === 429 || r.status === 529 ? "busy" : "upstream");
  }
  const reader = r.body.pipeThrough(new TextDecoderStream()).getReader();
  let buf = "", text = "", stop = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += value;
    let i: number;
    while ((i = buf.indexOf("\n")) !== -1) {
      const line = buf.slice(0, i).trim();
      buf = buf.slice(i + 1);
      if (!line.startsWith("data:")) continue;
      let ev: { type?: string; delta?: { type?: string; text?: string; stop_reason?: string }; error?: { type?: string; message?: string } };
      try { ev = JSON.parse(line.slice(5)); } catch { continue; }
      if (ev.type === "content_block_delta" && ev.delta?.type === "text_delta" && ev.delta.text) {
        text += ev.delta.text;
        await send({ t: ev.delta.text });
      } else if (ev.type === "message_delta" && ev.delta?.stop_reason) {
        stop = ev.delta.stop_reason;
      } else if (ev.type === "error") {
        console.error("anthropic stream error", JSON.stringify(ev.error));
        throw new Error(ev.error?.type === "overloaded_error" || ev.error?.type === "rate_limit_error" ? "busy" : "upstream");
      }
    }
  }
  return { text, stop };
}

async function build(req: Request, send: Send): Promise<void> {
  const url = Deno.env.get("SUPABASE_URL")!, service = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
  const key = Deno.env.get("ANTHROPIC_API_KEY");
  if (!key) { await send({ error: "not configured" }); return; }

  // 1. Signed-in users only: the token must belong to a real user.
  const jwt = (req.headers.get("Authorization") ?? "").replace(/^Bearer\s+/i, "");
  if (!jwt) { await send({ error: "sign in" }); return; }
  const admin = createClient(url, service, { auth: { persistSession: false } });
  const { data: who, error: authErr } = await admin.auth.getUser(jwt);
  const user = who?.user;
  if (!user) { if (authErr) console.error("auth", authErr.message); await send({ error: "sign in" }); return; }

  // 1b. NHS, HSE and university staff, or an approved access request (migration 010).
  //     If the migration has not been run yet, the check is skipped rather than blocking everyone.
  const { data: access, error: accErr } = await admin.rpc("access_for", { p_user: user.id });
  if (accErr) console.error("access check", accErr.message);
  else if (access !== "ok") { await send({ error: "no access" }); return; }

  let body: { topic?: unknown; theme?: unknown; prompt?: unknown; key?: unknown; fresh?: unknown };
  try { body = await req.json(); } catch { await send({ error: "bad request" }); return; }
  const topic = String(body.topic ?? body.theme ?? "").trim().slice(0, 300);
  const prompt = String(body.prompt ?? "");
  const topicKey = String(body.key ?? "");
  const fresh = body.fresh === true;
  if (!topic || !/^B-[a-z0-9-]{1,29}$/.test(topicKey) || prompt.length < 200 || prompt.length > 60000) {
    await send({ error: "bad request" }); return;
  }

  // 2. Reuse a saved protocol for the same theme: no Claude call, no charge.
  if (!fresh) {
    const after = new Date(Date.now() - REUSE_DAYS * 86400000).toISOString();
    const { data: saved, error } = await admin.from("built_audits").select("protocol")
      .eq("topic_key", topicKey).eq("reused", false).gte("created_at", after)
      .order("created_at", { ascending: false }).limit(1);
    if (error) console.error("reuse lookup", error.message);
    if (saved && saved.length && saved[0].protocol) {
      const { error: insErr } = await admin.from("built_audits").insert({ user_id: user.id, topic, topic_key: topicKey, reused: true });
      if (insErr) console.error("reuse insert", insErr.message);
      await send({ done: saved[0].protocol }); return;
    }
  }

  // 3. Caps on new builds: per user per day, and for the whole site per 30 days.
  const day = new Date(Date.now() - 86400000).toISOString();
  const { count } = await admin.from("built_audits").select("id", { count: "exact", head: true })
    .eq("user_id", user.id).eq("reused", false).gte("created_at", day);
  if ((count ?? 0) >= LIMIT) { await send({ error: "daily limit" }); return; }
  if (MONTHLY > 0) {
    const month = new Date(Date.now() - 30 * 86400000).toISOString();
    const { count: siteCount } = await admin.from("built_audits").select("id", { count: "exact", head: true })
      .eq("reused", false).gte("created_at", month);
    if ((siteCount ?? 0) >= MONTHLY) { await send({ error: "monthly limit" }); return; }
  }

  // 4. Write it, streaming the text to the page; save once complete.
  const { text, stop } = await writeWithClaude(key, prompt, send);
  if (stop === "refusal") { await send({ error: "not an audit topic" }); return; }
  let protocol: Record<string, unknown>;
  try { protocol = firstJson(text) as Record<string, unknown>; } catch {
    console.error("invalid json from model", stop, text.slice(0, 300));
    await send({ error: "invalid json" }); return;
  }
  if (!protocol || typeof protocol !== "object" || !protocol.question) { await send({ error: "not an audit topic" }); return; }
  const { error: saveErr } = await admin.from("built_audits").insert({ user_id: user.id, topic, topic_key: topicKey, protocol });
  if (saveErr) console.error("save", saveErr.message);
  await send({ done: protocol });
}

Deno.serve((req) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: cors });
  if (req.method !== "POST") {
    return new Response(JSON.stringify({ error: "method" }), { status: 405, headers: { ...cors, "Content-Type": "application/json" } });
  }

  const { readable, writable } = new TransformStream<Uint8Array, Uint8Array>();
  const writer = writable.getWriter();
  const enc = new TextEncoder();
  let open = true;
  const send: Send = async (o) => {
    if (!open) return;
    try { await writer.write(enc.encode(JSON.stringify(o) + "\n")); } catch { open = false; }   // the page went away
  };
  send({ hb: 1 });                                          // first bytes at once, so headers go out now
  const beat = setInterval(() => { send({ hb: 1 }); }, 5000);

  const work = build(req, send)
    .catch(async (e) => {
      console.error("build failed", e instanceof Error ? e.message : String(e));
      await send({ error: e instanceof Error && e.message === "busy" ? "busy" : "upstream" });
    })
    .finally(async () => {
      clearInterval(beat);
      open = false;
      try { await writer.close(); } catch { /* already closed */ }
    });
  // Keep the worker alive until the build has finished and been saved.
  // deno-lint-ignore no-explicit-any
  const rt = (globalThis as any).EdgeRuntime;
  if (rt && typeof rt.waitUntil === "function") rt.waitUntil(work);

  return new Response(readable, {
    status: 200,
    headers: { ...cors, "Content-Type": "application/x-ndjson; charset=utf-8", "Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no" },
  });
});
