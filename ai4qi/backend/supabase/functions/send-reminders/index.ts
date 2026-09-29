// Ai4Qi send-reminders: once a day, emails signed-in users whose running audits have a next step
// due. Called by a pg_cron schedule (see README, "Reminder emails (optional)"), never by the page.
// Reads only public.run_reminders: a run id, the audit question, a next step made of counts, a due
// date. Audit data (patient rows) stays on the user's device and never reaches this function.
//
// Which rows: email_opt_in, due from 14 days ago up to tomorrow (UK date), not emailed in the last
// 3 days, fewer than 6 emails so far. One email per user lists all their due items; then
// last_sent_at and sends are updated for each item sent.
//
// Secrets (Supabase dashboard > Edge Functions > Secrets): CRON_SECRET (the schedule sends
// "Authorization: Bearer <CRON_SECRET>"), RESEND_API_KEY, REMINDER_FROM (e.g. "Ai4Qi
// <reminders@yourdomain>"), SITE_URL (the app address, e.g. https://yourname.github.io/ai4qi/app).
// SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are provided by Supabase automatically.
// Optional: REMINDERS_DRY_RUN=1 logs what would be sent and changes nothing.
// Deploy with --no-verify-jwt: CRON_SECRET is checked here instead of a Supabase JWT.

import { createClient } from "npm:@supabase/supabase-js@2";

const BRAND = "#3346D3";
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const MAX_SENDS = 6;
const GAP_DAYS = 3;
const LOOKBACK_DAYS = 14;

type Row = {
  user_id: string;
  run_id: string;
  audit_title: string;
  next_step: string;
  due_date: string; // YYYY-MM-DD
  sends: number;
};

const reply = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

// Constant-time comparison, so the secret cannot be guessed from response times.
function sameSecret(a: string, b: string): boolean {
  const x = new TextEncoder().encode(a), y = new TextEncoder().encode(b);
  let diff = x.length ^ y.length;
  for (let i = 0; i < Math.max(x.length, y.length); i++) diff |= (x[i] ?? 0) ^ (y[i] ?? 0);
  return diff === 0;
}

// Today's date in the UK, as YYYY-MM-DD.
function ukToday(): string {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: "Europe/London", year: "numeric", month: "2-digit", day: "2-digit",
  }).format(new Date());
}

function addDays(ymd: string, days: number): string {
  const d = new Date(ymd + "T00:00:00Z");
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
}

// '2026-10-12' -> '12 Oct 2026' (or '12 Oct' without the year).
function fmtDate(ymd: string, year = true): string {
  const [y, m, d] = ymd.split("-").map(Number);
  return `${d} ${MONTHS[m - 1]}` + (year ? ` ${y}` : "");
}

const esc = (s: string) =>
  s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");

// One line of text, no control characters (the values come from the app, but keep the email tidy).
const clean = (s: string) => s.replace(/[\u0000-\u001f\u007f]+/g, " ").trim();

function buildEmail(items: Row[], today: string, link: string) {
  const sorted = [...items].sort((a, b) => a.due_date.localeCompare(b.due_date));
  const dueWord = (r: Row) => (r.due_date < today ? "Was due" : "Due");
  const subject = sorted.length === 1
    ? `Your audit: next step ${sorted[0].due_date < today ? "was due" : "due"} ${fmtDate(sorted[0].due_date, false)}`
    : `Your audits: ${sorted.length} next steps due`;
  const footer = "You get this because email reminders are on for your Ai4Qi audits. Turn them off for all audits in Ai4Qi under Account.";
  const intro = sorted.length === 1
    ? "This is a reminder of the next step in your audit."
    : "This is a reminder of the next steps in your audits.";

  const text = [
    "Hello,",
    "",
    intro,
    "",
    ...sorted.flatMap((r) => [
      clean(r.audit_title),
      `Next step: ${clean(r.next_step)}`,
      `${dueWord(r)}: ${fmtDate(r.due_date)}`,
      "",
    ]),
    `Open My audits: ${link}`,
    "",
    "Ai4Qi",
    "",
    "--",
    footer,
  ].join("\n");

  const font = "font-family:Arial,Helvetica,sans-serif;";
  const rows = sorted.map((r) =>
    `<tr><td style="padding:12px 0;border-top:1px solid #e3e5ee;${font}">` +
    `<div style="font-size:15px;font-weight:bold;color:#1a1c28;">${esc(clean(r.audit_title))}</div>` +
    `<div style="font-size:14px;color:#1a1c28;margin-top:4px;">Next step: ${esc(clean(r.next_step))}</div>` +
    `<div style="font-size:14px;color:#555a6e;margin-top:4px;">${dueWord(r)}: ${esc(fmtDate(r.due_date))}</div>` +
    `</td></tr>`
  ).join("");
  const html =
    `<!doctype html><html lang="en-GB"><body style="margin:0;padding:0;background:#f5f6fa;">` +
    `<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f5f6fa;"><tr><td align="center" style="padding:24px 12px;">` +
    `<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width:560px;background:#ffffff;border-radius:8px;"><tr><td style="padding:24px;${font}">` +
    `<p style="margin:0 0 12px;font-size:15px;color:#1a1c28;${font}">Hello,</p>` +
    `<p style="margin:0 0 16px;font-size:15px;color:#1a1c28;${font}">${esc(intro)}</p>` +
    `<table role="presentation" width="100%" cellpadding="0" cellspacing="0">${rows}</table>` +
    `<p style="margin:20px 0 8px;"><a href="${esc(link)}" style="display:inline-block;background:${BRAND};color:#ffffff;text-decoration:none;font-size:15px;font-weight:bold;padding:12px 20px;border-radius:6px;${font}">Open My audits</a></p>` +
    `<p style="margin:16px 0 0;font-size:15px;color:#1a1c28;${font}">Ai4Qi</p>` +
    `</td></tr></table>` +
    `<p style="max-width:560px;margin:16px auto 0;font-size:12px;color:#6b7083;${font}">${esc(footer)}</p>` +
    `</td></tr></table></body></html>`;

  return { subject, text, html };
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

Deno.serve(async (req) => {
  if (req.method !== "POST") return reply(405, { error: "method" });

  const secret = Deno.env.get("CRON_SECRET") ?? "";
  if (!secret) return reply(503, { error: "not configured" });
  const given = (req.headers.get("Authorization") ?? "").replace(/^Bearer\s+/i, "");
  if (!sameSecret(given, secret)) return reply(401, { error: "unauthorised" });

  const dryRun = Deno.env.get("REMINDERS_DRY_RUN") === "1";
  const resendKey = Deno.env.get("RESEND_API_KEY") ?? "";
  const from = Deno.env.get("REMINDER_FROM") ?? "";
  const site = (Deno.env.get("SITE_URL") ?? "").replace(/\/+$/, "");
  if (!dryRun && (!resendKey || !from || !site)) return reply(503, { error: "not configured" });
  const link = `${site}/#/my-audits`;

  const url = Deno.env.get("SUPABASE_URL")!, service = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!;
  const admin = createClient(url, service, { auth: { persistSession: false } });

  // 1. Due rows, page by page.
  const today = ukToday();
  const before = new Date(Date.now() - GAP_DAYS * 86400000).toISOString();
  const rows: Row[] = [];
  const PAGE = 1000;
  for (let start = 0; ; start += PAGE) {
    const { data, error } = await admin.from("run_reminders")
      .select("user_id, run_id, audit_title, next_step, due_date, sends")
      .eq("email_opt_in", true)
      .lte("due_date", addDays(today, 1))
      .gte("due_date", addDays(today, -LOOKBACK_DAYS))
      .lt("sends", MAX_SENDS)
      .or(`last_sent_at.is.null,last_sent_at.lt.${before}`)
      .order("user_id").order("run_id")
      .range(start, start + PAGE - 1);
    if (error) { console.error("select failed", error.message); return reply(500, { error: "database" }); }
    rows.push(...(data as Row[]));
    if (!data || data.length < PAGE) break;
  }

  // 2. Group by user.
  const byUser = new Map<string, Row[]>();
  for (const r of rows) {
    const list = byUser.get(r.user_id) ?? [];
    list.push(r);
    byUser.set(r.user_id, list);
  }

  // 3. One email per user.
  let emails = 0, items = 0, failed = 0;
  for (const [uid, list] of byUser) {
    const { data: got, error } = await admin.auth.admin.getUserById(uid);
    const email = got?.user?.email;
    if (error || !email) { console.warn("no email for user", uid); failed++; continue; }
    const msg = buildEmail(list, today, link);

    if (dryRun) {
      console.log(JSON.stringify({ dry_run: true, user: uid, subject: msg.subject, items: list.length }));
      console.log(msg.text);
      emails++; items += list.length;
      continue;
    }

    const r = await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${resendKey}`,
        "Content-Type": "application/json",
        // A retry on the same day for the same user does not send a second email.
        "Idempotency-Key": `ai4qi-reminder-${uid}-${today}`,
      },
      body: JSON.stringify({ from, to: [email], subject: msg.subject, text: msg.text, html: msg.html }),
    });
    if (!r.ok) {
      console.error("resend failed", uid, r.status, (await r.text()).slice(0, 300));
      failed++;
      if (r.status === 429) await sleep(2000);
      continue;
    }
    await r.body?.cancel();
    emails++;

    const sentAt = new Date().toISOString();
    for (const row of list) {
      const { error: upErr } = await admin.from("run_reminders")
        .update({ last_sent_at: sentAt, sends: row.sends + 1 })
        .eq("user_id", row.user_id).eq("run_id", row.run_id);
      if (upErr) console.error("update failed", row.user_id, row.run_id, upErr.message);
      else items++;
    }
    await sleep(600); // stay under Resend's per-second request limit
  }

  return reply(200, { users: byUser.size, emails, items, failed, dry_run: dryRun });
});
