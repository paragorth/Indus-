# Ai4Qi optional backend: feedback, sign-in, usage counting and visitor analytics

The app in `ai4qi/app/` works entirely on its own. Everything here is optional and is switched on by
filling in keys in `ai4qi/app/config.json`. With the keys empty (the default) the app behaves exactly
as before: feedback stays on the device (or goes to `feedback_url`), there is no sign-in, and no
third-party script is loaded.

| Piece | What it adds | Config keys |
|---|---|---|
| Supabase (stage 1) | Shared thumbs up/down feedback, "N people found this useful", optional sign-in by email link, profile (grade, specialty, region), "My audits" tracker, admin pages `#/admin/feedback` and `#/admin/stats` | `supabase_url`, `supabase_anon_key` |
| Visitor analytics | Page views and traffic sources, no cookies | `analytics`, `plausible_script`, `plausible_domain`, `plausible_host`, `cloudflare_token` |

No patient data is stored anywhere. Feedback holds an audit id, a rating, fixed reasons and an
optional comment of up to 500 characters. Profiles hold only values from fixed lists. The tracker
holds only an audit id, a step and dates. Analytics receive only cleaned page addresses (the route,
such as `#/proposed/ONA-012`; search words and filters are removed) and the `from` tag.

## Files

- `supabase/migrations/001_feedback_auth.sql`: tables `feedback`, `profiles`, `admins`; row level
  security; `feedback_summary()`; the limit of 30 feedback rows per device per day.
- `supabase/migrations/002_usage.sql`: region on profiles and fixed lists for grade, specialty and
  region; tables `my_audits` and `usage_events`; `record_activity()`; the admin-only functions
  `weekly_stats()` and `signup_breakdown()`. Groups of fewer than five people are suppressed.
- `supabase/migrations/003_built_audits.sql`: lets trackers accept built audits (ids `B-…`) and adds
  `built_audits`, where every protocol built on the site is kept for review (server only).
- `supabase/migrations/004_reminders.sql`: table `run_reminders` (reminder details for a running
  audit: run id, audit question, next step, due date, opt-in); users see and edit only their own
  rows, cannot change `last_sent_at`/`sends`, and can keep at most 50.
- `supabase/functions/build-audit/`: the function that builds an audit on request with Claude.
- `supabase/functions/send-reminders/`: the daily job that emails reminders through Resend.
- `supabase/migrations/005_delete_account.sql`: `delete_my_account()`, so signed-in users can delete
  their own account from the account page (cascades to their profile, progress and reminders).
- `pull_feedback.py`: copies feedback from Supabase into `ai4qi/new_audits/feedback.json`.

## Owner steps: Supabase

1. **Create the project.** Go to <https://supabase.com/dashboard>, choose *New project*, and pick
   the **London (eu-west-2)** region, or another EU region. Save the database password in your
   password manager.
2. **Run the migrations.** Open *SQL Editor* → *New query*, paste the whole of
   `supabase/migrations/001_feedback_auth.sql`, and run it. Then do the same with
   `002_usage.sql`. Both are safe to run again later.
3. **Add yourself as an admin.** In the SQL editor run (with your own address, in lower case):

   ```sql
   insert into public.admins (email) values ('you@example.nhs.uk');
   ```

   Admins can open `#/admin/feedback` and `#/admin/stats` after signing in. Nobody else can see raw
   feedback or statistics.
4. **Configure sign-in.** *Authentication* → *URL Configuration*:
   - *Site URL*: the address of the app, e.g. `https://yourname.github.io/ai4qi/app/`
   - *Redirect URLs*: add the same address followed by `**`, e.g. `https://yourname.github.io/ai4qi/app/**`

   *Authentication* → *Sign In / Providers*: keep **Email** on (it sends a magic link; no password).
   The built-in email service is limited to a few emails an hour and is meant for testing. Before
   launch, add your own SMTP service under *Authentication* → *Emails* → *SMTP settings*.
5. **Connect the app.** *Project Settings* → *API Keys*. Copy the **Project URL** and the
   **publishable** key (older projects call it the *anon public* key) into `ai4qi/app/config.json`:

   ```json
   "supabase_url": "https://abcdefghijkl.supabase.co",
   "supabase_anon_key": "sb_publishable_..."
   ```

   These two values are designed to be public; row level security protects the data.
   **Never** put the *secret* / *service_role* key in `config.json` or in any committed file.
6. **Rebuild and publish.** Run `python3 ai4qi/build_app_data.py` (it keeps `config.json` and stamps
   a new offline-cache version so installed copies update), then commit and push the app.
7. **Pull feedback into the library** when you want it:

   ```sh
   export SUPABASE_URL=https://abcdefghijkl.supabase.co
   export SUPABASE_SERVICE_KEY=...        # the secret key; keep it in your shell, never in a file in the repo
   python3 ai4qi/backend/pull_feedback.py              # add --dry-run to preview
   python3 ai4qi/new_audits/compile.py                 # attach the new totals to each audit
   ```

   The script adds new rows as `{"id", "rating", "reasons", "comment", "source": "app", "at",
   "feedback_id"}`, keeps every existing entry, and skips rows it has already copied (matched by
   `feedback_id`), so it can be run as often as you like. Device and user ids are not copied.

8. **Switch on "Build an audit"** (optional; without it the hosted site shows the published
   evidence and the closest ready-made protocol, and building works only inside Claude):
   - Get an API key at https://console.anthropic.com (Settings > API keys) and add billing.
   - Supabase dashboard > Edge Functions > Secrets: add `ANTHROPIC_API_KEY`. Optional:
     `ALLOWED_ORIGIN` (your site address), `BUILD_DAILY_LIMIT` (default 20 per user per day).
   - Run the SQL in `003_built_audits.sql`, then deploy the function:
     `supabase functions deploy build-audit --project-ref abcdefghijkl`
     (Supabase CLI; or paste `index.ts` into Dashboard > Edge Functions > Create function).
   - In `config.json` set `"build_url": "https://abcdefghijkl.supabase.co/functions/v1/build-audit"`.
   Building needs sign-in (emailed link, no password). Each new build costs roughly 3–5p.
   Cost controls, all in Edge Function secrets:
   - A theme built before is served from the saved copy (free) for `REUSE_DAYS` (default 180);
     only "Build another version" makes a new one.
   - `BUILD_DAILY_LIMIT`: new builds per person per day (default 20).
   - `BUILD_MONTHLY_LIMIT`: off by default (0), so building stays free for everyone. Emergency
     brake only: set a number of new builds per 30 days, and past it the page offers "Build it in
     Claude" (the `claude_link` in `config.json`, empty by default; people use their own Claude account).
   - `BUILD_MODEL`: set `claude-haiku-4-5` to halve the cost (weaker protocols).
   Spend so far: `select count(*) filter (where not reused) as new_builds, count(*) filter (where reused)
   as reused from built_audits where created_at > now() - interval '30 days';`
   Built protocols are in the `built_audits` table: review them and add good ones to
   `new_audits/nonortho_parts/generated.json` or `new_audits/parts/generated.json`.

### Costs and limits

- **Free plan**: enough for this app's traffic, but the project is **paused after about a week with
  no activity** (you restore it from the dashboard), and there are **no backups**. Fine for a pilot.
- **Pro plan**: about **US$25 a month** per organisation; no pausing, daily backups, larger limits
  and your own SMTP. Recommended once real users depend on it.

Check <https://supabase.com/pricing> for current prices.

## Reminder emails (optional)

A signed-in user can turn on reminders for an audit they are running. The app then saves only the
audit question, a short next step made of counts (e.g. "Finish cycle 1 data collection (32 of 40
entered)"), a due date and the on/off choice in `run_reminders`. The audit data itself (patient
rows) never leaves the device. Once a day, `send-reminders` emails each person one message listing
every audit with a next step due from 14 days ago up to tomorrow, with a link to *My audits*. Each
item is emailed at most every 3 days and at most 6 times in all. Without these steps nothing is sent.

1. **Run the SQL.** In *SQL Editor* run the whole of `supabase/migrations/004_reminders.sql`.
2. **Create a Resend account** at <https://resend.com>. Under *Domains* → *Add domain*, add the
   domain you will send from (a subdomain such as `mail.yourdomain.org` is fine) and add the DNS
   records it lists (SPF/DKIM `TXT` and `MX` records, and optionally DMARC) at your DNS provider.
   Wait until the domain shows *Verified*. Then *API Keys* → *Create API key* with *Sending access*
   only, and copy it.
3. **Set the secrets.** Supabase dashboard → *Edge Functions* → *Secrets*:
   - `RESEND_API_KEY`: the key from step 2.
   - `REMINDER_FROM`: e.g. `Ai4Qi <reminders@mail.yourdomain.org>` (must be on the verified domain).
   - `SITE_URL`: the app address, e.g. `https://yourname.github.io/ai4qi/app` (the email links to
     `SITE_URL/#/my-audits`).
   - `CRON_SECRET`: a long random string, e.g. from `openssl rand -hex 32`. Anyone without it gets 401.
   - Optional `REMINDERS_DRY_RUN=1`: the function logs what it would send (see *Edge Functions* →
     *Logs*) and neither sends nor updates anything. Remove it to go live.
4. **Deploy the function** without Supabase's JWT check (the function checks `CRON_SECRET` itself):
   `supabase functions deploy send-reminders --no-verify-jwt --project-ref abcdefghijkl`
   (or paste `index.ts` into *Edge Functions* → *Create function* and turn off *Verify JWT*).
5. **Schedule it daily at 08:00 UK time.** *Database* → *Extensions*: enable **pg_cron** and
   **pg_net**. Then in the SQL editor (with your project ref and your `CRON_SECRET`):

   ```sql
   select cron.schedule('ai4qi-reminders', '0 7 * * *', $$ select net.http_post(url := 'https://<ref>.supabase.co/functions/v1/send-reminders', headers := jsonb_build_object('Authorization', 'Bearer <CRON_SECRET>', 'Content-Type','application/json')) $$);
   ```

   pg_cron runs in UTC: `0 7 * * *` is 08:00 in British Summer Time and 07:00 in winter (use
   `0 8 * * *` in winter if the hour matters). Due dates are worked out on the UK date. To test once
   by hand: `curl -X POST -H "Authorization: Bearer <CRON_SECRET>" https://<ref>.supabase.co/functions/v1/send-reminders`;
   it answers `{"users": …, "emails": …, "items": …, "failed": …, "dry_run": …}`. See past runs with
   `select * from cron.job_run_details order by start_time desc limit 10;` and the replies with
   `select * from net._http_response order by created desc limit 10;`. Stop it with
   `select cron.unschedule('ai4qi-reminders');`.

**Costs.** Resend's free plan (checked 27 Sep 2026) allows 3,000 emails a month and 100 a day, on up
to 3 domains; Pro is US$20 a month for 50,000. One email per person per day at most, so the free plan
covers up to about 100 people with something due on the same day. Check <https://resend.com/pricing>
for current limits. pg_cron, pg_net and Edge Function calls fit within the Supabase plans above.

## What the app does with Supabase

- **Feedback**: saved on the device first and sent when online (queued while offline). Each device
  gets a random id, used only for the daily limit and to count one vote per person.
- **Sign in**: optional, by emailed link. Signed-in feedback carries the user id.
- **Profile**: grade, specialty and region from fixed lists, all optional ("Prefer not to say").
- **My audits**: on a proposed audit page, a signed-in user can press *Start this audit*, then mark
  each step: Started → Cycle 1 collected → Change made → Re-audit done → Loop closed. *Loop closed*
  counts as completed. The list is at `#/my-audits`.
- **Usage counting**: one "active day" per signed-in user per day, a "sign-up" on first sign-in, and
  started / step changed / completed events from the tracker. Users can add only their own rows and
  can read none of the raw events.
- **Admin statistics** (`#/admin/stats`): per ISO week (Monday start, UTC) sign-ups, registered users
  to date, audits started, audits completed, active users and returning users (active in the week
  and also active in an earlier week); bar charts; sign-ups by grade, specialty and region with
  groups under five shown as "fewer than 5"; a date range; CSV download for a sponsor pack.

## Owner steps: visitor analytics (choose one)

Both options set no cookies and store nothing on the visitor's device, so no consent banner is
needed. Analytics are per domain: decide the final address of the site first.

### Plausible (recommended; paid, about €9 a month, 30-day free trial)

1. Create an account at <https://plausible.io> and *Add a website* with the exact domain the app
   is served from (for GitHub Pages, e.g. `yourname.github.io`; a custom domain is better).
2. Plausible then shows a snippet containing a per-site script, e.g.
   `<script async src="https://plausible.io/js/pa-XXXXXXXX.js"></script>`. Copy only that `src`
   address. (You do not paste the snippet into the page; the app loads it for you.)
3. In the site's *Settings* → *Custom properties*, add a property called **`from`**.
4. In `config.json`:

   ```json
   "analytics": "plausible",
   "plausible_script": "https://plausible.io/js/pa-XXXXXXXX.js",
   "plausible_domain": "yourname.github.io",
   "plausible_host": "https://plausible.io"
   ```

   `plausible_domain` and `plausible_host` are used only for the link to the dashboard on the admin
   statistics page. Change `plausible_host` only if you self-host Plausible.

The app sends a page view for every screen (hash routes such as `#/proposed/ONA-012` are counted
separately) with the address cleaned first. Tag links you share with `?from=`, for example
`https://yourname.github.io/ai4qi/app/?from=qisw`; the tag is sent as the `from` custom property.
Standard `utm_source`, `utm_medium`, `utm_campaign` and `ref` parameters, and the referring site,
work as usual.

### Cloudflare Web Analytics (free)

1. In the Cloudflare dashboard open *Analytics & Logs* → *Web Analytics* → *Add a site*, enter the
   domain (the GitHub Pages default domain works with the JavaScript snippet) and copy the
   **token** from the snippet it shows (`data-cf-beacon='{"token": "..."}'`).
2. In `config.json`:

   ```json
   "analytics": "cloudflare",
   "cloudflare_token": "0123456789abcdef0123456789abcdef"
   ```

Limits: Cloudflare has no custom properties, so `?from=` is visible only as part of the landing-page
address or referrer. The app loads the beacon with single-page tracking off, so Cloudflare counts
visits and landing pages but not each screen inside the app (this keeps search words out of it).

## Suggested privacy note for the site

> **Privacy.** Ai4Qi does not use cookies for tracking and never asks for patient information.
> We count visits with a privacy-friendly analytics service (Plausible Analytics / Cloudflare Web
> Analytics), which records the page visited, the referring site and the approximate country, but
> no personal data and no identifiers stored on your device. Search words are never sent.
> If you give feedback on a proposed audit, we store your rating, any reasons and comment you
> choose, and a random device number used to prevent spam. Signing in is optional: if you do, we
> store your email address, any grade, specialty and region you choose, and the steps you record in
> My audits. These are used only for the service and in anonymous totals (groups of fewer than five
> people are never reported). Data are held by Supabase in the EU (London). To have your account
> and data deleted, contact [owner contact].

## Testing locally

With Docker running: `npx supabase init`, copy the two migrations into `supabase/migrations/` with
a numeric prefix (e.g. `20260927000001_feedback_auth.sql`, `20260927000002_usage.sql`), set
`site_url` and `additional_redirect_urls` in `supabase/config.toml` to the local app address, then
`npx supabase start`. Put the printed API URL and publishable key in a local copy of `config.json`
(do not commit it), serve `ai4qi/app` with `python3 -m http.server 8765`, and read magic-link emails
in Mailpit at <http://127.0.0.1:54324>. `npx supabase stop` when done.
