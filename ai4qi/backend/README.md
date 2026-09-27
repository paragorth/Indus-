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

### Costs and limits

- **Free plan**: enough for this app's traffic, but the project is **paused after about a week with
  no activity** (you restore it from the dashboard), and there are **no backups**. Fine for a pilot.
- **Pro plan**: about **US$25 a month** per organisation; no pausing, daily backups, larger limits
  and your own SMTP. Recommended once real users depend on it.

Check <https://supabase.com/pricing> for current prices.

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
