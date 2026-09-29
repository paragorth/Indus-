# Ai4Qi launch runbook (for a browser agent, with owner stops)

A browser agent (Claude in Chrome, or similar) can do almost all of this. It stops and hands back to
the owner at four kinds of step: **paying**, **accepting legal terms**, **two-factor codes or
passkeys**, and **signing**. Everything else is clicking, copying and pasting.

Code and SQL live in the GitHub repository `paragorth/Indus-`, branch
`claude/ai4qi-audit-library-pipeline-8yph4w`, folder `ai4qi/`. Open files with the GitHub page's
**Raw** or **Copy raw file** button.

---

## Prompt to paste into the browser agent

> You are setting up the hosting for Ai4Qi, a free clinical audit website. Follow the runbook at
> https://github.com/paragorth/Indus-/blob/claude/ai4qi-audit-library-pipeline-8yph4w/ai4qi/governance/LAUNCH_RUNBOOK.md
> section by section, in order.
> Rules:
> 1. **STOP and ask me** before any payment, before ticking or clicking anything that accepts terms,
>    a data processing agreement or a contract, when a two-factor code, passkey or password is
>    needed, and before anything that cannot be undone (deleting a project, changing DNS
>    nameservers).
> 2. **Secrets:** API keys, the database password, the Supabase secret or service_role key and the
>    CRON_SECRET are only ever copied from one dashboard straight into another dashboard's secret
>    field. Never type them into a chat, a document, GitHub or an email, and never show them to me
>    in your summary. Save the database password to my browser's password manager when offered.
> 3. Never click through CAPTCHAs or bot checks; ask me to do them.
> 4. When a section is done, tell me in one line what you did and what is left.
> 5. At the end, give me only the **public** values listed in section 9, so I can pass them to the
>    developer.

---

## 1. Domain (owner pays)

1. Cloudflare dashboard → **Domain Registration** → **Register Domains**, and search for the name
   (for example `ai4qi.org` or `ai4qi.co.uk`).
2. **STOP (owner pays).** Buying the domain puts it straight into Cloudflare DNS.

## 2. Website hosting: Cloudflare Pages

1. Cloudflare → **Workers & Pages** → **Create** → **Pages** → **Connect to Git** → GitHub →
   repository `paragorth/Indus-` (the GitHub app must be allowed to see it: **STOP** if GitHub asks
   the owner to authorise).
2. Production branch: `claude/ai4qi-audit-library-pipeline-8yph4w`. Framework preset: **None**.
   Build command: *(leave empty)*. Build output directory: `ai4qi/app`. **Save and Deploy**.
3. Project → **Custom domains** → add the domain from section 1 (root, or `www`). Cloudflare adds
   the DNS record and the HTTPS certificate.
4. Open the site and check that the home page loads, a search returns results, and **My audits**
   asks for a passcode.

## 3. Database and sign-in: Supabase (London)

1. https://supabase.com/dashboard → sign up. **STOP (owner accepts the terms and the DPA; turns on
   two-factor authentication under Account → Security).**
2. **New project**: name `ai4qi`, region **London (eu-west-2)**, and generate a strong database
   password (save it in the password manager). Free plan to start.
3. **SQL Editor** → **New query**. Run each file below in this order, pasting the whole file each time:
   `ai4qi/backend/supabase/migrations/001_feedback_auth.sql`, `002_usage.sql`,
   `003_built_audits.sql`, `004_reminders.sql`, `005_delete_account.sql`, `006_consent_profile.sql`, `007_profile_names.sql`, `008_reminders_default.sql`.
   Each should end with "Success. No rows returned".
4. Same editor, with the owner's own sign-in email in lower case:
   `insert into public.admins (email) values ('owner@example.org');`
5. **Authentication** → **URL Configuration**:
   - **Site URL**: the live address from section 2, e.g. `https://ai4qi.org/`.
   - **Redirect URLs**: add the same address followed by `**`.
6. **Authentication** → **Sign In / Providers**: keep **Email** on (a magic link, no password).
7. **Database** → **Extensions**: turn on **pg_cron** and **pg_net**.

## 4. "Build an audit": Anthropic API

1. https://console.anthropic.com → sign up. **STOP (owner accepts the Commercial Terms, which
   include the data processing addendum, and adds billing).**
2. **Settings** → **Limits**: set a monthly spend limit (suggest US$50 to start).
3. **Settings** → **API keys** → **Create key**, named `ai4qi-build`. Copy it and go straight to
   step 4.
4. Supabase → **Edge Functions** → **Secrets** → add `ANTHROPIC_API_KEY` = the key. Also add
   `ALLOWED_ORIGIN` = the live site address (no trailing slash).
5. Supabase → **Edge Functions** → **Deploy a new function** → **Via editor**. Name it
   `build-audit` and paste the whole of `ai4qi/backend/supabase/functions/build-audit/index.ts`.
   Deploy it with **Verify JWT off**. The project uses Supabase's new JWT signing keys, and the
   legacy "Verify JWT" check rejects real users' tokens. The function checks sign-in itself: it calls
   `auth.getUser` with the request's `Authorization` token and returns 401, without calling Claude,
   when the token is missing or invalid.
   The function streams the protocol to the page as it is written. Optional secret `BUILD_EFFORT`
   (`low` by default, for speed; `medium` or `high` for more depth, slower).

## 5. Email: Resend (sign-in links and reminders)

1. https://resend.com → sign up. **STOP (owner accepts the terms; the DPA is part of them).**
2. **Domains** → **Add domain** → `mail.<your domain>`, region **Ireland (eu-west-1)** if offered.
3. Add every DNS record Resend lists (TXT for SPF and DKIM, MX, and the optional DMARC) in the
   Cloudflare DNS for the domain. Wait until Resend shows **Verified**, which usually takes a few
   minutes.
4. **API Keys** → **Create API key**, named `ai4qi`, with permission **Sending access** only.
   Copy it straight into:
   - Supabase → **Edge Functions** → **Secrets** → `RESEND_API_KEY`.
   - Supabase → **Authentication** → **Emails** → **SMTP Settings**: enable custom SMTP. Host
     `smtp.resend.com`, port `465`, username `resend`, password = the same key, sender
     `Ai4Qi <signin@mail.<your domain>>`.
4b. Supabase → **Authentication** → **Emails** → **Templates**: paste the branded sign-in emails from
   `ai4qi/backend/supabase/email-templates/` (**Magic Link** and **Confirm signup**; subjects and steps
   in that folder's README).
5. Add these secrets in Supabase → **Edge Functions** → **Secrets**:
   - `REMINDER_FROM` = `Ai4Qi <reminders@mail.<your domain>>`
   - `SITE_URL` = the live site address (no trailing slash)
   - `CRON_SECRET` = a random string of 64 letters and numbers. Use the password manager's
     generator and do not store it anywhere else.
6. **Edge Functions** → **Deploy a new function** → **Via editor**. Name it `send-reminders` and
   paste `ai4qi/backend/supabase/functions/send-reminders/index.ts`. Turn **Verify JWT off**
   (the function checks CRON_SECRET itself and returns 401 on a mismatch, 503 if it is not set), then deploy.
7. **SQL Editor**, replacing `<ref>` (the project ref in the dashboard address) and `<CRON_SECRET>`:
   ```sql
   select cron.schedule('ai4qi-reminders', '0 7 * * *', $$ select net.http_post(url := 'https://<ref>.supabase.co/functions/v1/send-reminders', headers := jsonb_build_object('Authorization', 'Bearer <CRON_SECRET>', 'Content-Type','application/json')) $$);
   ```
8. Run the data-retention jobs listed in `ai4qi/governance/RETENTION_SCHEDULE.md` (section
   "Scheduled deletion") in the same editor.

## 6. ICO data protection fee (owner pays)

1. https://ico.org.uk/for-organisations/data-protection-fee/ → **Register (pay fee)**.
2. The agent fills in the form from `ai4qi/governance/ICO_REGISTRATION_ANSWERS.md`.
3. **STOP (owner checks the answers, declares and pays).** Note the registration number
   (it starts with ZB or ZA).

## 7. Analytics (optional)

Either:
- **Plausible**: https://plausible.io → add the site. **STOP (owner pays or starts the trial,
  and accepts the DPA).** Or
- **Cloudflare Web Analytics** (free): Cloudflare → **Analytics & Logs** → **Web Analytics** → add
  the site → copy the token. The token is public; it goes in `cloudflare_token` in section 9.

## 8. Security basics (owner)

Turn on two-factor authentication on Cloudflare, Supabase, Anthropic, Resend, GitHub and the owner's
email account. **STOP: this needs the owner's phone or passkey.**

## 9. Hand back to the developer (public values only)

Send these to the developer (in the Claude Code chat is fine; none of them is secret):

| Value | Where it is |
|---|---|
| Live site address | Cloudflare Pages → Custom domains |
| `supabase_url` | Supabase → Project Settings → API → Project URL |
| `supabase_anon_key` | Supabase → Project Settings → API Keys → **publishable** key (starts `sb_publishable_`), never the secret key |
| `build_url` | `https://<ref>.supabase.co/functions/v1/build-audit` |
| Analytics choice | `plausible` + domain, or `cloudflare` + token |
| Legal details | Owner's legal name, postal address, contact email, ICO registration number |

The developer puts these in `ai4qi/app/config.json`, rebuilds, and pushes. Cloudflare Pages then
redeploys by itself.

## 10. Owner's own paperwork (not for the agent)

- Read and sign the DPIA (`ai4qi/governance/DPIA.md`); keep a signed PDF.
- Read the terms of use and the privacy notice once.
- Save a PDF of each processor's DPA with the date (Supabase, Resend, Anthropic, Cloudflare, and
  Plausible if used). See `ai4qi/governance/PROCESSORS_AND_TRANSFERS.md`.
