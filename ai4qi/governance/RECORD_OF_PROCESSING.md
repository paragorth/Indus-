# Ai4Qi — record of processing activities (UK GDPR Article 30)

> **Draft for the owner's review — not legal advice.** Facts about the app were checked against the
> code on 28 September 2026 (`app/app.js`, `backend/supabase/migrations/001–006`,
> `backend/supabase/functions/build-audit` and `send-reminders`, `app/_headers`, `DEPLOY.md`).
> Update this record whenever a row changes, and review it with the DPIA every 12 months.

## Do we need this record?

Yes. Article 30(5) gives organisations with fewer than 250 employees a limited exemption, but it
does not apply to processing that:

- is not occasional; or
- is likely to result in a risk to people's rights and freedoms; or
- involves special category or criminal offence data.

Ai4Qi signs people in, stores their profiles and sends reminders every day. That processing is
**not occasional**, so the exemption does not apply to it. The ICO also says it is good practice to
document everything even where you need not. This record therefore covers every activity below.

Ai4Qi does not intend to process special category data. Health data could arrive by mistake in a
build topic or a feedback comment; that is handled as a possible breach (see `BREACH_PROCEDURE.md`).

## Controller details

| | |
|---|---|
| Controller | [Owner name], trading as Ai4Qi |
| Address | [Postal address] |
| Contact | [Contact email] |
| ICO registration number | [ICO registration number] |
| Data protection officer | None. Not required: Ai4Qi is not a public authority, and its core activities are not large-scale monitoring or large-scale special category processing |
| Joint controllers | None |
| EU representative (Irish users) | [Decision pending — DPIA action A8] |
| Record owner and last review | [Owner name], [date] |

## Not Ai4Qi's processing

Audit records (patient-level rows), audit details (team names, hospital) and the results code stay
in the user's browser, encrypted with the user's passcode (PBKDF2-SHA-256, 310,000 iterations →
AES-256-GCM; 15-minute lock; shared-computer mode keeps them only until the browser closes). They
are never sent to Ai4Qi or its processors. The user's employing organisation is the controller of
that data. It is listed here only so that the boundary is recorded.

## Processing activities

Abbreviations: **LI** = legitimate interests, Art. 6(1)(f). **Users** = clinicians and students who
use Ai4Qi, mostly in the UK and Ireland. **Supabase** project region: London (eu-west-2).

| # | Activity | Purpose | Lawful basis | Data subjects | Data categories | Recipients / processors | International transfers | Retention | Security measures |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Account and sign-in | Sign people in by emailed link (no password); keep their progress | LI (alternatively 6(1)(b)) | Users who sign in | Email address; account id; sign-up and last sign-in times; auth logs (may include IP address and browser) | Supabase (database, auth); Resend (sends the sign-in email, if set as SMTP) | Supabase Pte. Ltd (Singapore) and its sub-processors, incl. US; Resend (US) | Until deleted by the user, or 24 months after last activity; auth logs per Supabase plan (Free 1 day, Pro 7 days) | Magic link only; row level security (RLS) on every table; service key only in Edge Function secrets; two-factor authentication on provider accounts |
| 2 | Profile | Tailor the service; count users by group in totals | LI | Users who sign in | Grade, specialty, region, work setting, audit purpose (all optional, fixed lists) | Supabase | As row 1 | Life of the account | RLS: users see only their own row; database checks allow only list values |
| 3 | Email consents and news / sponsor emails | Send Ai4Qi news and sponsor offers only to people who opted in; keep proof of the choice | Consent, Art. 6(1)(a) (and PECR reg. 22) for the emails; LI for keeping the consent record | Users who tick either box | Two yes/no choices (both off by default); time of last change; email address (via `mailing_list()`) | Supabase; [email-sending tool used for news, e.g. Resend] | As row 1 | Choices: life of the account. Mailing-list exports: delete after each send | Admin-only function; addresses never given to sponsors |
| 4 | "My audits" progress and usage counting | Show users their progress; count sign-ups, audits started and completed | LI | Users who sign in | Audit id, step, dates; events (sign-up, audit started / step changed / completed, active day) | Supabase | As row 1 | Progress: life of the account. Raw events: 24 months, then weekly totals | RLS; nobody can read raw events through the API; admins see totals only |
| 5 | Reminders | Email a user when a step they chose is due | LI (service messages, not marketing) | Users who turn reminders on | User id, run id, audit question (≤200 characters), next step made of counts, due date, send count; email address at send time | Supabase (`send-reminders`); Resend | As row 1; Resend stores account data in the US | Deleted when reminders are turned off or the audit is closed; purge 90 days after due date. Resend logs 30 days | Opt-in per audit; ≤6 sends, ≥3 days apart; ≤50 rows per user; job protected by `CRON_SECRET` |
| 6 | Build an audit (topics and saved builds) | Write an audit protocol from the topic typed; reuse it for the same topic; daily limit; library review | LI | Users who sign in and build | Topic text (≤300 characters, scrubbed of titled names and numbers in the browser); topic key; protocol; user id; time | Supabase; Anthropic (receives the topic and library extracts, **no** email or user id) | Supabase as row 1; Anthropic (US; inference may run in US, Europe, Asia or Australia) | Protocol and topic kept for library review; user id removed after 90 days. Anthropic: deleted within 30 days | Server-only table (no API access); prompt carries no identifiers; Anthropic may not train on customer content |
| 7 | Feedback on proposed audits | Improve the library; show vote totals | LI | Anyone who gives feedback | Audit id, thumbs up/down, fixed reasons, optional comment (≤500 characters, scrubbed), random device id, user id if signed in, app version, time | Supabase; copies of comments (no ids) in the Ai4Qi repository on GitHub | As row 1; GitHub (US) for copied comments | 24 months, then device id, user id and comment removed (or row deleted) | RLS: only admins read rows; public sees totals only; 30 rows per device per day |
| 8 | Visitor analytics (only if switched on) | See which pages are used and where visitors come from | LI (no cookies or device storage, so no PECR consent needed) | Website visitors | Cleaned page address, referrer, browser, OS, country. Raw IP and User-Agent not stored (Plausible) | Plausible (or Cloudflare Web Analytics if chosen instead) | Plausible: EU only (Germany). Cloudflare: US | Aggregate statistics while the account runs | Search words and filters removed before sending; no identifiers |
| 9 | Website delivery and security logs | Serve the site; protect it from abuse | LI | Website visitors | IP address, browser, pages requested, time | Cloudflare Pages (Cloudflare, Inc.) | US | Provider default | HTTPS, HSTS, strict Content Security Policy, no third-party scripts except those listed in `app/_headers` |
| 10 | Support and rights requests by email | Answer questions, complaints and data rights requests | LI; Art. 6(1)(c) for rights requests and complaints | Anyone who writes in | Name, email address, message content, request log | [Mailbox provider] | [Depends on mailbox provider] | 24 months after the last message; rights-request and complaint log 3 years | Mailbox with two-factor authentication; no patient data asked for |
| 11 | Admin statistics and sponsor reports | Report use of Ai4Qi to the owner and sponsors | LI | Users who sign in | Weekly counts; sign-ups by grade, specialty, region, work setting, audit purpose, NHS/HSE email (yes/no), consent totals. Groups under 5 hidden | Owner and listed admins; sponsors receive totals only | None for totals | Totals kept while the service runs | Admin-only functions checked by `is_admin()`; suppression of groups under 5; CSV contains no names or emails |
| 12 | Admin list | Decide who can see feedback and statistics | LI | Owner and admins | Email address | Supabase | As row 1 | While the person is an admin | Only the owner adds rows (SQL editor); review quarterly |
| 13 | Breach log | Meet Art. 33(5) duty to record breaches | Art. 6(1)(c) | People affected by an incident | Facts of each incident, kept to the minimum | Owner only | None | Life of the service | Stored offline or in the owner's encrypted drive |

## Processors

Details, DPAs and transfer mechanisms: see `PROCESSORS_AND_TRANSFERS.md`. Retention and deletion
jobs: see `RETENTION_SCHEDULE.md`.

## Sources checked (28 Sep 2026)

- ICO, Documentation: who needs to document their processing activities (small and medium-sized
  organisations): https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/accountability-and-governance/documentation/who-needs-to-document-their-processing-activities/
- ICO, What do we need to document under Article 30 of the UK GDPR?:
  https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/accountability-and-governance/documentation/what-do-we-need-to-document-under-article-30-of-the-gdpr/
- ICO notice on that guidance: "Due to changes made by the Data (Use and Access) Act, this guidance
  is under review and may be subject to change":
  https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/accountability-and-governance/documentation/
- ICO, Data protection officers (when a DPO is required):
  https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/accountability-and-governance/guide-to-accountability-and-governance/data-protection-officers/
- Supabase pricing (log retention, backups): https://supabase.com/pricing
- Supabase DPA (importer Supabase Pte. Ltd, Singapore): https://supabase.com/legal/dpa
- Resend pricing (30-day data retention) and regions (account data stored in the US):
  https://resend.com/pricing , https://resend.com/docs/dashboard/domains/regions
- Anthropic retention for API data and server locations:
  https://privacy.claude.com/en/articles/7996866-how-long-do-you-store-my-organization-s-data ,
  https://privacy.claude.com/en/articles/7996890-where-are-your-servers-located-do-you-host-your-models-on-eu-servers
- Anthropic Commercial Terms ("Anthropic may not train models on Customer Content from Services"):
  https://www.anthropic.com/legal/commercial-terms
- Plausible DPA (EU hosting; raw IP and User-Agent not stored): https://plausible.io/dpa
