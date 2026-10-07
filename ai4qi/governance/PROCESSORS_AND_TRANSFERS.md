# Ai4Qi — processors, sub-processors and international transfers

> **Draft for the owner's review — not legal advice.** Checked on 28 September 2026. Re-check the
> Data Privacy Framework (DPF) list and each sub-processor list every 6 months, and before adding a
> processor. Save a dated PDF of every DPA you rely on in `[secure folder]`.

## How to read the transfer column

- **EU/EEA:** covered by UK adequacy regulations. No extra paperwork.
- **US company on the UK Extension to the EU–US DPF** ("UK–US data bridge"): covered by UK adequacy
  regulations for that company, as long as its certification is active and covers non-HR data.
- **Anyone else outside the UK** (US companies not on the UK Extension; Singapore): rely on the
  **EU Standard Contractual Clauses (SCCs) with the ICO's UK Addendum**, which each DPA below
  incorporates, plus a short **transfer risk assessment (TRA)**. The notes in the last column are the
  TRA summary; keep a one-page TRA per processor using the ICO's tool if a reviewer asks for it.

DPF status below is from the official participant list (dataprivacyframework.gov), downloaded
28 September 2026 (list dated "Monday, September 28, 2026 6:00 AM EST").

## Register

| Processor | Service | Personal data processed | Location | DPA and how accepted | Transfer mechanism | Retention by the processor | Transfer risk note |
|---|---|---|---|---|---|---|---|
| **Supabase Pte. Ltd** (Singapore). Sub-processors include Supabase, Inc. (US, support), AWS (hosting) and others | Database, sign-in, Edge Functions (`build-audit`, `send-reminders`), pg_cron | Everything in `RECORD_OF_PROCESSING.md` rows 1–7 and 11–12: email, profile, consents, progress, events, reminders, built topics, feedback; auth and function logs (may include IP address) | Project region **London (eu-west-2)**. Support and sub-processors may access from elsewhere, incl. the US | https://supabase.com/legal/dpa (Version 1, 1 Aug 2026). Part of the Terms of Service; "acceptance of the Agreement shall have the same effect as signing the SCCs" (§12.2) and the UK Addendum (Sch. 2 ¶2.3). Security incidents notified "where feasible, within forty-eight (48) hours" | **Not on the DPF list** (neither Supabase Pte. Ltd nor Supabase, Inc.). Singapore has no UK adequacy regulations. Rely on SCCs + UK Addendum in the DPA, plus TRA | Data kept for the life of the contract; on request within 30 days of expiry, a copy is returned, then deleted. Logs: Free plan 1 day, Pro 7 days. Backups (Pro): daily, kept 7 days | Low. Data are stored in London and are low-sensitivity professional data (no patient data). Remote access by support is the main transfer. Subscribe to sub-processor updates at https://supabase.com/legal/customer-resources/subprocessor-list (list dated 1 Jun 2026) |
| **Resend** (Plus Five Five, Inc., US) | Sign-in emails (if used as Supabase SMTP) and reminder emails | Recipient email address; email subject and body (audit question, next step, due date); delivery logs | Account data, email metadata and logs **stored in the US** whatever sending region is chosen. Sending region can be Ireland (eu-west-1). Hosting sub-processor: AWS (US) | https://resend.com/legal/dpa. Entered into on acceptance of Resend's Terms; SCCs "deemed entered into". Breach notice "without undue delay" | **On the UK Extension** (status "Active – Re-certification under Review", non-HR data: yes): https://www.dataprivacyframework.gov/participant/8907 . Fallback: SCCs + UK Addendum in DPA §6.4 | Data retention 30 days (Free and Pro plans). Customer data deleted within 90 days of account termination | Low. Email content is short and non-clinical. If the DPF certification lapses, the DPA's SCCs apply automatically. Choose the Ireland sending region |
| **Anthropic PBC** (US) | "Build an audit" on the hosted site, via the API key held in Supabase Edge Function secrets | The build prompt: fixed instructions, the scrubbed topic (≤300 characters) and library extracts; the generated protocol. **No email, user id or audit records** | By default Anthropic "may route customer traffic to select countries in the US, Europe, Asia and Australia". Workspace data at rest per the Console workspace geo setting | https://www.anthropic.com/legal/data-processing-addendum (effective 24 Feb 2025). Part of the Commercial Terms (https://www.anthropic.com/legal/commercial-terms), which apply to Console API keys; SCCs "deemed to have been executed"; UK Addendum in Schedule 3. Breach notice "in any event within 48 hours". Sub-processors: https://www.anthropic.com/subprocessors | **Not on the DPF list.** Rely on SCCs + UK Addendum in the DPA, plus TRA | API inputs and outputs deleted "within 30 days of receipt or generation", unless a zero data retention agreement applies, a Usage Policy flag (up to 2 years; classifier scores up to 7 years) or the law requires longer. Anthropic "may not train models on Customer Content" | Low. The prompt holds no identifiers by design. The residual risk is a user typing a patient's name into the topic (see `BREACH_PROCEDURE.md`). Optional: ask Anthropic about zero data retention; pin US-only inference with `inference_geo` if a reviewer wants one known country (costs 1.1×) |
| **Cloudflare, Inc.** (US) — if the site is on Cloudflare Pages | Hosting the static site; DNS; TLS; optional Web Analytics | Visitor IP address, browser, pages requested (logs). No account data | Global network | https://www.cloudflare.com/cloudflare-customer-dpa/ (Version 6.4, effective 3 Apr 2026). Incorporated by reference into the Self-Serve Subscription Agreement (https://www.cloudflare.com/terms/) | **On the UK Extension** ("Active – Re-certification under Review", non-HR data: yes): https://www.dataprivacyframework.gov/participant/5666 . DPA §6.4: DPF transfers are not Restricted Transfers; SCCs + UK Addendum as fallback | Provider default for logs (not published per plan for Pages; not verified) | Low. Only technical data about visitors |
| **Netlify, Inc.** (US) — only if Netlify is chosen instead of Cloudflare | Hosting the static site | Visitor IP address, browser, pages requested (logs) | US and global CDN | https://www.netlify.com/pdf/netlify-dpa.pdf (last updated 9 Jun 2026). Forms part of the Self-Serve Subscription Agreement. Breach notice "within forty-eight (48) hours" | **On the UK Extension** ("Active", non-HR data: yes): https://www.dataprivacyframework.gov/participant/6208 . DPA §14.4: IDTA applies where the UK Extension does not | Provider default (not verified) | Low. Only technical data about visitors |
| **Plausible Insights OÜ** (Estonia) — only if analytics are switched on | Visitor statistics | Page address (cleaned), referrer, browser, OS, country. Raw IP and User-Agent are hashed with a daily salt and "never stored" | EU: servers in Germany; sub-processors Hetzner (DE), Bunny (SI), UpCloud (FI) | https://plausible.io/dpa (last updated Sep 2026). "Use of the service constitutes acceptance of this DPA. No separate signature is required." Breach notice "no later than 48 hours" | EU/EEA: UK adequacy regulations | Aggregate statistics until you delete the site or account | Very low. No identifiers leave the browser |
| **GitHub, Inc.** (US) | Source code repository | The owner's own account and commit details. **Also:** `new_audits/feedback.json` holds feedback comments copied by `pull_feedback.py` (no device or user ids). A comment could contain personal data if someone typed it | US | GitHub customer terms and DPA (https://github.com/customer-terms/github-data-protection-agreement — page returned 403 to our checker; open it in a browser) | **On the UK Extension** ("Active", non-HR data: yes): https://www.dataprivacyframework.gov/participant/6174 | Until deleted from the repository and its history | Low. Before each `pull_feedback.py` run, read new comments and delete any that name a person |
| **[Mailbox provider]** for [Contact email] | Support, rights requests, complaints | Names, email addresses, message content | [Check] | [Provider's DPA] | [Check DPF list or DPA] | [Provider default] | [Add after choosing] |

Not processors for Ai4Qi:

- **Anthropic, when Ai4Qi is used inside Claude (claude.ai).** Building there uses the viewer's own
  Claude account, under the viewer's own agreement with Anthropic.
- **The user's device and browser.** Audit records never leave it.

## When a processor changes

1. Read the new sub-processor or DPA notice. Object within the notice period if the change adds risk
   (Supabase: 30 days' notice if you subscribe; Resend: object within 14 days).
2. Update this register and `RECORD_OF_PROCESSING.md`.
3. If a US company's DPF status becomes "Inactive", switch the transfer column to "SCCs + UK
   Addendum" and add a TRA.
4. If the change is material, update the privacy notice and tell signed-in users.

## Actions before launch

- [ ] Accept each DPA in use, save a dated PDF.
- [ ] Update the privacy notice: Supabase's contracting entity is **Supabase Pte. Ltd (Singapore)**,
      not a US company; Resend relies on the **UK Extension**; Anthropic and Supabase rely on **SCCs
      with the UK Addendum**.
- [ ] Choose Resend's **Ireland (eu-west-1)** sending region.
- [ ] Delete the hosting row you do not use (Cloudflare or Netlify).
- [ ] Write the one-page TRAs for Supabase and Anthropic.

## Sources checked (28 Sep 2026)

- DPF participant list (official download from https://www.dataprivacyframework.gov/list ,
  "DataPrivacyFrameworkParticipantsList.xlsx", dated 28 Sep 2026): Resend, Cloudflare, Netlify and
  GitHub on the UK Extension; no entry for Anthropic or Supabase.
- ICO, Is the restricted transfer covered by adequacy regulations? (EEA full adequacy; US partial,
  UK Extension only): https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/international-transfers/adequacy-regulations/is-the-restricted-transfer-covered-by-adequacy-regulations/
- ICO, How does the UK Extension to the EU-US Data Privacy Framework work?:
  https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/international-transfers/adequacy-regulations/how-does-the-uk-extension-to-the-eu-us-data-privacy-framework-work/
- ICO, International transfers guide (IDTA, Addendum, TRA): https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/international-transfers/international-transfers-a-guide/
- Supabase DPA, sub-processor list (1 Jun 2026) and pricing: https://supabase.com/legal/dpa ,
  https://supabase.com/legal/customer-resources/subprocessor-list , https://supabase.com/pricing
- Resend DPA, sub-processors, regions, pricing: https://resend.com/legal/dpa ,
  https://resend.com/legal/subprocessors , https://resend.com/docs/dashboard/domains/regions ,
  https://resend.com/pricing
- Anthropic DPA, Commercial Terms, retention, server locations, data residency:
  https://www.anthropic.com/legal/data-processing-addendum ,
  https://www.anthropic.com/legal/commercial-terms ,
  https://privacy.claude.com/en/articles/7996866-how-long-do-you-store-my-organization-s-data ,
  https://privacy.claude.com/en/articles/7996890-where-are-your-servers-located-do-you-host-your-models-on-eu-servers ,
  https://platform.claude.com/docs/en/manage-claude/data-residency
- Cloudflare DPA and terms: https://www.cloudflare.com/cloudflare-customer-dpa/ ,
  https://www.cloudflare.com/terms/
- Netlify DPA: https://www.netlify.com/pdf/netlify-dpa.pdf
- Plausible DPA and privacy policy: https://plausible.io/dpa , https://plausible.io/privacy
