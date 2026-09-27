# Ai4Qi — owner's launch checklist (data protection)

> DRAFT — not legal advice. Tick each item before switching on sign-in, reminders or building on the
> hosted site. Links checked on 27 September 2026.

## 1. Register and identify yourself

- [ ] **Pay the ICO data protection fee** (most controllers must; tier 1 is the lowest):
      https://ico.org.uk/for-organisations/data-protection-fee/ . Note your registration number.
- [ ] **Decide the controller's legal name** (you as an individual, or a company or charity) and a
      postal address for the privacy notice.
- [ ] **Set up a contact email**, e.g. `privacy@yourdomain`, that you check weekly. Put it in the
      privacy notice, the site footer and the reminder emails.
- [ ] **Irish users:** check whether you need an EU representative (EU GDPR Art. 27).
      Guidance: https://www.dataprotection.ie

## 2. Set up processors safely

- [ ] **Supabase: choose the London region (eu-west-2)** when creating the project. Pick that
      specific region, not a general "Europe" grouping. Regions:
      https://supabase.com/docs/guides/platform/regions
- [ ] **Accept the processors' DPAs** and save a PDF copy of each with the date:

| Company | DPA | How it is accepted |
|---|---|---|
| Supabase | https://supabase.com/legal/dpa | Accepting the terms has the same effect as signing the SCCs (DPA §12.2). If an NHS reviewer wants a signed copy, look in the Supabase dashboard's organisation settings or ask Supabase support |
| Resend | https://resend.com/legal/dpa | Binding when you accept Resend's terms. Sub-processors: https://resend.com/legal/subprocessors |
| Anthropic | https://www.anthropic.com/legal/data-processing-addendum | Part of the Commercial Terms (https://www.anthropic.com/legal/commercial-terms), which apply to API keys from the Console |
| Plausible | https://plausible.io/dpa | An addendum to Plausible's Terms of Service: "Use of the service constitutes acceptance of this DPA. No separate signature is required." |

- [ ] **Record the transfer mechanism** for each US company: UK Extension to the Data Privacy
      Framework (check https://www.dataprivacyframework.gov/list) or SCCs with the UK Addendum
      (in each DPA). Write it in DPIA §4.4 and the privacy notice.
- [ ] **Check Anthropic API data retention** and ask about zero data retention if available.
- [ ] **Two-factor authentication** on Supabase, Resend, Anthropic Console, Plausible, GitHub, your
      DNS provider and the admin email account.
- [ ] **Never commit** the Supabase service key or any API key. Keep them only in Edge Function
      secrets.
- [ ] **Use your own email sending** (Resend SMTP with SPF, DKIM and DMARC) for sign-in emails
      before launch.
- [ ] **Supabase Pro plan** once real users rely on it (backups; no pausing).

## 3. Confirm the app's controls are live

- [ ] Passcode encryption, 15-minute auto-lock, shared-computer mode.
- [ ] Month-only dates and "switch off free text" options.
- [ ] Warnings on the export panel.
- [ ] All scripts and fonts self-hosted; strict Content Security Policy on the hosted site (check
      with https://securityheaders.com or the browser console).
- [ ] Scrub build themes and feedback comments before sending (DPIA action A5).

## 4. Governance documents

- [ ] **Sign your DPIA** (`governance/DPIA.md`) as the data controller and keep it on file. Optional:
      have it reviewed by a freelance data protection consultant or DPO-as-a-service firm (usually a
      few hundred pounds). No NHS organisation needs to approve it. If a user's Trust asks about
      Ai4Qi, send them the DPIA and privacy notice.
- [ ] **Add the retention jobs** listed in DPIA §4.5 (pg_cron SQL).
- [ ] **Publish the privacy notice** (`governance/PRIVACY_NOTICE.md`) at `/privacy` on the hosted
      site, fill in every [placeholder], and link it from every page footer and the sign-in page.
      (The app's in-app page `#/privacy` explains device storage; link the two.)
- [ ] **Write down how you handle requests**: access, deletion (delete the auth user in Supabase →
      Authentication → Users; related rows cascade), and data breaches (ICO within 72 hours:
      https://ico.org.uk/for-organisations/report-a-breach/).

## 5. Tell users

- [ ] Show this message where people start an audit and in the welcome email:

  > **Register your audit with your audit or clinical governance department before you collect
  > any data.** Ai4Qi keeps your records on this device only, but they are still patient data
  > under your organisation's rules. Do not type names or numbers, use a passcode, choose
  > "shared computer" on ward PCs, and keep downloaded files on your organisation's systems.

- [ ] On every built audit: "Check the standard against its source and get your supervisor to
      approve the protocol before you start."

## 6. Review

- [ ] Review the DPIA every 12 months, and before any change that would send audit records to a
      server (that change would also bring in the NHS Data Security and Protection Toolkit).

- [ ] Run migrations 001–005 in the Supabase SQL editor, in order (005 adds self-service account deletion).
