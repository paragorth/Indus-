# Ai4Qi — Data Protection Impact Assessment (draft)

> **DRAFT — not legal advice.** Ai4Qi is an independent service, not run by or for an NHS
> organisation. The owner, as data controller for the account data, completes and signs this DPIA.
> An independent review (for example a freelance data protection consultant or DPO-as-a-service) is
> optional but recommended. NHS organisations whose staff use Ai4Qi may read it when deciding
> whether to allow the tool; their approval is not required for Ai4Qi to operate.
>
> Structure follows the ICO's sample DPIA template (Steps 1–7). Facts about the system were checked
> against the code on 27 September 2026: `ai4qi/backend/README.md`, `backend/supabase/migrations/001–004`,
> `backend/supabase/functions/build-audit` and `send-reminders`, and `ai4qi/app/app.js` / `export.js`.
> Controls marked **[live]** were being added to the code at the time of writing and must be
> confirmed as live before sign-off.

> **Update 27 Sep 2026:** the controls marked [planned] are now live and tested: passcode encryption
> (AES-256-GCM, PBKDF2-SHA-256, 310,000 iterations), 15-minute idle lock, shared-computer mode,
> month-only dates, free-text off, export warnings, encrypted backups, scrubbing of build themes and
> feedback comments, self-hosted scripts and fonts with a strict Content Security Policy (no
> third-party script, font or connection hosts on the hosted site except Supabase and the chosen
> analytics service), and a self-service "Delete my account" button (migration 005). Actions A1, A4,
> A5 and A6 below are done; the residual-risk scores should be re-checked by the reviewer.

| | |
|---|---|
| Controller | Paraggarg Limited (company number 16367622), trading as Ai4Qi. Registered office: 40 St James Buildings, St James Street, Taunton, Somerset TA1 1JR |
| ICO registration number | ZB979579 (Paraggarg Limited, tier 1; Ai4Qi added as a trading name) |
| Contact | privacy@ai4qi.com (security reports: security@ai4qi.com) |
| DPIA author | [NAME] |
| Version / date | 0.1 draft, 27 September 2026 |
| Next review | Before launch; then every 12 months or on any change listed in §2.6 |

---

## Step 1 — Identify the need for a DPIA

Ai4Qi is a free web app that helps UK and Irish clinicians (mostly doctors in training) find, design
and run clinical audits and quality improvement (QI) projects. It has three parts:

1. **A library** of published audits, standards and ready-made audit protocols (no personal data).
2. **Build an audit**: a user types a theme (e.g. "VTE prophylaxis on admission") and Claude, a
   large language model from Anthropic, writes a protocol with a data-collection template.
3. **Run an audit**: the user enters or uploads audit records (one row per patient case), sees
   results against the standard, and downloads a data sheet, slides, CSV and calendar file.

A DPIA is needed, or at least strongly advisable, because:

- The app is designed for use with **patient-level audit data** (health data, special category under
  UK GDPR Art. 9), even though that data is kept on the user's device and never reaches Ai4Qi.
  Getting the design wrong could expose health data.
- It uses **new technology**: an AI model generates clinical audit protocols that clinicians may
  act on.
- It is offered to NHS and Irish health service staff, who will often use **shared hospital
  computers**.
- Several **processors outside the UK** (US companies) handle account and usage data.

The ICO lists "innovative technology" and "sensitive data" among the criteria that make a DPIA
likely to be required. Even where the server-side processing alone would not strictly require one,
NHS organisations will expect to see a DPIA before their staff use the tool.

---

## Step 2 — Describe the processing

### 2.1 Nature of the processing

**A. On the user's device only (Ai4Qi never receives this data)**

| Data | Where it is kept | Notes |
|---|---|---|
| Audit records (patient-level rows) | Browser storage on the user's device (`localStorage`; encrypted, or `sessionStorage` in shared-computer mode **[live]**) | Never sent to any Ai4Qi server or processor. |
| Audit details: title, hospital/practice, department, audit lead, team members, supervising consultant, start date | Same | Staff names. Included in the downloaded slides. Not sent to the server. |
| "The change" description and date | Same | Free text, scrubbed (see below). |
| Protocols the user has built (last 30) | `localStorage` key `ai4qi_built_v1` | No patient data. |
| Feedback queue | `localStorage` key `ai4qi_feedback_v1` | Sent to Supabase when online, if the backend is on. |
| Random device id | `localStorage` key `ai4qi_device_v1` | Sent with feedback only. |
| Sign-in session token | `localStorage` key `sb-…-auth-token` (set by supabase-js) | Only if the user signs in. |
| Library data and app files | Service-worker cache | For offline use. No personal data. |

**How audit records are reduced before storage** (function `cleanRecord`, `prepareImport`, `scrub`
in `app.js`):

1. Only fields in the audit's template are kept. On upload, any column not in the template is
   dropped, and so is any column whose heading looks like an identifier (name, NHS number, CHI,
   MRN, hospital number, date of birth, address, postcode, phone, email, next of kin, GP name),
   unless it is the template's audit-code field.
2. A patient or hospital number in the audit-code field is replaced with a sequential code
   (P001, P002…). The original-to-code lookup exists only in memory during that import and is not
   stored. Short codes the user has already assigned (e.g. "A12") are kept as typed.
3. Text in free-text and non-matching choice fields is cut to 300 (or 120) characters and scrubbed
   of: 10-digit/NHS-style numbers, UK postcodes, email addresses, UK phone numbers, names with a
   title (Mr, Mrs, Ms, Miss, Mx, Dr, Prof, Sister, Nurse), "DOB/born" followed by a date, hospital
   numbers like K1234567, and any run of 7–12 digits.
4. Yes/no, number, date and choice fields are normalised to those types.

This is **pseudonymisation / de-identification, not anonymisation**. Dates of admission or
procedure, ward, age bands, rare conditions and free text that the scrubber does not recognise can
still identify a patient, especially to colleagues in the same hospital. The records remain
**patient data held by the user under their own organisation's rules**, and the organisation (not
Ai4Qi) is the controller of them.

**Controls added 27 Sep 2026 [live]** (code in progress in `app.js`, to be confirmed):

- Passcode encryption of stored records: AES-GCM (256-bit key) derived from a user passcode with
  PBKDF2-SHA-256 (310,000 iterations, random 16-byte salt). The key is held only in memory.
  The passcode cannot be recovered; forgetting it means erasing the data on that device.
- Automatic lock after 15 minutes without activity (key dropped from memory).
- "Shared computer" mode: records kept in `sessionStorage`, which the browser clears when it closes.
- Optional month-and-year-only dates, per audit (existing dates shortened).
- Option to switch off free-text fields, per audit (existing free text deleted).
- Warnings on exports: "These files hold de-identified patient records. Keep them on your
  organisation's systems and share them only inside it."
- Self-hosting of all third-party code and fonts, with a strict Content Security Policy on the
  hosted site. (Today the page loads Google Fonts, supabase-js from jsDelivr, and SheetJS, ExcelJS
  and JSZip from cdnjs.)

**B. On Ai4Qi's servers and processors (optional backend, switched on by `config.json`)**

| Processing | Data | Stored in | Who can read it |
|---|---|---|---|
| Sign-in by emailed magic link (no password) | Email address; Supabase Auth also records sign-in times and technical logs (may include IP address) | Supabase `auth.users`, Auth logs | Owner (dashboard) |
| Profile (all optional) | Grade/role, specialty, region — each from a fixed list | `profiles` | The user; owner. Admin statistics show only totals, with groups under 5 suppressed |
| Feedback on audits | Audit id, thumbs up/down, up to 6 fixed reasons, optional comment ≤500 characters, random device id, user id if signed in, app version, time | `feedback` | Admins only (RLS). Public sees only totals via `feedback_summary()` |
| My audits tracker | User id, audit id (library or built), step, dates | `my_audits` | The user |
| Usage events | User id, event (sign-up, audit started / status changed / completed, active day), audit id, time | `usage_events` | Nobody through the API; admins see weekly totals via `weekly_stats()` and `signup_breakdown()` |
| Built audits | User id, the typed theme (≤300 characters), a topic key, the generated protocol, whether reused, time | `built_audits` | Server only (service role); owner in dashboard |
| Reminders (opt-in) | User id, run id, the audit question (≤200), next step text made of counts (e.g. "Collect cycle 1 data (32 of 40 entered)"), due date, opt-in flag, times sent | `run_reminders` | The user; the `send-reminders` job |
| Reminder emails | Email address, audit question, next step, due date | Resend (sending and logs) | Owner (Resend dashboard) |
| Audit building | The build prompt: fixed instructions, the typed theme, and excerpts from the Ai4Qi library (published audits, standards, proposed audits). **No patient data and no user identifiers** | Sent from the `build-audit` Edge Function to the Anthropic API | Anthropic, as processor |
| Visitor analytics | Page views with cleaned address (route only, e.g. `#/proposed/ONA-012`; search words and filters removed; only `utm_*`, `ref` and `from` kept), referrer, browser/OS and country derived by Cloudflare Web Analytics (no cookies, nothing stored on the device) | Cloudflare | Owner |
| Feedback comments copied into the library | When the owner runs `pull_feedback.py`, free-text comments with their audit id, rating and reasons (no user or device id) are copied into `new_audits/feedback.json` | The project's GitHub repository (GitHub, Inc., US) | Owner; anyone with access to the repository. The owner reads new comments first and deletes any that name a person |

Admins are listed by email in the `admins` table and can read raw feedback and aggregate
statistics. Row level security (RLS) is enabled on every table.

**C. The version inside Claude (claude.ai artifact)**

A copy of the app (built by `build_artifact.py`) is published as a claude.ai artifact. There:

- Audit records are kept in the viewer's browser storage in the same way (local only).
- "Build an audit" uses the **viewer's own Claude account** (the artifact's sampling capability),
  so the build prompt (theme plus library excerpts) is processed by Anthropic under the viewer's
  own agreement with Anthropic, not under Ai4Qi's API key.
- Downloads use Claude's download capability.
- Whether the Supabase features and analytics run inside the artifact depends on the `config.json`
  copied into it and on claude.ai's own page restrictions. **To confirm** before publishing a
  configured copy.

### 2.2 Scope

- **Data subjects**: users (clinicians, students, QI staff) — adults, UK and Ireland mainly.
  Patients appear only in data held on users' devices, never in Ai4Qi's systems.
- **Special category data**: none processed by Ai4Qi by design. Risk of accidental receipt through
  free text (feedback comment, typed theme), treated in Step 5.
- **Volume**: pilot scale (hundreds to low thousands of users). Revisit if above 10,000 users.
- **Geography**: UK and Ireland users; data stored in the EU/UK region chosen for Supabase;
  processors in the US (see §4.4).

### 2.3 Context

- Users are health professionals bound by professional confidentiality and their employer's
  information governance (IG) policies. Many are trainees rotating between hospitals every 4–12
  months, often on shared hospital computers and personal phones.
- Clinical audit is part of the NHS's quality work. Local audit departments normally require
  registration; Caldicott principles apply to the use of patient data for audit.
- Users would reasonably expect an audit tool not to send patient data off their device, and the app
  says so on its privacy page (`#/privacy`).
- AI-generated content is a known concern in healthcare; outputs are labelled "Built for you – not
  yet run".

### 2.4 Purposes

- Help clinicians design and complete closed-loop audits quickly and to a good standard.
- Let users keep track of their audits and get reminders (optional).
- Improve the audit library using feedback.
- Measure use (sign-ups, audits started and completed) in aggregate, to report to sponsors and
  justify the service.
- Protect the service from abuse (rate limits, device id, daily build cap).

### 2.5 Benefits

Better-designed audits, more closed loops, less time spent on paperwork, and patient data kept off
third-party servers by design.

### 2.6 Changes that require this DPIA to be reviewed

Any server-side storage or syncing of audit records; team or multi-user audits; sending records or
record-derived text to an AI model; new processors; new analytics; integration with NHS systems;
use outside the UK and Ireland; a paid tier.

---

## Step 3 — Consultation process

| Who | How | Status |
|---|---|---|
| Users (trainee doctors) | In-app feedback; short survey of pilot users on privacy expectations and shared-computer use | [to do] |
| Independent data protection reviewer (optional) | Review of this DPIA | [to do] |
| Users' NHS organisations (on request) | May review the on-device design and user guidance when deciding whether their staff can use Ai4Qi; not an approval Ai4Qi needs | [as requested] |
| A Clinical Safety Officer (CSO) | Review of the DCB0129/0160 assessment (§4.7) and AI content risks | [to do] |
| A local audit department | Check the "register your audit" guidance and export warnings | [to do] |
| Security review | Review of RLS policies, Edge Functions and CSP | [to do] |
| Processors | Rely on published DPAs and security documentation | [to do] |

Record the outcome of each consultation here before sign-off.

---

## Step 4 — Necessity and proportionality

### 4.1 Roles

- **Ai4Qi owner** — controller for account, profile, feedback, usage, tracker, built-audit and
  reminder data, and analytics.
- **User's employing organisation** — controller for the patient audit data the user collects;
  the user processes it under that organisation's authority. Ai4Qi supplies software that runs on
  the user's device and has no access to that data, so it is neither controller nor processor of it.
  **Confirm this view with the reviewer.** It depends on patient data never reaching Ai4Qi, which
  the code currently ensures.
- **Supabase, Resend, Anthropic, Cloudflare (hosting, visitor statistics, email forwarding) and GitHub** — processors for Ai4Qi (see `PROCESSORS_AND_TRANSFERS.md`).

### 4.2 Lawful bases (UK GDPR Art. 6)

| Processing | Lawful basis | Notes |
|---|---|---|
| Account (email) and sign-in | Art. 6(1)(f) legitimate interests (alternatively 6(1)(b), a service the user asks for) | Needed to send the sign-in link and keep "My audits" |
| Profile (grade, specialty, region) | 6(1)(f) | Optional; "Prefer not to say" default |
| My audits tracker and usage events | 6(1)(f) | Aggregate reporting; nobody reads raw events via the API |
| Feedback incl. device id | 6(1)(f) | Improve library; device id for spam limit and one-vote counting |
| Built audits (theme, protocol, user id) | 6(1)(f) | Deliver the build; reuse; daily cap; library review |
| Reminder emails | 6(1)(f) | User turns them on per audit. Service messages, not marketing, so PECR marketing consent does not apply |
| Visitor analytics | 6(1)(f) | Cloudflare Web Analytics sets no cookies and stores nothing on the device, so PECR reg. 6 consent is not needed. Check the current PECR position after the Data (Use and Access) Act 2025 |
| Security logs (Supabase, Resend) | 6(1)(f) | Security and abuse prevention |

No Art. 9 condition is relied on, because Ai4Qi does not intend to process special category data.
Health data submitted by mistake in free text (feedback comment, typed theme) will be deleted when
found (see Step 6).

#### Legitimate interests assessment (short form)

- **Purpose test.** Ai4Qi has a legitimate interest in providing a working audit tool, keeping users'
  progress, reminding them of deadlines they asked about, improving the library, preventing abuse
  and showing sponsors that the tool is used. Users benefit directly.
- **Necessity test.** Email is the least data that allows password-free sign-in. Profile fields are
  optional and use fixed lists. Usage events hold no free text. Analytics are cookieless and
  aggregate. Each purpose could not reasonably be met with less data.
- **Balancing test.** The data are low-risk professional data about adults acting in a work capacity.
  Users would expect a sign-in service to hold their email and progress. Admin reports suppress
  groups under five. Nothing is sold, used for advertising or used to train AI models. Users can
  delete their account on request and turn reminders off at any time. Main residual concern:
  transfers to US processors, handled by DPAs and transfer safeguards (§4.4). **Outcome:**
  legitimate interests is appropriate, subject to the retention limits below and an easy way to
  delete an account.

### 4.3 Data minimisation and quality

- Patient-level data never leaves the device (verified in `syncRun`, which sends only the audit
  id, step, audit question, next-step text built from counts, and due date).
- Records are cut to template columns and scrubbed before storage (§2.1).
- The AI prompt contains only the typed theme and library excerpts. It contains no records, no
  email address and no user id (the Edge Function uses the user id only for the daily cap, and does
  not pass it to Anthropic).
- Profile fields are fixed lists enforced by database constraints. Feedback reasons are a fixed list;
  comment capped at 500 characters.
- Analytics addresses are cleaned by `cleanPageUrl()`: search words, filters and unknown routes
  are never sent.
- Grade/specialty/region breakdowns suppress cells under 5.
- Accuracy: users can edit their profile and tracker; generated protocols are labelled as not yet
  run and the evidence lines are filtered to library ids that exist.

### 4.4 Processors and international transfers

All to be checked and recorded before launch: signed or accepted DPA, sub-processor list, transfer
mechanism. For transfers from the UK to the US, the lawful route is either the **UK Extension to the
EU–US Data Privacy Framework** (the "UK–US data bridge", only if the company is certified and has
opted in to the UK Extension) or the **EU SCCs with the UK Addendum / UK IDTA**, plus a transfer risk
assessment. Transfers to the EU/EEA are covered by UK adequacy regulations.

| Processor | What it processes | Where | Transfer mechanism to check | DPA |
|---|---|---|---|---|
| Supabase Pte. Ltd (Singapore), the contracting party under the DPA; sub-processors include Supabase, Inc. (US, support) and AWS | Database (all tables above), Auth (email, sign-in logs), Edge Functions and their logs | Project region: **London (eu-west-2)**. Choose that specific region, not a "general" region grouping. Support and some sub-processors may access from elsewhere, including the US | Not on the DPF list; Singapore has no UK adequacy regulations. EU SCCs with the UK Addendum (IDTA Addendum), incorporated in the DPA, plus a transfer risk assessment | https://supabase.com/legal/dpa |
| Resend (Plus Five Five, Inc., US) | Recipient email, reminder content, delivery logs, sent from mail.ai4qi.com; also Supabase Auth sign-in emails if Resend is used as the SMTP provider | Account data, email metadata and logs stored in the **United States**, even when sending from the Ireland (eu-west-1) region | **UK Extension to the EU–US DPF** (certified, non-HR data); fallback: SCCs and UK Addendum in the DPA | https://resend.com/legal/dpa |
| Anthropic PBC (US) | Build prompts (theme and library excerpts) and generated protocols from the hosted site | US (check Anthropic's current processing locations and data-residency options) | Not on the DPF list. EU SCCs with the UK Addendum (IDTA Addendum) in the DPA, plus a transfer risk assessment; check API retention period and zero-retention options | https://www.anthropic.com/legal/data-processing-addendum (part of the Commercial Terms) |
| Cloudflare, Inc. (US) | Cloudflare Pages: web server logs (IP address, user agent) of every visitor. Cloudflare Web Analytics: page views, referrer, browser/OS and country, without cookies. Email Routing: forwards messages sent to privacy@ai4qi.com and security@ai4qi.com | Global network | **UK Extension to the EU–US DPF** (certified, non-HR data); fallback: SCCs and UK Addendum in the DPA | https://www.cloudflare.com/cloudflare-customer-dpa/ |
| GitHub, Inc. (US) | Source code repository; also `new_audits/feedback.json`, holding feedback comments copied by `pull_feedback.py` (no user or device ids), which could contain personal data if someone typed it | US | **UK Extension to the EU–US DPF** (certified, non-HR data) | https://github.com/customer-terms/github-data-protection-agreement |
| Font and script CDNs (Google Fonts, jsDelivr, cdnjs) — **hosted site: no longer used (self-hosted); Claude-artifact version still loads Google Fonts** | Visitor IP address and user agent when files are fetched | Global | Removed by the planned self-hosting | n/a |

For the Claude artifact version, building happens under the viewer's own Anthropic account and
terms; Ai4Qi's processor contract does not cover it. Say so in the privacy notice.

**Ireland.** Users in Ireland are covered by the EU GDPR. If Ai4Qi has no establishment in the EU
and offers its service to people in Ireland, check whether an **EU representative (EU GDPR Art. 27)**
is required, or whether the exemption for occasional, low-risk processing applies.

### 4.5 Retention (full schedule and deletion jobs: `RETENTION_SCHEDULE.md`)

| Data | Proposed retention | How |
|---|---|---|
| Account (email) and profile | Until the user asks for deletion, or 24 months after last sign-in (email a warning 30 days before) | Scheduled SQL job; deleting the auth user cascades to `profiles`, `my_audits`, `usage_events`, `run_reminders` |
| Feedback rows | 24 months; then delete the row, or remove `device_id` and `user_id` and keep the rating/reasons | Scheduled SQL job. Comments copied into the library by `pull_feedback.py` carry no device or user id |
| Usage events | 24 months, then aggregate into weekly totals and delete raw rows | Scheduled SQL job |
| My audits tracker | Life of the account | Cascade on account deletion |
| Run reminders | Deleted automatically when the audit is closed or reminders are turned off (already in `syncRun`); purge rows with a due date more than 90 days in the past | Add a scheduled purge |
| Built audits | Protocol and theme kept for library review; set `user_id` to null after 90 days (the daily cap only needs 24 hours) | Scheduled SQL job |
| Admin list | While the person is an admin | Manual |
| Supabase Auth and Edge Function logs | 1 day (free plan) or 7 days (paid plan) | Supabase |
| Resend email logs | 30 days | Resend |
| Anthropic API inputs/outputs | Deleted within 30 days under Anthropic's commercial API terms, with the exceptions they list (request zero data retention if offered) | Anthropic |
| Cloudflare Web Analytics statistics | Aggregate; as set by Cloudflare | Cloudflare |
| Backups (Supabase Pro) | Platform default (daily backups, typically 7 days on Pro) | Supabase |
| Records on user devices | Under the user's and their organisation's control. App guidance: delete the audit from the device once it is presented and archived on the organisation's systems | "Delete this audit and its data from this device" button; shared-computer mode **[live]** |

Supabase's free plan has no backups and pauses inactive projects; use the Pro plan once real users
depend on it.

### 4.6 Data subject rights

| Right | How it is met |
|---|---|
| Be informed | Privacy notice at `/privacy` (see `PRIVACY_NOTICE.md`) and the in-app page `#/privacy` |
| Access | Email [CONTACT EMAIL]; owner exports the user's rows from each table (SQL by `user_id`) within one month |
| Rectification | Profile and tracker editable in the app; email for anything else |
| Erasure | Email request; owner deletes the auth user (cascades) and nulls `user_id` in `feedback` and `built_audits`. Self-service "Delete my account" button on the account page (migration 005), plus email requests |
| Restrict / object | Email request; reminders can be turned off in the app; profile fields can be cleared |
| Portability | Export on request (JSON/CSV) |
| Automated decisions | None with legal or similar effects |
| Complaint | ICO, https://ico.org.uk/make-a-complaint/ (Irish users: Data Protection Commission) |

Feedback given without signing in is linked only to a random device id, which Ai4Qi cannot link to
a person. It can be deleted if the user supplies the id (shown on request) or the text of the comment.

Patients' rights over audit data sit with the user's organisation, which controls that data.

### 4.7 Clinical safety (DCB0129 / DCB0160) and medical device status

**Assessment (reasoned view, to be confirmed by a Clinical Safety Officer):**

- DCB0129 (manufacturers) and DCB0160 (deploying organisations) apply to health IT systems whose
  failure could affect patient care, particularly systems used in direct care.
- Ai4Qi is used for clinical audit and QI, which is a secondary use. It does not display or
  change a patient's record, does not support decisions about an individual patient, and does not
  connect to clinical systems. **Likely view: it is not a clinical system in scope of DCB0129/0160.**
- However, generated protocols quote standards and propose changes to practice (e.g. a checklist or
  default). If a standard is wrong or misquoted, a change made after an audit could affect care
  indirectly.
- **Recommendation:** record this assessment; ask a CSO to confirm it; keep a short hazard log for
  the AI content anyway (hazard: incorrect or invented standard or evidence; controls in Step 6);
  revisit if Ai4Qi ever supports decisions about individual patients, or if an NHS organisation
  formally deploys it (they may then apply DCB0160 locally).
- **Medical device (UK MDR 2002 / MHRA):** software not intended for the diagnosis, prevention,
  monitoring or treatment of an individual patient is not a medical device. Ai4Qi's intended purpose
  (audit and QI design) should be stated in the terms and on the site so that this stays clear.

### 4.8 NHS Data Security and Protection Toolkit (DSPT)

- The DSPT is required of organisations that have access to NHS patient data and systems.
- **Ai4Qi processes no NHS patient data and has no access to NHS systems, so the DSPT is not
  required at present.** State this in answers to NHS IG teams, with this DPIA.
- **It would become required (or be requested) if Ai4Qi:**
  - stored, synced or backed up audit records on its servers (e.g. cloud backup or team audits);
  - received record-derived data from users (e.g. sending records to an AI model for analysis);
  - connected to NHS systems (EPR, NHS login, NHS mail directory);
  - supplied a service to an NHS organisation under contract that involves patient data.
- NHS organisations that formally procure digital tools may also ask for the **Digital Technology
  Assessment Criteria (DTAC)**, which covers clinical safety, data protection, technical security,
  interoperability and accessibility. Consider preparing it if a trust adopts Ai4Qi.

---

## Step 5 — Identify and assess risks

Scoring as in the ICO template. Likelihood: Remote / Possible / Probable. Severity: Minimal /
Significant / Severe. Overall: Low / Medium / High. Scores are **before** the measures in Step 6
(but after controls already in the code).

| # | Risk (source, impact on individuals) | Likelihood | Severity | Overall |
|---|---|---|---|---|
| R1 | **Re-identification from pseudonymised records on shared NHS computers.** Records sit in browser storage of a shared ward PC; the next user opens Ai4Qi or the browser's developer tools and sees patient rows (dates, ward, clinical details) that colleagues can link to real patients. Breach of patient confidentiality. | Probable | Significant | High |
| R2 | **Identifiers in free text.** A user types a name without a title, a hospital number in another format, an address or a rare-condition description; the regex scrubber misses it. Identifiable patient data stored and exported. | Probable | Significant | High |
| R3 | **Exports shared by email.** CSV, backup JSON (currently unencrypted and containing all records and staff names), Excel or slides sent to personal email, WhatsApp or cloud drives, or presented outside the organisation. | Possible | Significant | Medium |
| R4 | **Third-party script compromise.** A compromised CDN file (supabase-js, SheetJS, ExcelJS, JSZip, Google Fonts CSS, the Cloudflare Web Analytics beacon) runs in the page and reads browser storage, including decrypted records while unlocked. Mass exposure of patient data across users. | Remote | Severe | Medium |
| R5 | **Account or email breach.** Theft of the Supabase service key, an admin's email account (admin rights come from the email address), or a user's email (magic link). Exposure of emails, profiles, feedback comments, reminder text; misuse of admin statistics. | Possible | Significant | Medium |
| R6 | **Misuse of reminders.** Audit question or next-step text containing patient information emailed via Resend; reminders sent to the wrong address; the email reveals the user's work to others with access to their inbox. Also: the reminder feature used to spam. | Remote | Minimal | Low |
| R7 | **AI-generated content inaccuracies (clinical safety).** Claude invents or misquotes a standard, target or evidence; a trainee runs the audit and a department changes practice based on a wrong standard. Indirect harm to patients; reputational harm to the user. | Possible | Significant | Medium |
| R8 | **Users uploading identifiable data despite warnings.** A full EPR extract with names and NHS numbers is uploaded; header matching fails (e.g. column named "Pt"), so identifiers land in a template field; or identifiers are typed into the audit-code field in a format the code keeps as is. | Possible | Significant | Medium |
| R9 | **Patient or personal data in text sent to Ai4Qi's servers.** Users type patient details into the build theme (sent to Anthropic and stored in `built_audits`) or a feedback comment. | Possible | Significant | Medium |
| R10 | **Loss of audit data.** Browser storage cleared, passcode forgotten, shared-computer mode closes the tab. Loss of the user's work (availability), and pressure to keep copies in unsafe places. | Probable | Minimal | Medium |
| R11 | **Row level security or configuration error.** A new table without RLS, a wrong policy, or the service key committed to the repository. Exposure of server data. | Remote | Significant | Low |
| R12 | **Staff personal data in exports.** Audit lead, team and supervisor names in slides; low risk but present. | Possible | Minimal | Low |
| R13 | **International transfers.** US processors subject to US law enforcement access. | Remote | Minimal | Low |
| R14 | **Claude artifact version.** Users assume Ai4Qi's safeguards and contracts cover building inside Claude; the theme is processed under their own Claude account and its settings. | Possible | Minimal | Low |

---

## Step 6 — Identify measures to reduce risk

| Risk | Measures (E = exists in code; P = planned this week; R = recommended) | Effect | Residual risk | Approved |
|---|---|---|---|---|
| R1 Shared computers | P: passcode encryption (AES-GCM, PBKDF2 310k); P: auto-lock after 15 min; P: shared-computer mode (session storage only); E: "Delete this audit and its data from this device"; R: ask "Is this a shared computer?" on first use and default to shared mode on unknown devices; R: guidance to prefer an organisation-issued personal device; R: lock on sign-out and on tab hide after a shorter time | Reduced | Low–Medium (a weak passcode, or a PC left unlocked within 15 min) | [ ] |
| R2 Free-text identifiers | E: template-only columns, audit codes, regex scrub, 300-character cap; P: switch off free-text per audit; P: month-only dates; R: show a warning when a free-text field contains a capitalised word pair or numbers; R: templates prefer choice fields over text (add to `BUILD_RULES`); R: tell users the scrubber is a safety net, not a guarantee (the in-app page already says so) | Reduced | Medium | [ ] |
| R3 Exports | P: warning on the export panel; R: encrypt the backup file with the passcode; R: omit free-text fields from CSV by default; R: add a "share only inside your organisation" line in the slide footer (slides already say "check before sharing outside the department"); R: aggregate-only option for slides | Reduced | Low–Medium | [ ] |
| R4 Script compromise | P: self-host all third-party code and fonts; P: strict CSP (`default-src 'self'`, `connect-src` only Supabase, the build function and Cloudflare Web Analytics; no inline script; `frame-ancestors 'none'`); R: Subresource Integrity on anything still external; R: pin versions and review updates | Reduced | Low | [ ] |
| R5 Account breach | E: magic link (no passwords stored); E: RLS on all tables; admins see aggregates, not raw events; E: service key only in Edge Function secrets; R: use a dedicated admin mailbox with two-factor authentication; R: two-factor authentication on Supabase, Resend, Anthropic, Cloudflare and GitHub accounts; R: own SMTP with SPF/DKIM/DMARC; R: short magic-link expiry; R: review the admin list quarterly | Reduced | Low | [ ] |
| R6 Reminders | E: only question, next-step counts and due date sent; E: opt-in per audit; E: ≤6 sends, ≥3 days apart, ≤50 audits per user; E: users cannot change send counts; E: `CRON_SECRET` protects the job; R: keep the audit question from the protocol, not from user free text (as now) | Reduced | Low | [ ] |
| R7 AI accuracy | E: prompt requires verbatim standard wording from the library's standards list, or "Local standard"; E: evidence lines filtered to existing library ids; E: "Built for you – not yet run" label; R: show "Check the standard against the source before you start; get your supervisor to approve the protocol" on every built audit; R: owner reviews `built_audits` and promotes good ones; R: CSO-confirmed hazard log (§4.7) | Reduced | Low–Medium | [ ] |
| R8 Identifiable uploads | E: identifier-looking columns dropped; E: preview before storage ("Nothing has been stored yet") listing kept and dropped columns; R: refuse the import (not just drop) when more than N scrubs occur and tell the user to use the Ai4Qi data sheet; R: warn if the audit-code column contains values longer than 6 characters that look like hospital numbers | Reduced | Low–Medium | [ ] |
| R9 Server-bound free text | E: theme ≤300 characters, comment ≤500 characters; R: run the same `scrub()` on the theme and feedback comment before sending; R: "Do not enter patient information" next to both boxes (the sign-in page already says it); R: owner deletes any such content found, and treats it as a possible breach | Reduced | Low | [ ] |
| R10 Data loss | E: backup download; E: "no space left" warnings; R: prompt a backup at each stage change; R: explain clearly that a lost passcode cannot be reset | Reduced | Low | [ ] |
| R11 Configuration | E: RLS and revoked grants in every migration; E: README warns never to commit the service key; R: automated test that anon and authenticated roles cannot read other users' rows; R: secret scanning on the repository | Reduced | Low | [ ] |
| R12 Staff names | R: note in the details form that names appear in the slides; allow initials | Accepted | Low | [ ] |
| R13 Transfers | R: accept DPAs; check DPF/UK Extension or SCCs + UK Addendum; transfer risk assessment; prefer EU/UK regions | Reduced | Low | [ ] |
| R14 Artifact version | R: note on the artifact's privacy page that building uses the viewer's Claude account under Anthropic's terms | Reduced | Low | [ ] |

**Personal data breach procedure.** Owner assesses any incident within 24 hours; reports to the ICO
within 72 hours if it is likely to result in a risk to people; tells affected users without undue
delay if the risk is high; records every incident. A leak of patient data from a user's device is
the user's organisation's incident: tell the user to report it to their IG team.

---

## Step 7 — Sign off and outcomes

| Item | Name / position / date | Notes |
|---|---|---|
| Measures approved by | | Integrate actions back into the project plan, with date and owner |
| Residual risks approved by | | If accepting any residual high risk, consult the ICO before going ahead |
| Independent reviewer's advice provided (optional) | | Should advise on compliance, Step 6 measures and whether processing can proceed |
| Summary of reviewer's advice | | |
| Reviewer's advice accepted or overruled by (owner) | | If overruled, give reasons |
| Comments | | |
| Owner sign-off (data controller) | | Name, date |
| Clinical Safety Officer review | | Confirm the DCB0129/0160 assessment in §4.7 |
| Consultation responses reviewed by | | If the decision departs from individuals' views, give reasons |
| This DPIA will be kept under review by | | Named person responsible for review |

### Action log

| # | Action | Owner | Due | Done |
|---|---|---|---|---|
| A1 | Confirm the planned controls (encryption, lock, shared mode, month-only dates, free-text off, export warnings, self-hosting, CSP) are live | | Before launch | [x] 27 Sep 2026 |
| A2 | Accept/sign DPAs; record transfer mechanisms (§4.4) | | Before launch | [ ] |
| A3 | Add retention jobs (§4.5) | | Before launch | [ ] |
| A4 | Add "Delete my account" | | Within 3 months | [x] 27 Sep 2026 |
| A5 | Scrub build themes and feedback comments before sending (R9) | | Before launch | [x] 27 Sep 2026 |
| A6 | Encrypt backup files (R3) | | Within 3 months | [x] 27 Sep 2026 |
| A7 | CSO review of §4.7 and AI hazard log | | Before launch | [ ] |
| A8 | Check EU representative requirement for Irish users | | Before launch | [ ] |
