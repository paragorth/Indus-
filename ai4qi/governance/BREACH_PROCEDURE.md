# Ai4Qi — personal data breach procedure

> **Draft for the owner's review — not legal advice.** For one person: the owner, [Owner name].
> Keep a printed copy, and keep the numbers and links below where you can reach them if your own
> email or laptop is the problem.

## 1. What counts as a breach

A personal data breach is a security incident that leads to the accidental or unlawful
**destruction, loss, change, unauthorised disclosure of, or access to** personal data. It covers
accidents as well as attacks, and loss of access as well as leaks.

Examples for Ai4Qi:

- someone other than you reads the Supabase database, or a secret key leaks;
- someone gets into the admin email account and signs in as an admin;
- an email reaches the wrong person;
- personal data we should not hold turns up (for example a patient's name in a build topic);
- the database is deleted or corrupted with no backup.

**Not an Ai4Qi breach:** a leak of audit records from a user's own device or downloaded files.
Ai4Qi never holds those records. The user's organisation is the controller. Tell the user to report
it to their information governance team, and record that you gave that advice.

## 2. The first hour

1. **Write the time you became aware.** The 72-hour clock starts then.
2. **Open a log entry** (template in §6). Write facts as you find them.
3. **Contain it.** Rotate the leaked key, sign the account out, turn off the feature, delete the
   misplaced data. Use the scenarios in §5.
4. **Preserve evidence.** Take screenshots. Export the relevant logs now: Supabase keeps logs for
   only 7 days on Pro (1 day on Free), Resend for 30 days.
5. **Check your processors' notices.** Supabase, Anthropic, Netlify and Plausible promise notice
   within 48 hours; Resend and Cloudflare "without undue delay". A processor's breach of our data is
   our breach to assess.

## 3. Assess the risk (within 24 hours)

Ask:

- **Whose data, and how many people?** Users (clinicians) or, by mistake, patients?
- **What data?** An email address alone is lower risk. Health information, or anything about a
  patient, is higher.
- **Was it seen or taken, or only exposed?** Is it still out there?
- **What could happen to people?** Phishing, embarrassment, professional harm, loss of patient
  confidentiality.
- **Can we reduce the harm?** Deleted quickly, key rotated, recipient confirmed deletion.

Then decide, and write down why:

| Result | What you must do |
|---|---|
| Risk to people **unlikely** | Record it in the log. No report. Keep your reasons |
| Risk **likely** | Report to the ICO without undue delay and **within 72 hours** of becoming aware |
| Risk **high** | Report to the ICO **and** tell the people affected without undue delay |

If unsure, use the ICO self-assessment
(https://ico.org.uk/for-organisations/report-a-breach/personal-data-breach-assessment/) or call the
ICO breach advice line on **0303 123 1113**.

## 4. Telling the ICO and people

**ICO report.** Online form (about 30 minutes; it cannot be saved part-way):
https://ico.org.uk/for-organisations/report-a-breach/personal-data-breach/personal-data-breach-reporting/

Have ready: what happened; when and how you found out; who is or may be affected and roughly how
many; what you are doing about it; your contact details and who else you have told. Report early
even if facts are missing; send the rest later without undue delay. If you report after 72 hours,
give the reasons for the delay.

**Irish users affected.** Irish users are covered by the EU GDPR. Also consider notifying the Irish
Data Protection Commission: https://forms.dataprotection.ie/report-a-breach-of-personal-data

**Others to consider:** the National Cyber Security Centre if an attacker was involved
(https://www.ncsc.gov.uk/section/about-ncsc/incident-management); Report Fraud (the UK's fraud and
cybercrime reporting centre; Police Scotland in Scotland) if a crime is suspected; your insurer.

**Telling users (high risk).** Email each affected person. Use plain words:

> Subject: Ai4Qi — a security problem affecting your account
>
> On [date] we found that [what happened]. This affected [what data about you]. It did not
> affect any audit records, which stay on your own device. We have [what we did]. We suggest you
> [practical step, e.g. be wary of emails asking you to sign in]. We have reported this to the
> Information Commissioner's Office. Questions: [Contact email]. [Owner name], Ai4Qi.

## 5. Scenarios

### A. The Supabase secret (service role) key has leaked

(For example, committed to GitHub or pasted into a chat.) With this key anyone can read and change
every table, including emails.

1. Find and remove the source of the leak first (rewrite the commit, delete the message).
2. Supabase → *Settings → API Keys*: create a new secret key.
3. Put the new key everywhere the old one was used: your shell for `pull_feedback.py`, any scripts.
   The Edge Functions read `SUPABASE_SERVICE_ROLE_KEY`, which Supabase supplies.
4. Delete the old secret key (cannot be undone), or for a legacy `service_role` key, deactivate
   legacy keys (can be undone). Then test "Build an audit" and a reminders dry run
   (`REMINDERS_DRY_RUN=1`). If they fail, re-enable legacy keys, fix the functions, and try again.
5. Check the API logs (*Logs → API*) for requests with that key from unknown addresses since the
   leak.
6. Assess. If there is evidence that someone used the key, treat it as likely risk (emails and
   profiles exposed) and report to the ICO. Consider telling users (phishing risk).

### B. The admin email account is compromised

Admin rights come from the email address in `admins`. An attacker could sign in by magic link and
read raw feedback, the admin statistics and **`mailing_list()`, which returns the email addresses
of opted-in users**.

1. Remove admin rights at once in the SQL editor:
   `delete from public.admins where email = '[admin email]';`
2. End that user's sessions:
   `delete from auth.sessions where user_id = (select id from auth.users where email = '[admin email]');`
   An access token already issued stays valid until it expires (check *JWT expiry* in Auth
   settings).
3. Secure the mailbox: new password, two-factor authentication, remove unknown forwarding rules and
   app passwords.
4. Check Supabase API logs for calls to `mailing_list`, `feedback` and `signup_breakdown` during the
   period.
5. Add the admin back only when the mailbox is secure.
6. Assess. If `mailing_list` was called, users' email addresses were disclosed: likely risk, report
   to the ICO, and tell users to watch for phishing.

### C. Resend delivered an email to the wrong person

Reminder emails hold the audit question, the next step (counts only) and a due date. Sign-in emails
hold a one-time link.

1. Find the message in the Resend dashboard (logs kept 30 days). Note recipient, time and content.
2. If a **sign-in link** went astray: end the intended user's sessions (see B step 2) so the link
   cannot be used later.
3. Ask the wrong recipient to delete the email.
4. If a bug caused it, turn reminders off: `select cron.unschedule('ai4qi-reminders');` and fix the
   code before re-scheduling.
5. Assess. One reminder with no patient data to one wrong person is usually **unlikely** to cause
   risk: record, do not report. A sign-in link that was used, or many emails, needs a fuller
   assessment.

### D. A user reports they typed a patient's name into a build topic

The topic was scrubbed in the browser (titled names and numbers removed) but a plain name gets
through. It was sent to Anthropic (deleted within 30 days) and stored in `built_audits`. A saved
protocol can be served again to anyone who types the same topic.

1. Thank the user. Ask for the approximate time and the topic wording. Do not ask them to repeat
   the patient's details.
2. Find and delete the rows (this also stops the protocol being served again):

   ```sql
   select id, topic_key, created_at from public.built_audits
    where created_at > now() - interval '2 days' and topic ilike '%[a word from the topic]%';
   delete from public.built_audits where topic_key = '[topic_key found above]';
   ```
3. Check the returned protocol text for the name; if it was served to others (`reused = true` rows
   with that key), note how many times.
4. Tell the user how to remove it from their device: the built audit is also saved in their
   browser (`ai4qi_built_v1`); clearing site data removes it.
5. Anthropic deletes API inputs within 30 days; nothing further is needed unless it was flagged.
6. Assess. A name with a topic can reveal health information about one patient. If deleted quickly
   and not served to anyone else, the risk is usually **unlikely** to be significant: record it.
   If the protocol was served to other users, treat as **likely** risk and report to the ICO.
   Suggest the user tells their information governance team, as their organisation may want to
   record it too.

The same steps apply to a patient name in a feedback comment (`feedback` table; delete the row; also
check `new_audits/feedback.json` in the repository).

## 6. Breach log template

Keep one row per incident, reported or not. Do not copy the leaked data into the log.

| Field | Entry |
|---|---|
| Reference | [YYYY-MM-DD-n] |
| Date and time found / how found | |
| Date and time it happened (if known) | |
| What happened | |
| Data and people affected (categories, approximate numbers) | |
| Processor involved | |
| Containment steps and times | |
| Risk assessment (unlikely / likely / high) and reasons | |
| ICO reported? Date, time, ICO reference | |
| Irish DPC reported? | |
| People told? Date and how | |
| Lessons and changes made | |
| Closed (date) | |

The ICO also publishes a template log: https://ico.org.uk/media2/chaiajvy/pdb-log-template.xlsx

## Sources checked (28 Sep 2026)

- ICO, UK GDPR data breach reporting (72 hours; report early, update later; online form; NCSC and
  Report Fraud): https://ico.org.uk/for-organisations/report-a-breach/personal-data-breach/
- ICO, Personal data breach reporting (form entry page):
  https://ico.org.uk/for-organisations/report-a-breach/personal-data-breach/personal-data-breach-reporting/
- ICO, Personal data breaches: a guide (definition; "within 72 hours of becoming aware"; high risk →
  tell individuals; document all breaches; reasons for delay):
  https://ico.org.uk/for-organisations/report-a-breach/personal-data-breach/personal-data-breaches-a-guide/
- ICO, 72 hours — how to respond to a personal data breach (advice line 0303 123 1113; log template):
  https://ico.org.uk/for-organisations/advice-for-small-organisations/personal-data-breaches/72-hours-how-to-respond-to-a-personal-data-breach/
- ICO breach self-assessment: https://ico.org.uk/for-organisations/report-a-breach/personal-data-breach-assessment/
- Irish DPC breach notification: https://www.dataprotection.ie/en/organisations/know-your-obligations/breach-notification
- NCSC incident management: https://www.ncsc.gov.uk/section/about-ncsc/incident-management
- Supabase API keys (rotating a leaked secret or legacy service_role key):
  https://supabase.com/docs/guides/api/api-keys
- Supabase, managing user data (JWT stays valid until it expires):
  https://supabase.com/docs/guides/auth/managing-user-data
- Processor breach-notice terms: https://supabase.com/legal/dpa ,
  https://www.anthropic.com/legal/data-processing-addendum , https://resend.com/legal/dpa ,
  https://www.cloudflare.com/cloudflare-customer-dpa/ , https://www.netlify.com/pdf/netlify-dpa.pdf ,
  https://plausible.io/dpa
