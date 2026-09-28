# Ai4Qi — handling data rights requests and complaints

> **Draft for the owner's review — not legal advice.** For one person: the owner, [Owner name].
> SQL checked against `backend/supabase/migrations/001–006` on 28 September 2026.

## 1. Recognise a request

A request can come by email, in a feedback comment, or by any other route. It need not use legal
words ("please delete my account" is an erasure request). Log every request on the day it arrives:

| Ref | Received (date) | From (email) | Right | ID checked (date) | Clock paused? | Due date | Done (date) | Notes |
|---|---|---|---|---|---|---|---|---|

## 2. Deadlines

- Respond **without undue delay and within one month** of receiving the request.
- If you need to check identity, the month starts when you receive what you asked for.
- If you ask the person to clarify what they want (only when reasonably needed), the clock
  **pauses** on the day you ask and **restarts** the day after they reply.
- You may **extend by up to two further months** if the request is complex or the person has sent
  several requests. Tell them, with reasons, **within the first month**.
- No fee, except for requests that are manifestly unfounded or excessive (then a reasonable fee, or
  refuse and explain why and their right to complain to the ICO).
- Search must be **reasonable and proportionate**.

Practical rule: aim to finish within 7 days. Ai4Qi holds little data about each person.

## 3. Identity check

Ask for ID only if needed, and only as much as needed.

- **Request from the email address on the account:** treat that as enough. Reply to that address.
- **Request from a different address:** reply to the **account address** and ask the person to
  confirm from it. Do not send data to the new address.
- **Feedback given without signing in:** it is linked only to a random device number. Ask for the
  exact comment text, or the device number (shown by opening the browser console on the Ai4Qi site
  and running `localStorage.getItem('ai4qi_device_v1')`).
- Never ask for passport scans or NHS ID.

## 4. Where a person's data is

Find the user id first (Supabase → *SQL Editor*):

```sql
select id, email, created_at, last_sign_in_at from auth.users where email = lower('[email]');
```

| Table | What it holds | Key |
|---|---|---|
| `auth.users` | Email, created, last sign-in, confirmation times | `id` |
| `auth.sessions`, `auth.identities` | Sign-in sessions and identity record | `user_id` |
| `auth.audit_log_entries` | Sign-in events (may include IP address) | `payload->>'actor_id'` |
| `public.profiles` | Grade, specialty, region, work setting, audit purpose, email consents and when changed | `user_id` |
| `public.my_audits` | Audits started and steps | `user_id` |
| `public.usage_events` | Sign-up, audit and active-day events | `user_id` |
| `public.run_reminders` | Reminder question, next step, due date, sends | `user_id` |
| `public.built_audits` | Topics typed and protocols built (user id removed after 90 days) | `user_id` |
| `public.feedback` | Ratings, reasons, comments | `user_id` or `device_id` |
| `public.admins` | Admin email addresses | `email` |

Outside Supabase: Resend logs (search by email in the Resend dashboard; kept 30 days); your support
mailbox; any mailing-list export on your computer; `new_audits/feedback.json` in the repository
(comments only, no ids). Anthropic and Plausible receive nothing that identifies the person.

**Not held by Ai4Qi:** audit records on the person's device. Say so in every reply.

## 5. Handling each right

### Access (subject access request)

Run this with the user id, save the result as a `.json` file and send it to the account email:

```sql
select jsonb_pretty(jsonb_build_object(
  'account',   (select jsonb_build_object('email', email, 'created_at', created_at, 'last_sign_in_at', last_sign_in_at) from auth.users where id = '[uid]'),
  'profile',   (select to_jsonb(p) - 'user_id' from public.profiles p where user_id = '[uid]'),
  'my_audits', (select coalesce(jsonb_agg(to_jsonb(m) - 'user_id'), '[]') from public.my_audits m where user_id = '[uid]'),
  'usage_events', (select coalesce(jsonb_agg(to_jsonb(e) - 'user_id'), '[]') from public.usage_events e where user_id = '[uid]'),
  'reminders', (select coalesce(jsonb_agg(to_jsonb(r) - 'user_id'), '[]') from public.run_reminders r where user_id = '[uid]'),
  'built_audits', (select coalesce(jsonb_agg(jsonb_build_object('topic', topic, 'created_at', created_at, 'reused', reused)), '[]') from public.built_audits where user_id = '[uid]'),
  'feedback',  (select coalesce(jsonb_agg(jsonb_build_object('audit_id', audit_id, 'rating', rating, 'reasons', reasons, 'comment', comment, 'created_at', created_at)), '[]') from public.feedback where user_id = '[uid]')
));
```

Add sign-in events if asked (`select created_at, ip_address, payload from auth.audit_log_entries
where payload->>'actor_id' = '[uid]'`). Include the supplementary information: purposes, lawful
bases, recipients, retention, rights and the right to complain. The privacy notice covers these;
attach or link it. This JSON file also meets a **portability** request.

### Rectification

Profile fields, consents and reminders can be changed in the app (account page, My audits). For
anything else (for example a wrong email address), make the change in the SQL editor or the
Supabase dashboard, and confirm by email.

### Erasure

**In the app:** the account page has **Delete my account**. It calls `delete_my_account()`, which
deletes the person's `auth.users` row. That cascades to `profiles`, `my_audits`, `usage_events`
and `run_reminders`, and sets `user_id` to null in `feedback` and `built_audits` (the rows stay,
no longer linked to the person). The app then signs them out. Audits on the device are not touched.

**By email:** do the same in *Authentication → Users → Delete user*, or:

```sql
-- Only if the person also wants their comments and build topics removed:
delete from public.feedback     where user_id = '[uid]';
delete from public.built_audits where user_id = '[uid]';
-- Then the account (cascades as above):
delete from auth.users where id = '[uid]';
```

Also delete: any mailing-list export holding the address; support emails you no longer need; their
row in `admins` if any. Resend logs expire within 30 days. Supabase Pro backups roll off within 7
days. Tell the person both points.

You may refuse erasure only in narrow cases (for example, you need the data for a legal claim).
Ai4Qi will rarely have such a reason.

### Objection and restriction

- **Emails:** turning off the consent boxes (account page) stops news and sponsor emails at once.
  Remove them from any list you have exported.
- **Reminders:** off per audit in My audits; or delete their `run_reminders` rows.
- **Other processing (legitimate interests):** stop unless you have compelling reasons that override
  their interests. In practice, offer deletion of the relevant rows or of the whole account.
- **Restriction** (for example while accuracy is disputed): keep the data but do not use it. Easiest
  route: remove their reminders and consents and note the restriction in the request log until
  resolved.

## 6. Complaints about how we handle data

Since 19 June 2026 every controller must give people a clear way to complain about data
protection, **acknowledge a complaint within 30 days**, look into it, and tell the person the
outcome.

- The 30 days start the day after you receive it. If the last day is a weekend or public holiday,
  you have until the next working day.
- Set a mailbox auto-reply on [Contact email] that acknowledges receipt. Keep a record of it.
- Arrange cover for holidays (an auto-reply counts as acknowledgement; set a reminder to reply in
  full).
- In the outcome letter, tell the person they can complain to the ICO
  (https://ico.org.uk/make-a-complaint/, 0303 123 1113), or, in Ireland, the Data Protection
  Commission (https://www.dataprotection.ie).

## 7. Template replies

**Acknowledgement (all requests and complaints)**

> Thank you for your message of [date]. We have received your [request / complaint] and will reply
> in full by [date, one month on]. [If needed: To protect your data, please confirm this request by
> replying from the email address you use to sign in to Ai4Qi.] — [Owner name], Ai4Qi,
> [Contact email]

**Access**

> Attached is a copy of the personal data Ai4Qi holds about you, as a JSON file that opens in any
> text editor. It covers your account, profile, audit progress, usage events, reminders, audits
> you built and feedback you gave while signed in. Ai4Qi does not hold your audit records: they stay
> on your own device. How and why we use this data, who processes it and how long we keep it are
> set out in our privacy notice: [privacy notice URL]. If you are unhappy with our reply you can
> complain to the Information Commissioner's Office: https://ico.org.uk/make-a-complaint/

**Erasure done**

> We have deleted your Ai4Qi account on [date], with your profile, audit progress, usage records and
> reminders. [Your feedback comments and build topics have also been deleted. / Feedback you gave
> and topics you built are kept without any link to you.] Copies in our email provider's logs
> expire within 30 days, and in our database backups within 7 days. Audits stored on your own
> device are not affected; you can delete them in Ai4Qi under My audits.

**Extension**

> Your request needs more time because [it is complex / you have sent several requests]. We will
> reply by [date, up to three months from receipt].

**Clarification**

> So that we can find what you need, could you tell us [what]? We will reply within one month of
> hearing from you. (The time limit is paused until then.)

**Feedback given without signing in**

> Feedback given without signing in is linked only to a random device number, so we cannot tell
> which comments are yours from your email address. If you send us the text of your comment, or the
> device number (we can explain how to find it), we will [send / delete] it.

## Sources checked (28 Sep 2026)

- ICO, A guide to subject access requests (updated 16 July 2026 for the Data (Use and Access) Act
  2025: one month; extension by up to two further months; identity; stopping the clock for
  clarification; fees; reasonable and proportionate search):
  https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/subject-access-requests/a-guide-to-subject-access/
- ICO, individual rights (erasure, rectification, objection, restriction, portability):
  https://ico.org.uk/for-organisations/uk-gdpr-guidance-and-resources/individual-rights/individual-rights/
- ICO, New data protection complaints law now in force (23 June 2026):
  https://ico.org.uk/about-the-ico/media-centre/news-and-blogs/2026/06/new-data-protection-complaints-law-now-in-force/
- ICO, What do we do when we receive a complaint? (acknowledge within 30 days; how to count):
  https://ico.org.uk/for-organisations/how-to-deal-with-data-protection-complaints/what-do-we-do-when-we-receive-a-complaint/
- ICO complaints page: https://ico.org.uk/make-a-complaint/
- Supabase, managing user data (deleting users; cascades):
  https://supabase.com/docs/guides/auth/managing-user-data
- Retention of processor logs and backups: https://resend.com/pricing , https://supabase.com/pricing
