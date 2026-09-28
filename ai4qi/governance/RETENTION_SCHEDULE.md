# Ai4Qi — retention schedule

> **Draft for the owner's review — not legal advice.** Matches DPIA §4.5 and the privacy notice.
> Table and column names checked against `backend/supabase/migrations/001–006` on 28 September
> 2026. Run the SQL below once, in the Supabase SQL editor, after migrations 001–006.

## Schedule

| Data item | Where | Retention | How it is deleted |
|---|---|---|---|
| Email, sign-in record (`auth.users`, sessions, identities) | Supabase | Until the user deletes the account, or **24 months after last activity** | In-app "Delete my account", owner deletion, or job **J5** |
| Profile, incl. email consents and `consent_updated_at` (`profiles`) | Supabase | Life of the account | Cascades when the account is deleted |
| "My audits" progress (`my_audits`) | Supabase | Life of the account | Cascades |
| Usage events (`usage_events`) | Supabase | **24 months**, then weekly totals only. The single `signup` row per account stays for the life of the account (the admin figures need it) | Job **J4**; cascades on account deletion |
| Reminders (`run_reminders`) | Supabase | Until reminders are turned off or the audit is closed (the app deletes the row); otherwise **90 days after the due date** | App (`syncRun`); job **J2**; cascades |
| Built audits (`built_audits`) | Supabase | Topic and protocol kept for library review. **User id removed after 90 days** | Job **J1**; user id set to null on account deletion |
| Feedback (`feedback`) | Supabase | **24 months**, then user id, device id and comment removed (rating and reasons kept) | Job **J3**; user id set to null on account deletion |
| Admin list (`admins`) | Supabase | While the person is an admin | Owner deletes the row by hand |
| Weekly usage totals (`usage_weekly_archive`, created below) | Supabase | While the service runs | No personal data |
| Supabase auth, API and function logs | Supabase | Free plan 1 day; Pro plan 7 days | Automatic |
| Supabase backups | Supabase | Pro plan: daily, kept 7 days (none on Free) | Automatic. Deleted data leaves backups within 7 days |
| Reminder and sign-in email logs | Resend | 30 days | Automatic |
| Build prompts and outputs | Anthropic | Deleted within 30 days (longer only if flagged under the Usage Policy or required by law) | Automatic |
| Visit statistics | Plausible | Aggregates while the account runs; no identifiers | Delete the site in Plausible when no longer needed |
| Hosting logs | Cloudflare or Netlify | Provider default | Automatic |
| Copied feedback comments (`new_audits/feedback.json`) | GitHub repository | While useful to the library | Owner edits the file; no device or user ids are copied |
| Mailing-list exports (`mailing_list()` output) | Owner's computer | Until the send is finished | Owner deletes the file |
| Support emails | [Mailbox provider] | 24 months after the last message | Owner deletes; set a mailbox rule if possible |
| Rights-request and complaint log | Owner's records | 3 years after closing | Owner deletes |
| Breach log | Owner's records | Life of the service | Keep; hold only the facts needed |

### Records on the user's device (the user controls these)

Ai4Qi cannot see or delete these. The user, or their organisation, decides how long to keep them.

| Data item | Browser storage key | How the user deletes it |
|---|---|---|
| Encrypted audit records and audit details | `ai4qi_vault_v1`, `ai4qi_runs_enc_v1` (`localStorage`, or `sessionStorage` in shared-computer mode) | "Delete this audit and its data from this device" in the app; closing the browser in shared-computer mode; clearing site data |
| Older unencrypted records, if any | `ai4qi_runs_v1` | Same as above |
| Audits built (last 30) | `ai4qi_built_v1` | Clear site data |
| Feedback waiting to be sent | `ai4qi_feedback_v1` | Sent when online; clear site data |
| Random device id | `ai4qi_device_v1` | Clear site data |
| Consent choice waiting for sign-in | `ai4qi_consent_pending` | Cleared after sign-in; clear site data |
| Demo mode flag | `ai4qi_demo` | "Demo mode off"; clear site data |
| Sign-in session | `sb-…-auth-token` | Sign out; clear site data |
| Library copy for offline use | Service-worker cache | Clear site data |
| Downloaded files (backup, CSV, Excel, slides) | The user's file system | The user deletes them under their organisation's rules |

App guidance: delete the audit from the device once it has been presented and saved on the
organisation's systems.

## SQL for the scheduled jobs

Before you start: *Database → Extensions* → enable **pg_cron** (already needed for reminders).
pg_cron runs in UTC. Each job is safe to run more than once. Test each `where` clause first as a
`select count(*)` before scheduling it.

```sql
-- J1. Unlink built audits from accounts after 90 days (daily 02:10 UTC).
select cron.schedule('ai4qi-built-unlink', '10 2 * * *', $$
  update public.built_audits set user_id = null
   where user_id is not null and created_at < now() - interval '90 days'
$$);

-- J2. Purge reminders 90 days after their due date (daily 02:20 UTC).
select cron.schedule('ai4qi-reminders-purge', '20 2 * * *', $$
  delete from public.run_reminders
   where due_date < (now() at time zone 'Europe/London')::date - 90
$$);

-- J3. Feedback older than 24 months: keep rating and reasons, remove the links to a person
--     (weekly, Sunday 02:30 UTC). The new device id is unique per row and cannot be linked back.
select cron.schedule('ai4qi-feedback-expire', '30 2 * * 0', $$
  update public.feedback
     set user_id = null, comment = '', device_id = 'expired-' || replace(id::text, '-', '')
   where created_at < now() - interval '24 months'
     and device_id not like 'expired-%'
$$);
-- (To delete old feedback instead, use:
--  delete from public.feedback where created_at < now() - interval '24 months';)

-- J4. Usage events older than 24 months: keep weekly totals, delete the raw rows
--     (monthly, 1st at 02:40 UTC). The signup row stays for the life of the account.
create table if not exists public.usage_weekly_archive (
  week   date   not null,
  event  text   not null,
  users  bigint not null,
  events bigint not null,
  primary key (week, event)
);
alter table public.usage_weekly_archive enable row level security;
revoke all on public.usage_weekly_archive from anon, authenticated;

select cron.schedule('ai4qi-usage-rollup', '40 2 1 * *', $$
  with gone as (
    delete from public.usage_events e
     where e.day < date_trunc('week', now() - interval '24 months')::date
       and e.event <> 'signup'
     returning e.day, e.event, e.user_id
  )
  insert into public.usage_weekly_archive as w (week, event, users, events)
  select date_trunc('week', day)::date, event, count(distinct user_id), count(*)
    from gone group by 1, 2
  on conflict (week, event) do update
    set users = w.users + excluded.users, events = w.events + excluded.events
$$);

-- J5. Delete accounts with no activity for 24 months (daily 02:50 UTC). "Activity" is the later of
--     the last sign-in and the last active day (a user who stays signed in does not sign in again).
--     Admins are never deleted by this job. Deleting the auth user cascades to profiles, my_audits,
--     usage_events and run_reminders, and sets user_id to null in feedback and built_audits.
select cron.schedule('ai4qi-inactive-accounts', '50 2 * * *', $$
  delete from auth.users u
   where greatest(
           coalesce(u.last_sign_in_at, u.created_at),
           coalesce((select max(e.day)::timestamptz from public.usage_events e
                      where e.user_id = u.id and e.event = 'active_day'), u.created_at)
         ) < now() - interval '24 months'
     and lower(u.email) not in (select a.email from public.admins a)
$$);

-- J6. Keep pg_cron's own run history short (weekly, Sunday 03:00 UTC).
select cron.schedule('ai4qi-cron-history', '0 3 * * 0', $$
  delete from cron.job_run_details where end_time < now() - interval '30 days'
$$);
```

### The 30-day warning before J5

The DPIA promises a warning email 30 days before an inactive account is deleted. The first account
can reach 24 months without activity in **October 2028** (launch is 2 October 2026). Set a calendar
reminder for **1 September 2028**. From then on, on the 1st of each month, run this and email each
person (one email each, not a group email):

```sql
select u.email
  from auth.users u
 where greatest(
         coalesce(u.last_sign_in_at, u.created_at),
         coalesce((select max(e.day)::timestamptz from public.usage_events e
                    where e.user_id = u.id and e.event = 'active_day'), u.created_at)
       ) between now() - interval '24 months' and now() - interval '23 months'
   and lower(u.email) not in (select a.email from public.admins a);
```

Suggested text: "You have not used your Ai4Qi account for nearly two years. We will delete it, with
your profile, audit progress and reminders, on [date]. To keep it, sign in before then. Audits on
your own device are not affected."

### Checking the jobs

```sql
select jobname, schedule, active from cron.job order by jobname;
select jobid, status, return_message, start_time
  from cron.job_run_details order by start_time desc limit 20;
-- Stop a job: select cron.unschedule('ai4qi-usage-rollup');
```

## Sources checked (28 Sep 2026)

- Supabase Cron (job name, schedule, SQL; times in GMT; `cron.job_run_details` is never cleaned up
  automatically): https://supabase.com/docs/guides/cron/quickstart
- Supabase, managing user data (deleting from `auth.users`; foreign keys with `on delete cascade`):
  https://supabase.com/docs/guides/auth/managing-user-data
- Supabase pricing (log retention 1 day Free / 7 days Pro; Pro daily backups kept 7 days):
  https://supabase.com/pricing
- Resend pricing (30-day data retention): https://resend.com/pricing
- Anthropic API retention (30 days):
  https://privacy.claude.com/en/articles/7996866-how-long-do-you-store-my-organization-s-data
- Plausible DPA (no raw IP or User-Agent stored; deletion on instruction): https://plausible.io/dpa
