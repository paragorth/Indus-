-- 012: remember which replies to "Report a problem or suggest a change" posts have been emailed.
-- The replies themselves are written by the Ai4Qi team in new_audits/idea_replies.json and published
-- at https://ai4qi.com/data/idea_replies.json (post id, status, reply text; no personal data).
-- send-reminders emails each reply once to the person who posted, then records it here.
-- Only the server (service role) uses this table. Safe to run again.

create table if not exists public.idea_emails (
  idea_id  uuid primary key,
  status   text not null,
  sent_at  timestamptz not null default now()
);
alter table public.idea_emails enable row level security;
