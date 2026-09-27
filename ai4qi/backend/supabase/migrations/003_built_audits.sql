-- Ai4Qi: audits built on request (hosted site).
-- Built audits have ids like B-reusable-ppe-theatre, so trackers and usage events accept them too.
-- Every built protocol is kept in built_audits so the team can review it and add the good ones to the
-- proposed library. Only the server (service role) reads or writes that table.

alter table public.my_audits drop constraint if exists my_audits_audit_id_check;
alter table public.my_audits add constraint my_audits_audit_id_check
  check (audit_id ~ '^(ONA|NNA)-[0-9]{1,5}$' or audit_id ~ '^B-[a-z0-9-]{1,29}$');

alter table public.usage_events drop constraint if exists usage_events_audit_id_check;
alter table public.usage_events add constraint usage_events_audit_id_check
  check (audit_id is null or audit_id ~ '^(ONA|NNA)-[0-9]{1,5}$' or audit_id ~ '^B-[a-z0-9-]{1,29}$');

create table if not exists public.built_audits (
  id         bigint generated always as identity primary key,
  user_id    uuid null references auth.users (id) on delete set null,
  topic      text not null check (char_length(topic) between 1 and 300),
  protocol   jsonb not null,
  created_at timestamptz not null default now()
);
create index if not exists built_audits_user_time_idx on public.built_audits (user_id, created_at);

alter table public.built_audits enable row level security;
-- No policies: the anon and authenticated roles cannot read or write it. The build-audit
-- function uses the service role; the owner reads it in the Supabase dashboard.
revoke all on public.built_audits from anon, authenticated;
