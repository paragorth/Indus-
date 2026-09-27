-- Ai4Qi stage 1 backend: anonymous feedback on proposed audits, optional sign-in profiles, admins.
--
-- Run once in the Supabase SQL editor (or with `supabase db push`). Safe to re-run: every object is
-- created with IF NOT EXISTS / OR REPLACE and policies are dropped before being recreated.
--
-- No patient data is stored anywhere. Feedback rows hold an audit id (e.g. ONA-012), a rating,
-- fixed reasons, an optional free-text comment of up to 500 characters, a random per-device id
-- and, only when the person is signed in, their auth user id.

create extension if not exists pgcrypto with schema extensions;

-- ---------------------------------------------------------------- tables

create table if not exists public.feedback (
  id          uuid primary key default gen_random_uuid(),
  audit_id    text not null
              check (audit_id ~ '^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$'),
  rating      text not null check (rating in ('up', 'down')),
  reasons     text[] not null default '{}'
              check (cardinality(reasons) <= 6
                     and array_position(reasons, null) is null
                     and reasons <@ array[
                       'Too generic', 'Too specific', 'Poor framing', 'Too complex',
                       'Not relevant to my specialty', 'Not an important topic to audit'
                     ]::text[]),
  comment     text not null default ''
              check (char_length(comment) <= 500),
  user_id     uuid null references auth.users (id) on delete set null,
  device_id   text not null
              check (device_id ~ '^[A-Za-z0-9_-]{8,64}$'),
  created_at  timestamptz not null default now(),
  app_version text null check (app_version is null or char_length(app_version) <= 32)
);

create index if not exists feedback_audit_idx on public.feedback (audit_id);
create index if not exists feedback_device_time_idx on public.feedback (device_id, created_at);
create index if not exists feedback_created_idx on public.feedback (created_at);

create table if not exists public.profiles (
  user_id    uuid primary key references auth.users (id) on delete cascade,
  specialty  text null check (specialty is null or char_length(specialty) <= 80),
  grade      text null check (grade is null or char_length(grade) <= 60),
  created_at timestamptz not null default now()
);

create table if not exists public.admins (
  email text primary key check (email = lower(email) and position('@' in email) > 1)
);

-- ---------------------------------------------------------------- helpers

-- True when the signed-in user's (verified, magic-link) email is listed in public.admins.
create or replace function public.is_admin()
returns boolean
language sql
stable
security definer
set search_path = ''
as $$
  select exists (
    select 1 from public.admins a
    where a.email = lower(coalesce(auth.jwt() ->> 'email', ''))
  );
$$;

revoke all on function public.is_admin() from public;
grant execute on function public.is_admin() to anon, authenticated;

-- Server-side guards on every new feedback row:
--   * created_at is always the server time (clients cannot back-date rows to dodge the rate limit);
--   * at most 30 rows per device per rolling 24 hours.
create or replace function public.feedback_before_insert()
returns trigger
language plpgsql
security definer
set search_path = ''
as $$
declare
  recent integer;
begin
  new.created_at := now();
  -- serialise concurrent inserts from the same device so the count is exact
  perform pg_advisory_xact_lock(hashtext('ai4qi_feedback:' || new.device_id));
  select count(*) into recent
    from public.feedback f
   where f.device_id = new.device_id
     and f.created_at > now() - interval '24 hours';
  if recent >= 30 then
    raise exception 'feedback rate limit reached for this device'
      using errcode = 'P0429', hint = 'Try again tomorrow.';
  end if;
  return new;
end;
$$;

revoke all on function public.feedback_before_insert() from public;

drop trigger if exists feedback_before_insert on public.feedback;
create trigger feedback_before_insert
  before insert on public.feedback
  for each row execute function public.feedback_before_insert();

-- Per-audit totals, readable by anyone. Only the latest vote from each person (signed-in user,
-- otherwise device) on each audit is counted, so changing a vote does not double count.
-- Comments, device ids and user ids are never returned.
create or replace function public.feedback_summary(p_audit_id text default null)
returns table (audit_id text, up bigint, down bigint, reasons jsonb)
language sql
stable
security definer
set search_path = ''
as $$
  with latest as (
    select distinct on (f.audit_id, coalesce(f.user_id::text, f.device_id))
           f.audit_id, f.rating, f.reasons
      from public.feedback f
     where p_audit_id is null or f.audit_id = p_audit_id
     order by f.audit_id, coalesce(f.user_id::text, f.device_id), f.created_at desc, f.id desc
  ),
  totals as (
    select l.audit_id,
           count(*) filter (where l.rating = 'up')   as up,
           count(*) filter (where l.rating = 'down') as down
      from latest l
     group by l.audit_id
  ),
  reason_counts as (
    select l.audit_id, jsonb_object_agg(r.reason, r.n) as reasons
      from (select l2.audit_id, x.reason, count(*) as n
              from latest l2, unnest(l2.reasons) as x(reason)
             where l2.rating = 'down'
             group by l2.audit_id, x.reason) r
      join (select distinct audit_id from latest) l on l.audit_id = r.audit_id
     group by l.audit_id
  )
  select t.audit_id, t.up, t.down, coalesce(rc.reasons, '{}'::jsonb)
    from totals t
    left join reason_counts rc on rc.audit_id = t.audit_id
   order by t.audit_id;
$$;

revoke all on function public.feedback_summary(text) from public;
grant execute on function public.feedback_summary(text) to anon, authenticated;

-- ---------------------------------------------------------------- privileges and row level security

alter table public.feedback enable row level security;
alter table public.profiles enable row level security;
alter table public.admins   enable row level security;

revoke all on public.feedback from anon, authenticated;
revoke all on public.profiles from anon, authenticated;
revoke all on public.admins   from anon, authenticated;

grant insert on public.feedback to anon, authenticated;
grant select, delete on public.feedback to authenticated;           -- rows limited to admins by policy
grant select, insert, update, delete on public.profiles to authenticated;
grant select on public.admins to authenticated;                     -- own row only, by policy

-- feedback: anyone may add a row; user_id must be empty or their own id.
drop policy if exists feedback_insert_anyone on public.feedback;
create policy feedback_insert_anyone on public.feedback
  for insert to anon, authenticated
  with check (user_id is null or user_id = auth.uid());

-- feedback: only admins may read or delete raw rows (comments included).
drop policy if exists feedback_admin_select on public.feedback;
create policy feedback_admin_select on public.feedback
  for select to authenticated
  using (public.is_admin());

drop policy if exists feedback_admin_delete on public.feedback;
create policy feedback_admin_delete on public.feedback
  for delete to authenticated
  using (public.is_admin());

-- profiles: each user sees and edits only their own row.
drop policy if exists profiles_own_select on public.profiles;
create policy profiles_own_select on public.profiles
  for select to authenticated using (user_id = auth.uid());

drop policy if exists profiles_own_insert on public.profiles;
create policy profiles_own_insert on public.profiles
  for insert to authenticated with check (user_id = auth.uid());

drop policy if exists profiles_own_update on public.profiles;
create policy profiles_own_update on public.profiles
  for update to authenticated using (user_id = auth.uid()) with check (user_id = auth.uid());

drop policy if exists profiles_own_delete on public.profiles;
create policy profiles_own_delete on public.profiles
  for delete to authenticated using (user_id = auth.uid());

-- admins: a signed-in admin can see their own row (lets the app show the admin page);
-- the list is edited only in the SQL editor / with the service key.
drop policy if exists admins_own_select on public.admins;
create policy admins_own_select on public.admins
  for select to authenticated
  using (email = lower(coalesce(auth.jwt() ->> 'email', '')));
