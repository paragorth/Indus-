-- Ai4Qi stage 1b: in-app usage counting (registration details, "My audits" tracker, usage events,
-- admin-only weekly statistics). Run after 001_feedback_auth.sql. Safe to re-run.
--
-- No patient data and no free text: profiles hold only values from fixed lists, the tracker holds
-- only an audit id, a status and timestamps, and usage events hold only a fixed event name, an
-- optional audit id and a timestamp. Admins see aggregates through functions, never raw rows, and
-- breakdown cells of fewer than five people are suppressed.

-- ---------------------------------------------------------------- fixed lists

create or replace function public.ai4qi_grades() returns text[] language sql immutable as $$
  select array[
    'Medical student', 'Foundation doctor (FY1–FY2)', 'Core or specialty trainee (CT/ST1–2)',
    'Specialty registrar (ST3+)', 'Specialty doctor or specialist (SAS)', 'Locally employed doctor',
    'Consultant', 'General practitioner', 'Nurse or midwife', 'Allied health professional', 'Pharmacist',
    'Quality improvement or clinical governance lead', 'Other'
  ]::text[] $$;

create or replace function public.ai4qi_specialties() returns text[] language sql immutable as $$
  select array[
    'Acute and general medicine', 'Anaesthesia', 'Cardiology', 'Cardiothoracic surgery',
    'Care of the elderly', 'Clinical governance and quality improvement', 'Dentistry and oral surgery',
    'Dermatology', 'Emergency medicine', 'Endocrinology and diabetes', 'ENT', 'Gastroenterology',
    'General practice', 'General surgery', 'Haematology', 'Infectious diseases and microbiology',
    'Intensive care', 'Medical education', 'Neonatology', 'Neurology', 'Neurosurgery',
    'Nursing and midwifery', 'Obstetrics and gynaecology', 'Oncology', 'Ophthalmology', 'Orthopaedics',
    'Paediatric surgery', 'Paediatrics', 'Palliative care', 'Pathology', 'Pharmacy', 'Plastic surgery',
    'Psychiatry', 'Radiology', 'Renal medicine', 'Respiratory', 'Rheumatology', 'Sexual health', 'Stroke',
    'Therapies and allied health', 'Urology', 'Vascular surgery', 'Other'
  ]::text[] $$;

create or replace function public.ai4qi_regions() returns text[] language sql immutable as $$
  select array[
    'North East and Yorkshire', 'North West', 'Midlands', 'East of England', 'London', 'South East',
    'South West', 'Scotland', 'Wales', 'Northern Ireland', 'Ireland', 'Outside the UK and Ireland'
  ]::text[] $$;

grant execute on function public.ai4qi_grades(), public.ai4qi_specialties(), public.ai4qi_regions()
  to anon, authenticated;

-- ---------------------------------------------------------------- profiles: fixed values only

alter table public.profiles add column if not exists region text null;
alter table public.profiles add column if not exists updated_at timestamptz not null default now();

-- NOT VALID: rows saved before this migration are left alone; every new or edited row is checked.
alter table public.profiles drop constraint if exists profiles_grade_list;
alter table public.profiles add constraint profiles_grade_list
  check (grade is null or grade = any (public.ai4qi_grades())) not valid;
alter table public.profiles drop constraint if exists profiles_specialty_list;
alter table public.profiles add constraint profiles_specialty_list
  check (specialty is null or specialty = any (public.ai4qi_specialties())) not valid;
alter table public.profiles drop constraint if exists profiles_region_list;
alter table public.profiles add constraint profiles_region_list
  check (region is null or region = any (public.ai4qi_regions())) not valid;

-- ---------------------------------------------------------------- tables

create table if not exists public.my_audits (
  user_id      uuid not null references auth.users (id) on delete cascade,
  audit_id     text not null check (audit_id ~ '^(ONA|NNA)-[0-9]{1,5}$'),
  status       text not null default 'started'
               check (status in ('started', 'cycle1', 'change', 'reaudit', 'closed')),
  started_at   timestamptz not null default now(),
  updated_at   timestamptz not null default now(),
  completed_at timestamptz null,
  primary key (user_id, audit_id)
);

create table if not exists public.usage_events (
  id         bigint generated always as identity primary key,
  user_id    uuid not null references auth.users (id) on delete cascade,
  event      text not null check (event in ('signup', 'audit_started', 'audit_status_changed',
                                            'audit_completed', 'active_day')),
  audit_id   text null check (audit_id is null or audit_id ~ '^(ONA|NNA)-[0-9]{1,5}$'),
  created_at timestamptz not null default now(),
  day        date not null default (now() at time zone 'utc')::date
);

create index if not exists usage_events_time_idx on public.usage_events (created_at);
create index if not exists usage_events_user_idx on public.usage_events (user_id, created_at);
create unique index if not exists usage_events_one_signup on public.usage_events (user_id)
  where event = 'signup';
create unique index if not exists usage_events_one_active_day on public.usage_events (user_id, day)
  where event = 'active_day';

-- ---------------------------------------------------------------- triggers

-- Server time only, on every event row.
create or replace function public.usage_events_stamp()
returns trigger language plpgsql security definer set search_path = '' as $$
begin
  new.created_at := now();
  new.day := (new.created_at at time zone 'utc')::date;
  return new;
end;
$$;
revoke all on function public.usage_events_stamp() from public;
drop trigger if exists usage_events_stamp on public.usage_events;
create trigger usage_events_stamp before insert on public.usage_events
  for each row execute function public.usage_events_stamp();

create or replace function public.profiles_touch()
returns trigger language plpgsql set search_path = '' as $$
begin
  new.updated_at := now();
  return new;
end;
$$;
drop trigger if exists profiles_touch on public.profiles;
create trigger profiles_touch before update on public.profiles
  for each row execute function public.profiles_touch();

-- Tracker timestamps are set by the server; started / status changed / completed events follow.
create or replace function public.my_audits_before()
returns trigger language plpgsql set search_path = '' as $$
begin
  if tg_op = 'INSERT' then
    new.started_at := now();
  else
    new.started_at := old.started_at;
    new.user_id := old.user_id;
    new.audit_id := old.audit_id;
  end if;
  new.updated_at := now();
  if new.status = 'closed' then
    new.completed_at := case when tg_op = 'UPDATE' and old.status = 'closed' then old.completed_at else now() end;
  else
    new.completed_at := null;
  end if;
  return new;
end;
$$;
drop trigger if exists my_audits_before on public.my_audits;
create trigger my_audits_before before insert or update on public.my_audits
  for each row execute function public.my_audits_before();

create or replace function public.my_audits_after()
returns trigger language plpgsql security definer set search_path = '' as $$
begin
  if tg_op = 'INSERT' then
    insert into public.usage_events (user_id, event, audit_id) values (new.user_id, 'audit_started', new.audit_id);
    if new.status <> 'started' then
      insert into public.usage_events (user_id, event, audit_id) values (new.user_id, 'audit_status_changed', new.audit_id);
    end if;
  elsif new.status is distinct from old.status then
    insert into public.usage_events (user_id, event, audit_id) values (new.user_id, 'audit_status_changed', new.audit_id);
  end if;
  if new.status = 'closed' and (tg_op = 'INSERT' or old.status <> 'closed') then
    insert into public.usage_events (user_id, event, audit_id) values (new.user_id, 'audit_completed', new.audit_id);
  end if;
  return null;
end;
$$;
revoke all on function public.my_audits_after() from public;
drop trigger if exists my_audits_after on public.my_audits;
create trigger my_audits_after after insert or update on public.my_audits
  for each row execute function public.my_audits_after();

-- Called by the app once per visit while signed in: records the first sign-in (signup) once and
-- one active_day per user per day. Repeated calls are harmless.
create or replace function public.record_activity()
returns void language plpgsql security definer set search_path = '' as $$
declare
  uid uuid := auth.uid();
begin
  if uid is null then return; end if;
  insert into public.usage_events (user_id, event) values (uid, 'signup') on conflict do nothing;
  insert into public.usage_events (user_id, event) values (uid, 'active_day') on conflict do nothing;
end;
$$;
revoke all on function public.record_activity() from public;
grant execute on function public.record_activity() to authenticated;

-- ---------------------------------------------------------------- admin-only statistics

-- Per ISO week (Monday start, UTC): sign-ups, cumulative users, audits started and completed,
-- active users, and returning users (active this week and also active in an earlier week).
create or replace function public.weekly_stats(p_from date default null, p_to date default null)
returns table (week_start date, iso_week text, signups bigint, cumulative_users bigint,
               audits_started bigint, audits_completed bigint, active_users bigint, returning_users bigint)
language plpgsql stable security definer set search_path = '' as $$
begin
  if not public.is_admin() then
    raise exception 'not authorised' using errcode = '42501';
  end if;
  return query
  with ev as (
    select e.user_id, e.event, e.audit_id, date_trunc('week', e.created_at at time zone 'utc')::date as wk
      from public.usage_events e
  ),
  bounds as (
    select coalesce(date_trunc('week', p_from)::date, min(wk), date_trunc('week', now() at time zone 'utc')::date) as lo,
           coalesce(date_trunc('week', p_to)::date, date_trunc('week', now() at time zone 'utc')::date) as hi
      from ev
  ),
  weeks as (
    select g::date as wk from bounds, generate_series(bounds.lo::timestamp, bounds.hi::timestamp, interval '1 week') g
  ),
  first_week as (   -- each user's first week of any activity
    select user_id, min(wk) as wk from ev group by user_id
  ),
  signup_week as (
    select user_id, min(wk) as wk from ev where event = 'signup' group by user_id
  ),
  active as (
    select distinct user_id, wk from ev
  )
  select w.wk,
         to_char(w.wk, 'IYYY-"W"IW'),
         (select count(*) from signup_week s where s.wk = w.wk),
         (select count(*) from signup_week s where s.wk <= w.wk),
         (select count(distinct (e.user_id, e.audit_id)) from ev e where e.wk = w.wk and e.event = 'audit_started'),
         (select count(distinct (e.user_id, e.audit_id)) from ev e where e.wk = w.wk and e.event = 'audit_completed'),
         (select count(*) from active a where a.wk = w.wk),
         (select count(*) from active a join first_week f on f.user_id = a.user_id
           where a.wk = w.wk and f.wk < w.wk)
    from weeks w
   order by w.wk;
end;
$$;
revoke all on function public.weekly_stats(date, date) from public;
grant execute on function public.weekly_stats(date, date) to authenticated;

-- Sign-ups in the period by grade, specialty and region. Cells with fewer than five people are
-- returned with users = null (shown as "fewer than 5") so that no individual can be picked out.
create or replace function public.signup_breakdown(p_from date default null, p_to date default null)
returns table (dimension text, value text, users bigint)
language plpgsql stable security definer set search_path = '' as $$
begin
  if not public.is_admin() then
    raise exception 'not authorised' using errcode = '42501';
  end if;
  return query
  with s as (
    select e.user_id from public.usage_events e
     where e.event = 'signup'
       and (p_from is null or e.day >= date_trunc('week', p_from)::date)
       and (p_to is null or e.day < date_trunc('week', p_to)::date + 7)
  ),
  p as (
    select coalesce(pr.grade, 'Not stated') as grade, coalesce(pr.specialty, 'Not stated') as specialty,
           coalesce(pr.region, 'Not stated') as region
      from s left join public.profiles pr on pr.user_id = s.user_id
  ),
  cells as (
    select 'grade'::text as d, grade as v, count(*) as n from p group by grade
    union all select 'specialty', specialty, count(*) from p group by specialty
    union all select 'region', region, count(*) from p group by region
  )
  select c.d, c.v, case when c.n < 5 then null else c.n end
    from cells c
   order by c.d, c.n desc, c.v;
end;
$$;
revoke all on function public.signup_breakdown(date, date) from public;
grant execute on function public.signup_breakdown(date, date) to authenticated;

-- ---------------------------------------------------------------- privileges and row level security

alter table public.my_audits    enable row level security;
alter table public.usage_events enable row level security;

revoke all on public.my_audits    from anon, authenticated;
revoke all on public.usage_events from anon, authenticated;

grant select, insert, update, delete on public.my_audits to authenticated;
grant insert on public.usage_events to authenticated;   -- insert only; nobody reads raw rows via the API

drop policy if exists my_audits_own_select on public.my_audits;
create policy my_audits_own_select on public.my_audits
  for select to authenticated using (user_id = auth.uid());
drop policy if exists my_audits_own_insert on public.my_audits;
create policy my_audits_own_insert on public.my_audits
  for insert to authenticated with check (user_id = auth.uid());
drop policy if exists my_audits_own_update on public.my_audits;
create policy my_audits_own_update on public.my_audits
  for update to authenticated using (user_id = auth.uid()) with check (user_id = auth.uid());
drop policy if exists my_audits_own_delete on public.my_audits;
create policy my_audits_own_delete on public.my_audits
  for delete to authenticated using (user_id = auth.uid());

-- Users may add only their own active_day rows directly (at most one a day, by the unique index);
-- sign-up and audit events come from record_activity() and the tracker triggers.
drop policy if exists usage_events_own_insert on public.usage_events;
create policy usage_events_own_insert on public.usage_events
  for insert to authenticated with check (user_id = auth.uid() and event = 'active_day');
