-- Ai4Qi: email consent (PECR), extra optional profile fields, and sponsor-ready sign-up figures.
-- Consent boxes are unticked by default; the time of the last change is kept as evidence.
-- "Verified" = signed up with an NHS or HSE email address (counted from auth.users, never exported).

alter table public.profiles add column if not exists work_setting       text null;
alter table public.profiles add column if not exists audit_purpose      text null;
alter table public.profiles add column if not exists consent_news       boolean not null default false;
alter table public.profiles add column if not exists consent_sponsors   boolean not null default false;
alter table public.profiles add column if not exists consent_updated_at timestamptz null;

alter table public.profiles drop constraint if exists profiles_work_setting_list;
alter table public.profiles add constraint profiles_work_setting_list check (work_setting is null or work_setting in (
  'NHS or HSE hospital', 'General practice', 'Community or mental health service', 'Private or independent sector',
  'University or medical school', 'Working outside the UK and Ireland', 'Not currently working in healthcare'));
alter table public.profiles drop constraint if exists profiles_audit_purpose_list;
alter table public.profiles add constraint profiles_audit_purpose_list check (audit_purpose is null or audit_purpose in (
  'Portfolio or ARCP', 'Job or training application', 'Portfolio pathway (CESR) or specialist registration',
  'Departmental or service improvement', 'Research or a qualification', 'Other'));

create or replace function public.is_verified_email(p_email text)
returns boolean language sql immutable set search_path = '' as $$
  select coalesce(lower(p_email) ~ '@(nhs\.net|([a-z0-9-]+\.)*nhs\.uk|([a-z0-9-]+\.)*nhs\.scot|([a-z0-9-]+\.)*nhs\.wales|([a-z0-9-]+\.)*hscni\.net|([a-z0-9-]+\.)*hse\.ie)$', false)
$$;

-- Same contract as in 002 (dimension, value, users; groups under 5 suppressed), with four more dimensions.
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
           coalesce(pr.region, 'Not stated') as region,
           coalesce(pr.work_setting, 'Not stated') as work_setting,
           coalesce(pr.audit_purpose, 'Not stated') as audit_purpose,
           case when public.is_verified_email(u.email) then 'NHS or HSE email' else 'Other email' end as verified,
           coalesce(pr.consent_news, false) as c_news, coalesce(pr.consent_sponsors, false) as c_sponsors
      from s
      left join public.profiles pr on pr.user_id = s.user_id
      left join auth.users u on u.id = s.user_id
  ),
  cells as (
    select 'grade'::text as d, grade as v, count(*) as n from p group by grade
    union all select 'specialty', specialty, count(*) from p group by specialty
    union all select 'region', region, count(*) from p group by region
    union all select 'work_setting', work_setting, count(*) from p group by work_setting
    union all select 'audit_purpose', audit_purpose, count(*) from p group by audit_purpose
    union all select 'verified', verified, count(*) from p group by verified
    union all select 'consent', 'Ai4Qi news', count(*) filter (where c_news) from p
    union all select 'consent', 'Sponsor offers', count(*) filter (where c_sponsors) from p
  )
  select c.d, c.v, case when c.n < 5 then null else c.n end
    from cells c
   order by c.d, c.n desc, c.v;
end;
$$;
revoke all on function public.signup_breakdown(date, date) from public, anon;
grant execute on function public.signup_breakdown(date, date) to authenticated;

-- Admin only: the mailing list for one kind of consent ('news' or 'sponsors'). The owner sends
-- sponsor messages themselves; email addresses are never given to sponsors.
create or replace function public.mailing_list(p_kind text)
returns table (email text, consent_updated_at timestamptz)
language plpgsql stable security definer set search_path = '' as $$
begin
  if not public.is_admin() then
    raise exception 'not authorised' using errcode = '42501';
  end if;
  if p_kind not in ('news', 'sponsors') then
    raise exception 'kind must be news or sponsors' using errcode = '22023';
  end if;
  return query
    select u.email::text, pr.consent_updated_at
      from public.profiles pr join auth.users u on u.id = pr.user_id
     where (p_kind = 'news' and pr.consent_news) or (p_kind = 'sponsors' and pr.consent_sponsors)
     order by pr.consent_updated_at nulls last;
end;
$$;
revoke all on function public.mailing_list(text) from public, anon;
grant execute on function public.mailing_list(text) to authenticated;
