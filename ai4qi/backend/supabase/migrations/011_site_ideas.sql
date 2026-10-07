drop function if exists public.site_ideas(int);
-- 011: let the Ai4Qi team's tools read the site ideas ("Suggest a change") without an admin login.
-- Returns only the post's id, what people wrote, which part of the site, and the date: never the
-- account, device or anything else. (Replies are emailed by send-reminders, which looks the person up.) The form tells people their idea is read by the team and not to add names or
-- patient details. Safe to run again.

create or replace function public.site_ideas(p_days int default 90)
returns table (id uuid, created date, part text, idea text)
language sql stable security definer set search_path = '' as $$
  select f.id, f.created_at::date, split_part(f.audit_id, '-', 2), f.comment
  from public.feedback f
  where f.audit_id like 'SITE-%' and f.comment <> ''
    and f.created_at > now() - make_interval(days => greatest(1, least(coalesce(p_days, 90), 365)))
  order by f.created_at desc
  limit 1000
$$;

revoke all on function public.site_ideas(int) from public;
grant execute on function public.site_ideas(int) to anon, authenticated;
