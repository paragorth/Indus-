-- 014: sign-up totals only (no names, emails or ids), so the Ai4Qi team's tools can report growth
-- without an admin login. Safe to run again.
create or replace function public.signup_totals()
returns table (total bigint, last_24h bigint, last_7d bigint, institutional bigint, personal bigint)
language sql stable security definer set search_path = '' as $$
  select count(*),
         count(*) filter (where u.created_at > now() - interval '24 hours'),
         count(*) filter (where u.created_at > now() - interval '7 days'),
         count(*) filter (where public.is_institutional_email(u.email)),
         count(*) filter (where not public.is_institutional_email(u.email))
  from auth.users u
$$;
revoke all on function public.signup_totals() from public;
grant execute on function public.signup_totals() to anon, authenticated;
