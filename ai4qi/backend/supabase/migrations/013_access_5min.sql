-- 013: personal (non-institutional) emails are approved automatically after 5 minutes, not 15,
-- unless an admin rejects them first. The page now files the request itself at sign-in. Safe to run again.

create or replace function public.access_for(p_user uuid)
returns text language plpgsql stable security definer set search_path = '' as $$
declare v_email text; r public.access_requests;
begin
  select u.email into v_email from auth.users u where u.id = p_user;
  if v_email is null then return 'none'; end if;
  if public.is_institutional_email(v_email) or exists (select 1 from public.admins a where lower(a.email) = lower(v_email)) then
    return 'ok';
  end if;
  select * into r from public.access_requests q where q.user_id = p_user;
  if not found then return 'none'; end if;
  if r.decision = 'rejected' then return 'rejected'; end if;
  if r.decision = 'approved' or r.created_at < now() - interval '5 minutes' then return 'ok'; end if;
  return 'pending';
end $$;
