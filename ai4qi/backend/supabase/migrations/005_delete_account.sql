-- Ai4Qi: let a signed-in user delete their own account (UK GDPR right to erasure, self-service).
-- Deleting the auth user cascades to profiles, my_audits, usage_events and run_reminders
-- (all reference auth.users on delete cascade). Feedback and built_audits rows keep no link to
-- the person: their user_id is set to null (feedback: on delete set null in 001; built_audits in 003).

create or replace function public.delete_my_account()
returns void
language plpgsql
security definer
set search_path = public, auth
as $$
declare
  uid uuid := auth.uid();
begin
  if uid is null then
    raise exception 'not signed in' using errcode = '28000';
  end if;
  delete from auth.users where id = uid;
end;
$$;

revoke all on function public.delete_my_account() from public, anon;
grant execute on function public.delete_my_account() to authenticated;
