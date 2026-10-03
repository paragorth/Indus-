-- Ai4Qi: optional email reminders for a user's running audits. Run after 001-003. Safe to re-run.
--
-- Audit data (patient rows) never leaves the user's device. When a signed-in user turns on reminders
-- for a running audit, the app syncs only this metadata: a run id, the audit question (no patient
-- data), a short next step made of counts only (e.g. 'Finish cycle 1 data collection (32 of 40
-- entered)'), a due date and the opt-in flag. The send-reminders function (service role) reads due
-- rows once a day and records last_sent_at and sends; users cannot change those two columns.

-- ---------------------------------------------------------------- table

create table if not exists public.run_reminders (
  user_id      uuid not null references auth.users (id) on delete cascade,
  run_id       text not null check (run_id ~ '^r-[a-z0-9]{6,24}$'),
  audit_title  text not null check (char_length(audit_title) between 1 and 200),
  next_step    text not null check (char_length(next_step) between 1 and 200),
  due_date     date not null,
  email_opt_in boolean not null default true,
  last_sent_at timestamptz null,              -- server only
  sends        integer not null default 0,    -- server only
  updated_at   timestamptz not null default now(),
  primary key (user_id, run_id)
);

-- The daily job looks for opted-in rows by due date.
create index if not exists run_reminders_due_idx on public.run_reminders (due_date) where email_opt_in;

-- ---------------------------------------------------------------- triggers

-- For app users (anon / authenticated roles):
--   * last_sent_at and sends cannot be set on insert or changed on update;
--   * user_id and run_id cannot be changed on update;
--   * at most 50 rows per user.
-- The service role (send-reminders) and the owner in the SQL editor may write every column.
-- updated_at is always the server time.
-- Not security definer, so current_user is the caller's role. The count below runs under RLS,
-- which for a user returns exactly their own rows.
create or replace function public.run_reminders_before()
returns trigger language plpgsql set search_path = '' as $$
declare
  n integer;
  app_user boolean := current_user in ('anon', 'authenticated');
begin
  if tg_op = 'INSERT' then
    if app_user then
      new.last_sent_at := null;
      new.sends := 0;
    end if;
    -- serialise concurrent inserts from the same user so the count is exact
    perform pg_advisory_xact_lock(hashtext('ai4qi_run_reminders:' || new.user_id::text));
    -- An upsert of an existing run (ON CONFLICT DO UPDATE) also fires this insert trigger:
    -- it is not a new row, so it must not be blocked by the cap.
    if not exists (select 1 from public.run_reminders r
                    where r.user_id = new.user_id and r.run_id = new.run_id) then
      select count(*) into n from public.run_reminders r where r.user_id = new.user_id;
      if n >= 50 then
        raise exception 'reminder limit reached (50 audits per user)'
          using errcode = 'PT429', hint = 'Turn off reminders for an audit you have finished.';
      end if;
    end if;
  else
    new.user_id := old.user_id;
    new.run_id := old.run_id;
    if app_user then
      new.last_sent_at := old.last_sent_at;
      new.sends := old.sends;
    end if;
  end if;
  new.updated_at := now();
  return new;
end;
$$;
revoke all on function public.run_reminders_before() from public;
drop trigger if exists run_reminders_before on public.run_reminders;
create trigger run_reminders_before before insert or update on public.run_reminders
  for each row execute function public.run_reminders_before();

-- ---------------------------------------------------------------- privileges and row level security

alter table public.run_reminders enable row level security;

revoke all on public.run_reminders from anon, authenticated;
grant select, insert, update, delete on public.run_reminders to authenticated;

drop policy if exists run_reminders_own_select on public.run_reminders;
create policy run_reminders_own_select on public.run_reminders
  for select to authenticated using (user_id = auth.uid());

drop policy if exists run_reminders_own_insert on public.run_reminders;
create policy run_reminders_own_insert on public.run_reminders
  for insert to authenticated with check (user_id = auth.uid());

drop policy if exists run_reminders_own_update on public.run_reminders;
create policy run_reminders_own_update on public.run_reminders
  for update to authenticated using (user_id = auth.uid()) with check (user_id = auth.uid());

drop policy if exists run_reminders_own_delete on public.run_reminders;
create policy run_reminders_own_delete on public.run_reminders
  for delete to authenticated using (user_id = auth.uid());
