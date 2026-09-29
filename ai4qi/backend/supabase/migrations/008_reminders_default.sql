-- Ai4Qi: email reminders are on by default for signed-in users, with one switch for all audits.
-- profiles.reminders_off = true stops them; the app then deletes the user's run_reminders rows.
alter table public.profiles add column if not exists reminders_off boolean not null default false;
