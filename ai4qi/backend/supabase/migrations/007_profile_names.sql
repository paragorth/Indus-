-- Ai4Qi: optional name, hospital and department on the profile, used only to fill in the
-- "Send to your supervisor" email and proposal for the signed-in user. Never shown in totals.
alter table public.profiles add column if not exists full_name    text null;
alter table public.profiles add column if not exists organisation text null;
alter table public.profiles add column if not exists department   text null;

alter table public.profiles drop constraint if exists profiles_names_length;
alter table public.profiles add constraint profiles_names_length check (
  coalesce(char_length(full_name), 0) <= 120 and coalesce(char_length(organisation), 0) <= 160 and coalesce(char_length(department), 0) <= 120);
