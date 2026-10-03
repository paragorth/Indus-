-- 010: Ai4Qi is for NHS, HSE and university staff.
-- Work and university emails get in straight away. Anyone else can ask for access: a request is
-- approved automatically after 5 minutes unless an admin rejects it first (or approves it sooner).
-- Admins can also allow a whole domain (for example a hospital that does not use nhs.net).
-- Safe to run again.

create table if not exists public.allowed_domains (
  domain     text primary key check (domain ~ '^[a-z0-9.-]+\.[a-z]{2,}$'),
  note       text,
  added_at   timestamptz not null default now()
);

create table if not exists public.access_requests (
  user_id    uuid primary key references auth.users (id) on delete cascade,
  email      text not null default '',
  role       text check (role is null or char_length(role) <= 80),
  workplace  text check (workplace is null or char_length(workplace) <= 160),
  reason     text check (reason is null or char_length(reason) <= 500),
  created_at timestamptz not null default now(),
  decision   text check (decision in ('approved', 'rejected')),
  decided_at timestamptz
);

-- The email on a request always comes from the account, never from what the browser sends.
create or replace function public.access_request_fill()
returns trigger language plpgsql security definer set search_path = '' as $$
begin
  new.email := coalesce((select u.email from auth.users u where u.id = new.user_id), '');
  if tg_op = 'INSERT' then
    new.created_at := now(); new.decision := null; new.decided_at := null;
  end if;
  return new;
end $$;
drop trigger if exists access_request_fill on public.access_requests;
create trigger access_request_fill before insert on public.access_requests
  for each row execute function public.access_request_fill();

alter table public.allowed_domains enable row level security;
alter table public.access_requests enable row level security;

drop policy if exists "own request: read" on public.access_requests;
create policy "own request: read" on public.access_requests for select to authenticated
  using (user_id = auth.uid() or public.is_admin());
drop policy if exists "own request: create" on public.access_requests;
create policy "own request: create" on public.access_requests for insert to authenticated
  with check (user_id = auth.uid());
drop policy if exists "admins decide" on public.access_requests;
create policy "admins decide" on public.access_requests for update to authenticated
  using (public.is_admin()) with check (public.is_admin());
drop policy if exists "admins: domains" on public.allowed_domains;
create policy "admins: domains" on public.allowed_domains for all to authenticated
  using (public.is_admin()) with check (public.is_admin());

-- NHS (all four nations), HSE, UK universities and the main Irish universities and colleges,
-- plus any domain an admin has allowed.
create or replace function public.is_institutional_email(p_email text)
returns boolean language sql stable security definer set search_path = '' as $$
  select coalesce(
    public.is_verified_email(p_email)
    or lower(p_email) ~ '@([a-z0-9-]+\.)*ac\.uk$'
    or lower(p_email) ~ '@([a-z0-9-]+\.)*(tcd|ucd|ucc|ul|dcu|mu|rcsi|universityofgalway|nuigalway|atu|tus|setu|mtu|tudublin)\.ie$'
    or lower(p_email) ~ '@([a-z0-9-]+\.)*rcsi\.com$'
    or exists (select 1 from public.allowed_domains d
               where lower(p_email) like '%@' || d.domain or lower(p_email) like '%.' || d.domain),
    false)
$$;

-- 'ok', 'pending', 'rejected' or 'none' for a given user.
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

-- The signed-in user's own status (for the page).
create or replace function public.my_access()
returns text language sql stable security definer set search_path = '' as $$
  select case when auth.uid() is null then 'signed_out' else public.access_for(auth.uid()) end
$$;

revoke all on function public.access_for(uuid) from public, anon, authenticated;
grant execute on function public.access_for(uuid) to service_role;
grant execute on function public.my_access() to authenticated;
grant execute on function public.is_institutional_email(text) to authenticated;
