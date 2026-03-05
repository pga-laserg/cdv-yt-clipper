-- Phase 1: enable RLS for tenant isolation

create or replace function public.is_org_member(p_org_id uuid)
returns boolean
language sql
stable
security definer
set search_path = public, auth
as $$
  select exists (
    select 1
    from public.organization_memberships om
    where om.organization_id = p_org_id
      and om.user_id = auth.uid()
  );
$$;

create or replace function public.org_role(p_org_id uuid)
returns text
language sql
stable
security definer
set search_path = public, auth
as $$
  select om.role
  from public.organization_memberships om
  where om.organization_id = p_org_id
    and om.user_id = auth.uid()
  limit 1;
$$;

grant execute on function public.is_org_member(uuid) to authenticated;
grant execute on function public.org_role(uuid) to authenticated;

alter table public.organizations enable row level security;
alter table public.organization_memberships enable row level security;
alter table public.jobs enable row level security;
alter table public.clips enable row level security;
alter table public.posts enable row level security;
alter table public.client_blog_profiles enable row level security;
alter table public.client_publish_destinations enable row level security;
alter table public.blog_generation_events enable row level security;
alter table public.blog_publish_events enable row level security;
alter table public.llm_cue_events enable row level security;

drop policy if exists organizations_select_member on public.organizations;
create policy organizations_select_member
  on public.organizations
  for select
  to authenticated
  using (public.is_org_member(id));

drop policy if exists organizations_update_admin on public.organizations;
create policy organizations_update_admin
  on public.organizations
  for update
  to authenticated
  using (public.org_role(id) in ('owner', 'admin'))
  with check (public.org_role(id) in ('owner', 'admin'));

drop policy if exists organization_memberships_select_member on public.organization_memberships;
create policy organization_memberships_select_member
  on public.organization_memberships
  for select
  to authenticated
  using (public.is_org_member(organization_id));

drop policy if exists organization_memberships_admin_write on public.organization_memberships;
create policy organization_memberships_admin_write
  on public.organization_memberships
  for all
  to authenticated
  using (public.org_role(organization_id) in ('owner', 'admin'))
  with check (public.org_role(organization_id) in ('owner', 'admin'));

drop policy if exists jobs_select_member on public.jobs;
create policy jobs_select_member
  on public.jobs
  for select
  to authenticated
  using (public.is_org_member(organization_id));

drop policy if exists jobs_insert_member on public.jobs;
create policy jobs_insert_member
  on public.jobs
  for insert
  to authenticated
  with check (public.org_role(organization_id) in ('owner', 'admin', 'member'));

drop policy if exists jobs_update_admin on public.jobs;
create policy jobs_update_admin
  on public.jobs
  for update
  to authenticated
  using (public.org_role(organization_id) in ('owner', 'admin'))
  with check (public.org_role(organization_id) in ('owner', 'admin'));

drop policy if exists jobs_delete_admin on public.jobs;
create policy jobs_delete_admin
  on public.jobs
  for delete
  to authenticated
  using (public.org_role(organization_id) in ('owner', 'admin'));

-- Shared tenant policy template

drop policy if exists clips_select_member on public.clips;
create policy clips_select_member
  on public.clips
  for select
  to authenticated
  using (public.is_org_member(organization_id));

drop policy if exists clips_write_admin on public.clips;
create policy clips_write_admin
  on public.clips
  for all
  to authenticated
  using (public.org_role(organization_id) in ('owner', 'admin'))
  with check (public.org_role(organization_id) in ('owner', 'admin'));

drop policy if exists posts_select_member on public.posts;
create policy posts_select_member
  on public.posts
  for select
  to authenticated
  using (public.is_org_member(organization_id));

drop policy if exists posts_write_admin on public.posts;
create policy posts_write_admin
  on public.posts
  for all
  to authenticated
  using (public.org_role(organization_id) in ('owner', 'admin'))
  with check (public.org_role(organization_id) in ('owner', 'admin'));

drop policy if exists client_blog_profiles_select_member on public.client_blog_profiles;
create policy client_blog_profiles_select_member
  on public.client_blog_profiles
  for select
  to authenticated
  using (public.is_org_member(organization_id));

drop policy if exists client_blog_profiles_write_admin on public.client_blog_profiles;
create policy client_blog_profiles_write_admin
  on public.client_blog_profiles
  for all
  to authenticated
  using (public.org_role(organization_id) in ('owner', 'admin'))
  with check (public.org_role(organization_id) in ('owner', 'admin'));

drop policy if exists client_publish_destinations_select_member on public.client_publish_destinations;
create policy client_publish_destinations_select_member
  on public.client_publish_destinations
  for select
  to authenticated
  using (public.is_org_member(organization_id));

drop policy if exists client_publish_destinations_write_admin on public.client_publish_destinations;
create policy client_publish_destinations_write_admin
  on public.client_publish_destinations
  for all
  to authenticated
  using (public.org_role(organization_id) in ('owner', 'admin'))
  with check (public.org_role(organization_id) in ('owner', 'admin'));

drop policy if exists blog_generation_events_select_member on public.blog_generation_events;
create policy blog_generation_events_select_member
  on public.blog_generation_events
  for select
  to authenticated
  using (public.is_org_member(organization_id));

drop policy if exists blog_generation_events_write_admin on public.blog_generation_events;
create policy blog_generation_events_write_admin
  on public.blog_generation_events
  for all
  to authenticated
  using (public.org_role(organization_id) in ('owner', 'admin'))
  with check (public.org_role(organization_id) in ('owner', 'admin'));

drop policy if exists blog_publish_events_select_member on public.blog_publish_events;
create policy blog_publish_events_select_member
  on public.blog_publish_events
  for select
  to authenticated
  using (public.is_org_member(organization_id));

drop policy if exists blog_publish_events_write_admin on public.blog_publish_events;
create policy blog_publish_events_write_admin
  on public.blog_publish_events
  for all
  to authenticated
  using (public.org_role(organization_id) in ('owner', 'admin'))
  with check (public.org_role(organization_id) in ('owner', 'admin'));

drop policy if exists llm_cue_events_select_member on public.llm_cue_events;
create policy llm_cue_events_select_member
  on public.llm_cue_events
  for select
  to authenticated
  using (organization_id is not null and public.is_org_member(organization_id));

drop policy if exists llm_cue_events_write_admin on public.llm_cue_events;
create policy llm_cue_events_write_admin
  on public.llm_cue_events
  for all
  to authenticated
  using (organization_id is not null and public.org_role(organization_id) in ('owner', 'admin'))
  with check (organization_id is not null and public.org_role(organization_id) in ('owner', 'admin'));
