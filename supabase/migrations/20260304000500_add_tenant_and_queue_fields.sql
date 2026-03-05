-- Phase 1: add tenant fields and queue lease columns

do $$
declare
  v_default_org uuid;
begin
  insert into public.organizations (name, slug)
  values ('Default Organization', 'default-org')
  on conflict (slug) do nothing;

  select id into v_default_org
  from public.organizations
  where slug = 'default-org'
  limit 1;

  if v_default_org is null then
    raise exception 'default organization missing';
  end if;

  alter table public.jobs add column if not exists organization_id uuid;
  alter table public.clips add column if not exists organization_id uuid;
  alter table public.posts add column if not exists organization_id uuid;
  alter table public.client_blog_profiles add column if not exists organization_id uuid;
  alter table public.client_publish_destinations add column if not exists organization_id uuid;
  alter table public.blog_generation_events add column if not exists organization_id uuid;
  alter table public.blog_publish_events add column if not exists organization_id uuid;
  alter table public.llm_cue_events add column if not exists organization_id uuid;

  update public.jobs set organization_id = v_default_org where organization_id is null;
  update public.clips set organization_id = v_default_org where organization_id is null;
  update public.posts set organization_id = v_default_org where organization_id is null;
  update public.client_blog_profiles set organization_id = v_default_org where organization_id is null;
  update public.client_publish_destinations set organization_id = v_default_org where organization_id is null;
  update public.blog_generation_events set organization_id = v_default_org where organization_id is null;
  update public.blog_publish_events set organization_id = v_default_org where organization_id is null;
  update public.llm_cue_events set organization_id = v_default_org where organization_id is null;

  alter table public.jobs alter column organization_id set not null;
  alter table public.clips alter column organization_id set not null;
  alter table public.posts alter column organization_id set not null;
  alter table public.client_blog_profiles alter column organization_id set not null;
  alter table public.client_publish_destinations alter column organization_id set not null;
  alter table public.blog_generation_events alter column organization_id set not null;
  alter table public.blog_publish_events alter column organization_id set not null;

  if exists (
    select 1
    from pg_constraint
    where conname = 'client_blog_profiles_pkey'
      and conrelid = 'public.client_blog_profiles'::regclass
  ) then
    alter table public.client_blog_profiles drop constraint client_blog_profiles_pkey;
  end if;

  alter table public.client_blog_profiles
    add constraint client_blog_profiles_pkey
    primary key (organization_id, client_id);

  alter table public.jobs add column if not exists claim_token text;
  alter table public.jobs add column if not exists claimed_by text;
  alter table public.jobs add column if not exists claimed_at timestamp with time zone;
  alter table public.jobs add column if not exists lease_expires_at timestamp with time zone;
  alter table public.jobs add column if not exists attempt_count integer not null default 0;
  alter table public.jobs add column if not exists last_error text;

  if not exists (
    select 1 from pg_constraint where conname = 'jobs_organization_id_fkey'
  ) then
    alter table public.jobs
      add constraint jobs_organization_id_fkey
      foreign key (organization_id) references public.organizations(id) on delete restrict;
  end if;

  if not exists (
    select 1 from pg_constraint where conname = 'clips_organization_id_fkey'
  ) then
    alter table public.clips
      add constraint clips_organization_id_fkey
      foreign key (organization_id) references public.organizations(id) on delete restrict;
  end if;

  if not exists (
    select 1 from pg_constraint where conname = 'posts_organization_id_fkey'
  ) then
    alter table public.posts
      add constraint posts_organization_id_fkey
      foreign key (organization_id) references public.organizations(id) on delete restrict;
  end if;

  if not exists (
    select 1 from pg_constraint where conname = 'client_blog_profiles_organization_id_fkey'
  ) then
    alter table public.client_blog_profiles
      add constraint client_blog_profiles_organization_id_fkey
      foreign key (organization_id) references public.organizations(id) on delete cascade;
  end if;

  if not exists (
    select 1 from pg_constraint where conname = 'client_publish_destinations_organization_id_fkey'
  ) then
    alter table public.client_publish_destinations
      add constraint client_publish_destinations_organization_id_fkey
      foreign key (organization_id) references public.organizations(id) on delete cascade;
  end if;

  if not exists (
    select 1 from pg_constraint where conname = 'blog_generation_events_organization_id_fkey'
  ) then
    alter table public.blog_generation_events
      add constraint blog_generation_events_organization_id_fkey
      foreign key (organization_id) references public.organizations(id) on delete cascade;
  end if;

  if not exists (
    select 1 from pg_constraint where conname = 'blog_publish_events_organization_id_fkey'
  ) then
    alter table public.blog_publish_events
      add constraint blog_publish_events_organization_id_fkey
      foreign key (organization_id) references public.organizations(id) on delete cascade;
  end if;

  if not exists (
    select 1 from pg_constraint where conname = 'llm_cue_events_organization_id_fkey'
  ) then
    alter table public.llm_cue_events
      add constraint llm_cue_events_organization_id_fkey
      foreign key (organization_id) references public.organizations(id) on delete set null;
  end if;
end $$;

create index if not exists jobs_org_created_idx on public.jobs(organization_id, created_at desc);
create index if not exists clips_org_created_idx on public.clips(organization_id, created_at desc);
create index if not exists posts_org_created_idx on public.posts(organization_id, created_at desc);
create index if not exists client_blog_profiles_org_idx on public.client_blog_profiles(organization_id, client_id);
create index if not exists client_publish_destinations_org_idx on public.client_publish_destinations(organization_id, created_at desc);
create index if not exists blog_generation_events_org_idx on public.blog_generation_events(organization_id, created_at desc);
create index if not exists blog_publish_events_org_idx on public.blog_publish_events(organization_id, created_at desc);
create index if not exists llm_cue_events_org_idx on public.llm_cue_events(organization_id, created_at desc);

create index if not exists jobs_claim_queue_idx
  on public.jobs(status, lease_expires_at, created_at);

create unique index if not exists client_blog_profiles_org_client_uidx
  on public.client_blog_profiles(organization_id, client_id);
