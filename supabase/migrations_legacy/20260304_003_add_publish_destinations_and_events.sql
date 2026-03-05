-- Add multi-destination publishing support and per-client author default hardening.
-- Safe to run multiple times.

alter table if exists public.client_blog_profiles
  alter column default_author_name set default 'Default Author';

create table if not exists public.client_publish_destinations (
  id uuid default gen_random_uuid() primary key,
  client_id text not null,
  name text,
  destination_type text not null,
  enabled boolean not null default true,
  publish_mode text not null default 'draft',
  config jsonb not null default '{}'::jsonb,
  field_mapping jsonb not null default '{}'::jsonb,
  created_at timestamp with time zone not null default now(),
  updated_at timestamp with time zone not null default now()
);

create index if not exists client_publish_destinations_client_idx
  on public.client_publish_destinations(client_id);
create index if not exists client_publish_destinations_enabled_idx
  on public.client_publish_destinations(enabled);

create table if not exists public.blog_publish_events (
  id uuid default gen_random_uuid() primary key,
  created_at timestamp with time zone not null default now(),
  destination_id uuid,
  destination_type text not null,
  job_id text not null,
  client_id text not null,
  post_id text not null,
  status text not null,
  remote_id text,
  remote_url text,
  attempt integer not null default 1,
  duration_ms integer,
  error text,
  metadata jsonb not null default '{}'::jsonb
);

create index if not exists blog_publish_events_job_idx
  on public.blog_publish_events(job_id);
create index if not exists blog_publish_events_client_idx
  on public.blog_publish_events(client_id);
create index if not exists blog_publish_events_status_idx
  on public.blog_publish_events(status);
create index if not exists blog_publish_events_created_idx
  on public.blog_publish_events(created_at desc);
