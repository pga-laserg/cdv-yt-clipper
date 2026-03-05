-- Ensure jobs.metadata exists for API/worker artifact provenance and status details.

alter table public.jobs
  add column if not exists metadata jsonb default '{}'::jsonb;

update public.jobs
set metadata = '{}'::jsonb
where metadata is null;

alter table public.jobs
  alter column metadata set default '{}'::jsonb;
