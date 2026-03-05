-- Phase 1: queue claim/heartbeat/complete RPC functions

create or replace function public.claim_next_job(
  p_worker_id text,
  p_lease_seconds integer default 120
)
returns public.jobs
language plpgsql
security definer
set search_path = public
as $$
declare
  v_job public.jobs;
  v_claim_token text := gen_random_uuid()::text;
  v_lease_seconds integer := greatest(30, coalesce(p_lease_seconds, 120));
begin
  update public.jobs j
  set status = 'processing',
      claim_token = v_claim_token,
      claimed_by = nullif(trim(coalesce(p_worker_id, '')), ''),
      claimed_at = now(),
      lease_expires_at = now() + make_interval(secs => v_lease_seconds),
      attempt_count = coalesce(j.attempt_count, 0) + 1
  where j.id = (
    select j2.id
    from public.jobs j2
    where j2.status = 'pending'
       or (
         (j2.status = 'processing' or j2.status like 'processing:%')
         and j2.lease_expires_at is not null
         and j2.lease_expires_at <= now()
       )
    order by j2.created_at asc
    for update skip locked
    limit 1
  )
  returning j.* into v_job;

  return v_job;
end;
$$;

create or replace function public.heartbeat_job_claim(
  p_job_id uuid,
  p_claim_token text,
  p_extend_seconds integer default 120
)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
declare
  v_lease_seconds integer := greatest(30, coalesce(p_extend_seconds, 120));
  v_updated integer := 0;
begin
  update public.jobs
  set lease_expires_at = now() + make_interval(secs => v_lease_seconds)
  where id = p_job_id
    and claim_token = p_claim_token
    and (status = 'processing' or status like 'processing:%');

  get diagnostics v_updated = row_count;
  return v_updated > 0;
end;
$$;

create or replace function public.complete_job_claim(
  p_job_id uuid,
  p_claim_token text,
  p_final_status text,
  p_error text default null
)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
declare
  v_updated integer := 0;
begin
  update public.jobs
  set status = coalesce(nullif(trim(p_final_status), ''), 'failed'),
      last_error = p_error,
      claim_token = null,
      claimed_by = null,
      claimed_at = null,
      lease_expires_at = null
  where id = p_job_id
    and claim_token = p_claim_token;

  get diagnostics v_updated = row_count;
  return v_updated > 0;
end;
$$;

grant execute on function public.claim_next_job(text, integer) to service_role;
grant execute on function public.heartbeat_job_claim(uuid, text, integer) to service_role;
grant execute on function public.complete_job_claim(uuid, text, text, text) to service_role;
