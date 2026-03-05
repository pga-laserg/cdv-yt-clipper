# SaaS Backlog

## Objectives and Success Metrics
- Enforce authenticated, tenant-scoped access for all core data operations.
- Replace ad-hoc worker polling with atomic claim/lease mechanics for horizontal workers.
- Establish a phased delivery model for billing, reliability, and compliance hardening.

Success metrics:
- 100% of job creation and job reads happen through authenticated server APIs.
- Zero cross-tenant data leakage in RLS policy tests.
- Duplicate job-claim rate is 0 under concurrent worker claim tests.
- Queue recovery from worker crash occurs automatically after lease expiry.

## Current State Snapshot
- Web app dashboard/job detail flows now use authenticated API v1 routes (`/api/v1/jobs*`) with org-scoped access checks.
- Organization tenancy schema (`organizations`, `organization_memberships`) is in place.
- Core tables have `organization_id` with backfill and tenant indexes.
- RLS is enabled across tenant-scoped core tables with membership/role policies.
- Worker claim loop now supports atomic queue claim/lease/heartbeat/complete RPC flow.
- Blog artifact and destination publishing support exists with client profile model (`client_blog_profiles`, `client_publish_destinations`).
- Remaining Phase 1 work is primarily test coverage + rollout hardening (staging/prod smoke tests, feature-flag validation).

## Phase 1 (Security + Tenancy + Queue Claim)

Scope:
- Add organization tenancy schema + memberships.
- Add `organization_id` to core tables and backfill existing data.
- Enable RLS policies for tenant isolation.
- Add queue claim/heartbeat/complete RPC functions.
- Move job create/list/detail/retry to authenticated Route Handlers.
- Add active org cookie selection endpoint.
- Switch web dashboard pages to API boundary (no direct client writes).
- Update worker to use atomic claim/lease mode.

Non-goals:
- Stripe billing and usage metering.
- Enterprise compliance controls (SOC2 workflows, DPA automation).
- Full production-grade distributed rate limiter.

Acceptance criteria:
- Unauthenticated API write/read attempts are denied.
- Authenticated users see only jobs for active organization.
- Owner/admin can retry jobs; member cannot.
- Two workers cannot claim the same job concurrently.
- Expired leases are reclaimable.

Implementation checklist:
- [x] Add migrations `004`–`007` (tenancy, tenant fields, RLS, queue RPC).
- [x] Add web `requireOrgContext` helper + rate-limit helper.
- [x] Add `/api/v1/jobs`, `/api/v1/jobs/:id`, `/api/v1/jobs/:id/retry`, `/api/v1/orgs/active`.
- [x] Remove client-side direct write to `jobs` from dashboard.
- [x] Update worker claim loop to `claim_next_job` + heartbeat + complete.
- [ ] Add queue and API tests.
- [ ] Add dashboard create-job UI controls for `client_id` profile selection and `content_type`.
- [ ] Add dashboard affordances for job source metadata (`manual`, `batch_id`, `monitor_id`, `monitor_rule_id`).
- [ ] Add profile management UI for `client_blog_profiles` (list/create/edit/default marker).

## Phase 2 (Billing + Entitlements)
- Stripe subscriptions and webhook signature verification.
- Organization plan model + entitlements table.
- Enforce quotas (jobs/day, minutes processed, destinations count, storage quotas).
- Add quota counters and metering pipeline.

## Phase 3 (Reliability + Observability + Cost Controls)
- Structured logs with `organization_id`, `job_id`, `post_id`, `destination_id`.
- Metrics and alerts: claim latency, queue depth, failure rates, retry loops.
- Dead-letter strategy and replay tooling.
- Cost attribution per organization and model/provider.

## Phase 4 (Compliance + Enterprise Readiness)
- Data retention policies and deletion workflows.
- Audit logs with immutable event stream for admin/security events.
- Secret lifecycle hardening (Vault/KMS rotation policies).
- Compliance documentation and controls mapping.

## Risk Register
- RLS misconfiguration can break app availability or leak data.
- Queue RPC rollout can starve jobs if heartbeat or completion semantics are wrong.
- Service-role fallback behavior can create privilege confusion.
- In-memory rate limit is insufficient for multi-instance deployments.

## Open Decisions
- Final distributed rate limiter provider (Upstash Redis / Cloudflare KV / API gateway).
- Org invitation/onboarding UX and role management APIs.
- Cross-region job execution strategy for media-heavy workloads.

## Milestones / Target Dates
- M1: Phase 1 DB + API + worker claim completed and tested.
- M2: Phase 2 billing and quota enforcement in production.
- M3: Phase 3 reliability and observability dashboards with SLO alerts.
- M4: Phase 4 compliance and enterprise readiness package.

## Prioritized Backlog
- P0: Remove client-side direct writes to `jobs`.
- P0: Add org + membership schema.
- P0: Add tenant_id + RLS policies on core tables.
- P0: Add atomic `claim_next_job` RPC + lease/heartbeat.
- P1: Add billing/usage metering.
- P1: Add destination secret management (vault/KMS).
- P1: Add audit log + security events.
- P1: Add create-job UX to select generation profile per job and expose content type/source selectors.
- P1: Add blog profile management UX so org admins can maintain multiple profiles without SQL/manual API calls.
- P2: Add SLO dashboards + alerting.
- P2: Add data retention/deletion workflows.
