import { NextRequest, NextResponse } from 'next/server';
import { getSupabaseServer } from '@/lib/supabase-server';
import { checkRateLimit } from '@/lib/rate-limit';
import { orgContextErrorResponse, requireOrgContext } from '@/lib/require-org-context';
import type { CreateJobRequest, JobListResponse } from '@/lib/api-types';

export const runtime = 'nodejs';

function isApiEnabled(): boolean {
  return String(process.env.API_V1_JOBS_ENABLED ?? 'true').toLowerCase() === 'true';
}

function isValidYoutubeUrl(raw: string): boolean {
  try {
    const url = new URL(raw);
    const host = url.hostname.toLowerCase();
    return host.includes('youtube.com') || host.includes('youtu.be');
  } catch {
    return false;
  }
}

function parseLimit(value: string | null, fallback: number): number {
  const n = Number(value ?? fallback);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(1, Math.min(100, Math.floor(n)));
}

function parseOffset(value: string | null, fallback: number): number {
  const n = Number(value ?? fallback);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(0, Math.floor(n));
}

function cleanId(value: unknown): string {
  return String(value ?? '').trim();
}

function isSafeIdentifier(value: string): boolean {
  return /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(value);
}

function parseContentType(value: unknown): 'sermon' | 'podcast' | 'interview' | 'talk' | 'generic' {
  const normalized = String(value ?? '').trim().toLowerCase();
  if (
    normalized === 'sermon' ||
    normalized === 'podcast' ||
    normalized === 'interview' ||
    normalized === 'talk' ||
    normalized === 'generic'
  ) {
    return normalized;
  }
  return 'sermon';
}

async function resolveClientId(organizationId: string, requestedClientId?: string): Promise<string | null> {
  const supabaseServer = getSupabaseServer();
  const requested = cleanId(requestedClientId);
  if (requested) {
    const { data, error } = await supabaseServer
      .from('client_blog_profiles')
      .select('client_id')
      .eq('organization_id', organizationId)
      .eq('client_id', requested)
      .maybeSingle();

    if (error) return null;
    return cleanId(data?.client_id) || null;
  }

  const { data, error } = await supabaseServer
    .from('client_blog_profiles')
    .select('client_id')
    .eq('organization_id', organizationId)
    .order('created_at', { ascending: true })
    .limit(1)
    .maybeSingle();

  if (error) return 'default';
  if (typeof data?.client_id === 'string' && data.client_id.trim()) {
    return data.client_id.trim();
  }
  return 'default';
}

export async function GET(request: NextRequest) {
  if (!isApiEnabled()) {
    return NextResponse.json(
      { error: 'API temporarily disabled', code: 'api_disabled' },
      { status: 503 }
    );
  }

  try {
    const supabaseServer = getSupabaseServer();
    const org = await requireOrgContext(request);
    const { searchParams } = new URL(request.url);
    const limit = parseLimit(searchParams.get('limit'), 25);
    const offset = parseOffset(searchParams.get('offset'), 0);
    const statusFilter = (searchParams.get('status') || '').trim();

    let query = supabaseServer
      .from('jobs')
      .select(
        'id, organization_id, created_at, status, youtube_url, title, video_url, srt_url, sermon_start_seconds, sermon_end_seconds, metadata',
        { count: 'exact' }
      )
      .eq('organization_id', org.organization_id)
      .order('created_at', { ascending: false })
      .range(offset, offset + limit - 1);

    if (statusFilter && statusFilter !== 'all') {
      query = query.eq('status', statusFilter);
    }

    const { data, error, count } = await query;
    if (error) {
      return NextResponse.json(
        { error: `Failed to fetch jobs: ${error.message}` },
        { status: 500 }
      );
    }

    const response: JobListResponse = {
      items: (data || []) as JobListResponse['items'],
      count: count || 0,
      limit,
      offset
    };

    return NextResponse.json(response);
  } catch (error) {
    return orgContextErrorResponse(error);
  }
}

export async function POST(request: NextRequest) {
  if (!isApiEnabled()) {
    return NextResponse.json(
      { error: 'API temporarily disabled', code: 'api_disabled' },
      { status: 503 }
    );
  }

  try {
    const supabaseServer = getSupabaseServer();
    const org = await requireOrgContext(request);
    const contentLength = Number(request.headers.get('content-length') || '0');
    if (Number.isFinite(contentLength) && contentLength > 20_000) {
      return NextResponse.json(
        { error: 'Payload too large' },
        { status: 413 }
      );
    }

    const rate = checkRateLimit(`jobs:create:${org.user_id}:${org.organization_id}`, {
      limit: 12,
      windowMs: 60_000
    });
    if (!rate.allowed) {
      return NextResponse.json(
        { error: 'Rate limit exceeded', code: 'rate_limited' },
        {
          status: 429,
          headers: { 'Retry-After': String(rate.retryAfterSec) }
        }
      );
    }

    const body = (await request.json()) as CreateJobRequest;
    if (!body || typeof body !== 'object') {
      return NextResponse.json(
        { error: 'Invalid JSON body' },
        { status: 400 }
      );
    }

    const youtubeUrl = String(body.youtube_url || '').trim();
    if (!youtubeUrl || !isValidYoutubeUrl(youtubeUrl)) {
      return NextResponse.json(
        { error: 'youtube_url must be a valid YouTube URL' },
        { status: 400 }
      );
    }

    const metadataInput = body.metadata && typeof body.metadata === 'object' && !Array.isArray(body.metadata)
      ? { ...(body.metadata as Record<string, unknown>) }
      : {};

    const metadataSize = Buffer.byteLength(JSON.stringify(metadataInput), 'utf8');
    if (metadataSize > 10_000) {
      return NextResponse.json(
        { error: 'metadata exceeds max size (10KB)' },
        { status: 400 }
      );
    }

    const requestedClientId = cleanId(body.client_id || metadataInput.client_id);
    if (requestedClientId && !isSafeIdentifier(requestedClientId)) {
      return NextResponse.json(
        { error: 'client_id has invalid format' },
        { status: 400 }
      );
    }

    const resolvedClientId = await resolveClientId(org.organization_id, requestedClientId || undefined);
    if (!resolvedClientId) {
      return NextResponse.json(
        { error: `Unknown client_id '${requestedClientId}' for this organization` },
        { status: 400 }
      );
    }

    metadataInput.client_id = resolvedClientId;
    metadataInput.content_type = parseContentType(body.content_type ?? metadataInput.content_type);

    const batchId = cleanId(body.batch_id || metadataInput.batch_id);
    if (batchId) {
      if (!isSafeIdentifier(batchId)) {
        return NextResponse.json(
          { error: 'batch_id has invalid format' },
          { status: 400 }
        );
      }
      metadataInput.batch_id = batchId;
      metadataInput.job_source = 'batch';
    }

    const monitorId = cleanId(body.monitor_id || metadataInput.monitor_id);
    if (monitorId) {
      if (!isSafeIdentifier(monitorId)) {
        return NextResponse.json(
          { error: 'monitor_id has invalid format' },
          { status: 400 }
        );
      }
      metadataInput.monitor_id = monitorId;
      metadataInput.job_source = 'monitor';
    }

    const monitorRuleId = cleanId(body.monitor_rule_id || metadataInput.monitor_rule_id);
    if (monitorRuleId) {
      if (!isSafeIdentifier(monitorRuleId)) {
        return NextResponse.json(
          { error: 'monitor_rule_id has invalid format' },
          { status: 400 }
        );
      }
      metadataInput.monitor_rule_id = monitorRuleId;
      metadataInput.job_source = 'monitor';
    }

    if (!metadataInput.job_source) {
      metadataInput.job_source = 'manual';
    }

    const title = typeof body.title === 'string' && body.title.trim()
      ? body.title.trim().slice(0, 180)
      : 'New Sermon Job';

    const insertPayload = {
      organization_id: org.organization_id,
      youtube_url: youtubeUrl,
      status: 'pending',
      title,
      metadata: metadataInput
    };

    const { data, error } = await supabaseServer
      .from('jobs')
      .insert(insertPayload)
      .select('id, organization_id, created_at, status, youtube_url, title, metadata')
      .single();

    if (error) {
      return NextResponse.json(
        { error: `Failed to create job: ${error.message}` },
        { status: 500 }
      );
    }

    return NextResponse.json(data, { status: 201 });
  } catch (error) {
    return orgContextErrorResponse(error);
  }
}
