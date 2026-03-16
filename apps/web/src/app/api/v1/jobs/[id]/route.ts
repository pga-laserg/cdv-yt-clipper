import { NextRequest, NextResponse } from 'next/server';
import { getSupabaseServer } from '@/lib/supabase-server';
import { orgContextErrorResponse, requireOrgContext } from '@/lib/require-org-context';
import type { JobDetailResponse } from '@/lib/api-types';

export const runtime = 'nodejs';

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

function isApiEnabled(): boolean {
  return String(process.env.API_V1_JOBS_ENABLED ?? 'true').toLowerCase() === 'true';
}

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ id: string }> }
) {
  if (!isApiEnabled()) {
    return NextResponse.json(
      { error: 'API temporarily disabled', code: 'api_disabled' },
      { status: 503 }
    );
  }

  try {
    const supabaseServer = getSupabaseServer();
    const org = await requireOrgContext(request);
    const { id } = await context.params;

    if (!UUID_RE.test(id)) {
      return NextResponse.json({ error: 'Invalid job id' }, { status: 400 });
    }

    const { data: job, error: jobError } = await supabaseServer
      .from('jobs')
      .select('id, organization_id, created_at, status, source_url, title, video_url, srt_url, sermon_start_seconds, sermon_end_seconds, metadata, claimed_at, lease_expires_at')
      .eq('id', id)
      .eq('organization_id', org.organization_id)
      .maybeSingle();

    if (jobError) {
      return NextResponse.json(
        { error: `Failed to load job: ${jobError.message}` },
        { status: 500 }
      );
    }

    if (!job) {
      return NextResponse.json({ error: 'Job not found' }, { status: 404 });
    }

    const { data: clips, error: clipsError } = await supabaseServer
      .from('clips')
      .select('id, job_id, organization_id, start_seconds, end_seconds, title, transcript_excerpt, status, video_url')
      .eq('job_id', id)
      .eq('organization_id', org.organization_id)
      .order('created_at', { ascending: true });

    if (clipsError) {
      return NextResponse.json(
        { error: `Failed to load clips: ${clipsError.message}` },
        { status: 500 }
      );
    }

    const response: JobDetailResponse = {
      job: job as JobDetailResponse['job'],
      clips: (clips || []) as JobDetailResponse['clips']
    };

    return NextResponse.json(response);
  } catch (error) {
    return orgContextErrorResponse(error);
  }
}

export async function PATCH(
  request: NextRequest,
  context: { params: Promise<{ id: string }> }
) {
  if (!isApiEnabled()) {
    return NextResponse.json(
      { error: 'API temporarily disabled', code: 'api_disabled' },
      { status: 503 }
    );
  }

  try {
    const supabaseServer = getSupabaseServer();
    const org = await requireOrgContext(request);
    const { id } = await context.params;
    const body = await request.json();

    if (!UUID_RE.test(id)) {
      return NextResponse.json({ error: 'Invalid job id' }, { status: 400 });
    }

    const { mode } = body; // 'abort' or 'retry'
    let updateData: Record<string, any> = {};

    if (mode === 'abort') {
      updateData = {
        status: 'failed',
        last_error: 'Cancelled by user',
        claim_token: null,
        claimed_at: null,
        claimed_by: null,
        lease_expires_at: null
      };
    } else if (mode === 'retry') {
      updateData = {
        status: 'pending',
        last_error: null,
        claim_token: null,
        claimed_at: null,
        claimed_by: null,
        lease_expires_at: null,
        current_stage: null,
        progress_percentage: 0
      };
    } else {
      return NextResponse.json({ error: 'Invalid update mode' }, { status: 400 });
    }

    const { data, error } = await supabaseServer
      .from('jobs')
      .update(updateData)
      .eq('id', id)
      .eq('organization_id', org.organization_id)
      .select('id, status')
      .single();

    if (error) {
      return NextResponse.json({ error: error.message }, { status: 500 });
    }

    return NextResponse.json({ success: true, job: data });
  } catch (error) {
    return orgContextErrorResponse(error);
  }
}

export async function DELETE(
  request: NextRequest,
  context: { params: Promise<{ id: string }> }
) {
  if (!isApiEnabled()) {
    return NextResponse.json(
      { error: 'API temporarily disabled', code: 'api_disabled' },
      { status: 503 }
    );
  }

  try {
    const supabaseServer = getSupabaseServer();
    const org = await requireOrgContext(request);
    const { id } = await context.params;

    if (!UUID_RE.test(id)) {
      return NextResponse.json({ error: 'Invalid job id' }, { status: 400 });
    }

    // Actually delete the record
    const { error } = await supabaseServer
      .from('jobs')
      .delete()
      .eq('id', id)
      .eq('organization_id', org.organization_id);

    if (error) {
      return NextResponse.json(
        { error: `Failed to delete job: ${error.message}` },
        { status: 500 }
      );
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    return orgContextErrorResponse(error);
  }
}
