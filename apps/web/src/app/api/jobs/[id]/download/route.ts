import fs from 'fs';
import path from 'path';
import { Readable } from 'stream';
import { NextRequest, NextResponse } from 'next/server';
import { orgContextErrorResponse, requireOrgContext } from '@/lib/require-org-context';
import { getSupabaseServer } from '@/lib/supabase-server';

export const runtime = 'nodejs';

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

function resolveWorkerRootCandidates() {
  const cwd = process.cwd();
  return [
    path.resolve(cwd, 'apps/worker/work_dir'),
    path.resolve(cwd, '../worker/work_dir')
  ];
}

function isInside(parent: string, target: string) {
  const rel = path.relative(parent, target);
  return rel.length > 0 && !rel.startsWith('..') && !path.isAbsolute(rel);
}

function pickFullResPath(jobId: string, storedPath: unknown): string | null {
  const roots = resolveWorkerRootCandidates();
  const defaultCandidates = roots.map((root) =>
    path.resolve(root, jobId, 'processed', 'sermon_horizontal.mp4')
  );

  if (typeof storedPath === 'string' && storedPath.trim()) {
    const absolute = path.resolve(storedPath);
    if (roots.some((root) => isInside(root, absolute) || absolute === path.resolve(root, jobId, 'processed', 'sermon_horizontal.mp4'))) {
      return absolute;
    }
  }

  for (const candidate of defaultCandidates) {
    if (fs.existsSync(candidate)) return candidate;
  }
  return null;
}

function pickFullResRemoteUrl(storedUrl: unknown): string | null {
  if (typeof storedUrl !== 'string') return null;
  const candidate = storedUrl.trim();
  if (!candidate) return null;
  try {
    const url = new URL(candidate);
    if (url.protocol !== 'http:' && url.protocol !== 'https:') return null;
    return url.toString();
  } catch {
    return null;
  }
}

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ id: string }> }
) {
  const supabaseServer = getSupabaseServer();
  let orgId = '';
  try {
    const org = await requireOrgContext(request);
    orgId = org.organization_id;
  } catch (error) {
    return orgContextErrorResponse(error);
  }

  const { id } = await context.params;
  if (!UUID_RE.test(id)) {
    return NextResponse.json({ error: 'Invalid job id' }, { status: 400 });
  }

  const { data: job, error } = await supabaseServer
    .from('jobs')
    .select('id, organization_id, original_filename, metadata')
    .eq('id', id)
    .eq('organization_id', orgId)
    .single();

  if (error || !job) {
    return NextResponse.json({ error: 'Job not found' }, { status: 404 });
  }

  const remoteUrl = pickFullResRemoteUrl(job.metadata?.full_res_video_url);
  if (remoteUrl) {
    return NextResponse.redirect(remoteUrl, { status: 307 });
  }

  const fullResPath = pickFullResPath(id, job.metadata?.full_res_video_path);
  if (!fullResPath || !fs.existsSync(fullResPath)) {
    return NextResponse.json(
      { error: 'Full-resolution file is not available on this host' },
      { status: 404 }
    );
  }

  const stat = fs.statSync(fullResPath);
  const baseName = typeof job.original_filename === 'string' && job.original_filename.trim()
    ? path.parse(job.original_filename).name
    : `job-${id}`;
  const downloadName = `${baseName}-fullres.mp4`;

  const nodeStream = fs.createReadStream(fullResPath);
  const body = Readable.toWeb(nodeStream) as ReadableStream<Uint8Array>;

  return new NextResponse(body, {
    headers: {
      'Content-Type': 'video/mp4',
      'Content-Length': String(stat.size),
      'Content-Disposition': `attachment; filename="${downloadName}"`,
      'Cache-Control': 'no-store'
    }
  });
}
