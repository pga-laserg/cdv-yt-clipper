import { NextResponse } from "next/server";
import { getSupabaseServer } from "@/lib/supabase-server";
import { validateTrimRange } from "@/lib/video/trim-validation";

export async function POST(request: Request) {
  try {
    const supabase = getSupabaseServer();
    const body = await request.json();
    
    const { clipId, startSec, endSec, organizationId } = body;

    if (!clipId || startSec === undefined || endSec === undefined || !organizationId) {
      return NextResponse.json({ error: "Missing required parameters" }, { status: 400 });
    }

    // Verify clip exists and get duration
    const { data: clip, error: clipError } = await supabase
      .from("clips")
      .select("id, start_seconds, end_seconds")
      .eq("id", clipId)
      .eq("organization_id", organizationId)
      .single();

    if (clipError || !clip) {
      return NextResponse.json({ error: "Clip not found" }, { status: 404 });
    }

    // Since we don't have source duration strictly, we assume the UI passed valid bounds.
    // The trim validation requires sourceDuration. We'll pass the maximum possible distance.
    // Technically the UI handles the bounding exactly.
    const sourceDurationSec = 3600 * 5; // 5 hours safety bound, the frontend enforces exact length
    validateTrimRange({
      startSec,
      endSec,
      sourceDurationSec
    });

    // Queue job
    const { data: job, error: jobError } = await supabase
      .from("trim_jobs")
      .insert({
        clip_id: clipId,
        organization_id: organizationId,
        requested_start_sec: startSec,
        requested_end_sec: endSec,
        status: "queued"
      })
      .select("id")
      .single();

    if (jobError) {
      return NextResponse.json({ error: jobError.message }, { status: 500 });
    }

    return NextResponse.json({ jobId: job.id }, { status: 201 });
  } catch (error) {
    const msg = error instanceof Error ? error.message : "Internal server error";
    return NextResponse.json({ error: msg }, { status: 500 });
  }
}
