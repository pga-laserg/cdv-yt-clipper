import { getSupabaseServer } from "@/lib/supabase-server";
import { parseSrtToSegments, type TranscriptSegment } from "@/lib/video/parse-srt";
import VideoTrimEditor from "@/components/video-trim/video-trim-editor";

export default async function ClipTrimmerPage({
  params
}: {
  params: { id: string };
}) {
  const supabase = getSupabaseServer();
  const { id } = params;

  // Fetch clip details
  const { data: clip, error: clipError } = await supabase
    .from("clips")
    .select("*, jobs(srt_url)")
    .eq("id", id)
    .single();

  if (clipError || !clip) {
    return (
      <div className="min-h-screen flex items-center justify-center p-8 bg-gray-50 text-red-600">
        <p>Clip not found or an error occurred.</p>
      </div>
    );
  }

  const srtUrl = clip.jobs?.srt_url;
  let transcript: TranscriptSegment[] = [];

  if (srtUrl) {
    try {
      const response = await fetch(srtUrl);
      if (response.ok) {
        const srtText = await response.text();
        transcript = parseSrtToSegments(srtText);
      }
    } catch (error) {
      console.error("Failed to fetch/parse SRT:", error);
    }
  }

  // Filter transcript to just the ones around the existing clip window.
  // We'll give it some padding (120s before, 120s after) just in case.
  const paddedStart = Math.max(0, clip.start_seconds - 120);
  const paddedEnd = clip.end_seconds + 120;
  
  const relevantTranscript = transcript.filter(
    (seg) => seg.start >= paddedStart && seg.end <= paddedEnd
  );

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center">
      <header className="w-full max-w-7xl mx-auto p-4 sm:p-8 shrink-0 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Trim Clip</h1>
          <p className="text-sm text-gray-500">
            {clip.title || "Untitled Clip"}
          </p>
        </div>
      </header>
      
      <main className="w-full max-w-7xl mx-auto p-4 sm:p-8 grow">
        <VideoTrimEditor
          clipId={clip.id}
          videoUrl={clip.video_url}
          organizationId={clip.organization_id}
          initialStartSec={clip.start_seconds}
          initialEndSec={clip.end_seconds}
          transcript={relevantTranscript}
        />
      </main>
    </div>
  );
}
