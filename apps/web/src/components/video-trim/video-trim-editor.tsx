"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { supabase } from "@/lib/supabase";
import { TranscriptSegment } from "@/lib/video/parse-srt";
import { clampTrimRange, MAX_FINAL_LENGTH_SEC } from "@/lib/video/clamp-range";
import VideoPlayer from "./video-player";
import WaveformTimeline from "./waveform-timeline";
import TranscriptPanel from "./transcript-panel";
import TrimToolbar from "./trim-toolbar";

export default function VideoTrimEditor({
  clipId,
  videoUrl,
  organizationId,
  initialStartSec,
  initialEndSec,
  transcript,
}: {
  clipId: string;
  videoUrl: string;
  organizationId: string;
  initialStartSec: number;
  initialEndSec: number;
  transcript: TranscriptSegment[];
}) {
  const router = useRouter();
  const videoRef = useRef<HTMLVideoElement | null>(null);
  
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [trimStart, setTrimStart] = useState(0);
  const [trimEnd, setTrimEnd] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    // We initially estimate the bounds to match the original clip window
    // (Relative to the padded output). If video loads, we can rely on real video time.
    setTrimStart(0);
    setTrimEnd(initialEndSec - initialStartSec); // Wait, this is relative to the video file?
    // The videoUrl is a padded video rendering, so start/end of the clip are within it.
    // If the video file itself is exactly paddedStart to paddedEnd, then we should sync
    // the UI with the file's duration.
  }, [initialStartSec, initialEndSec]);

  const onLoadedMetadata = (d: number) => {
    setDuration(d);
    // As a default, the initial clip was padded by 60s before and 120s after. Let's assume the clip falls in the middle.
    // BUT we don't know the exact padding used for rendering this specific `.mp4`. 
    // We will just set start to initialStartSec if possible, or 0 if it doesn't fit.
    const start = 0; // Simplified for now
    const end = Math.min(d, MAX_FINAL_LENGTH_SEC);
    setTrimStart(start);
    setTrimEnd(end);
  };

  const handleUpdateEnd = (start: number, end: number) => {
    const clamped = clampTrimRange(start, end, duration);
    setTrimStart(clamped.startSec);
    setTrimEnd(clamped.endSec);
  };

  const setStartFromTranscript = (t: number) => {
    // t is an absolute time from SRT. We need to map it to the video's relative time.
    // The video starts at `paddedStart`. So video time = t - paddedStart.
    const paddedStart = Math.max(0, initialStartSec - 60); // Assuming 60s padding
    const vidTime = Math.max(0, t - paddedStart);
    handleUpdateEnd(vidTime, trimEnd);
  };

  const setEndFromTranscript = (t: number) => {
    const paddedStart = Math.max(0, initialStartSec - 60);
    const vidTime = Math.max(0, t - paddedStart);
    handleUpdateEnd(trimStart, vidTime);
  };

  const seek = (t: number) => {
    if (videoRef.current) videoRef.current.currentTime = t;
  };

  const seekFromTranscript = (t: number) => {
    const paddedStart = Math.max(0, initialStartSec - 60);
    seek(Math.max(0, t - paddedStart));
  };

  const submitTrimJob = async () => {
    setIsSubmitting(true);
    try {
      const { data } = await supabase.auth.getSession();
      const token = data.session?.access_token;
      
      const headers: Record<string, string> = { "Content-Type": "application/json" };
      if (token) {
        headers.Authorization = `Bearer ${token}`;
      }

      const response = await fetch("/api/admin/trim", {
        method: "POST",
        headers,
        body: JSON.stringify({
          clipId,
          organizationId,
          startSec: trimStart,
          endSec: trimEnd,
        }),
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      alert("Trim job submitted successfully!");
      router.push("/");
    } catch (error) {
      alert(`Submission failed: ${error}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-6">
      <div className="space-y-6 flex flex-col">
        <VideoPlayer
          videoRef={videoRef}
          videoUrl={videoUrl}
          onLoadedMetadata={onLoadedMetadata}
          onTimeUpdate={setCurrentTime}
        />

        <WaveformTimeline
          videoUrl={videoUrl}
          duration={duration}
          trimStart={trimStart}
          trimEnd={trimEnd}
          onTrimChange={handleUpdateEnd}
          onSeek={seek}
        />

        <TrimToolbar
          trimStart={trimStart}
          trimEnd={trimEnd}
          maxDuration={MAX_FINAL_LENGTH_SEC}
          isSubmitting={isSubmitting}
          onSubmit={submitTrimJob}
        />
      </div>

      <aside className="border rounded-2xl bg-white shadow-sm flex flex-col h-[800px] overflow-hidden">
        <div className="p-4 border-b bg-gray-50 flex items-center justify-between font-medium shrink-0">
          <span>Transcript</span>
        </div>
        <TranscriptPanel
          transcript={transcript}
          currentTime={currentTime}
          paddedStart={Math.max(0, initialStartSec - 60)}
          onSeek={seekFromTranscript}
          onSetStart={setStartFromTranscript}
          onSetEnd={setEndFromTranscript}
        />
      </aside>
    </div>
  );
}
