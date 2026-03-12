"use client";

import { RefObject } from "react";

export default function VideoPlayer({
  videoRef,
  videoUrl,
  onLoadedMetadata,
  onTimeUpdate,
}: {
  videoRef: RefObject<HTMLVideoElement | null>;
  videoUrl: string;
  onLoadedMetadata: (duration: number) => void;
  onTimeUpdate: (time: number) => void;
}) {
  return (
    <div className="bg-black rounded-2xl aspect-video overflow-hidden shadow-inner flex items-center justify-center relative group">
      <video
        ref={videoRef}
        src={videoUrl}
        controls
        className="w-full h-full"
        onLoadedMetadata={(e) => onLoadedMetadata(e.currentTarget.duration)}
        onTimeUpdate={(e) => onTimeUpdate(e.currentTarget.currentTime)}
      />
    </div>
  );
}
