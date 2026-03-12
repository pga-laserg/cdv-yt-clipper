"use client";

import { useEffect, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import RegionsPlugin from "wavesurfer.js/dist/plugins/regions.esm.js";

export default function WaveformTimeline({
  videoUrl,
  duration,
  trimStart,
  trimEnd,
  onTrimChange,
  onSeek,
}: {
  videoUrl: string;
  duration: number;
  trimStart: number;
  trimEnd: number;
  onTrimChange: (start: number, end: number) => void;
  onSeek: (time: number) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WaveSurfer | null>(null);
  const regionsRef = useRef<RegionsPlugin | null>(null);
  const isUpdatingRef = useRef(false);

  useEffect(() => {
    if (!containerRef.current || !videoUrl) return;

    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: "#cbd5e1",
      progressColor: "#3b82f6",
      url: videoUrl,
      height: 80,
      barWidth: 2,
      interact: true,
    });
    
    wsRef.current = ws;

    const regions = ws.registerPlugin(RegionsPlugin.create());
    regionsRef.current = regions;

    ws.on("click", (relativeX: number) => {
      if (duration) {
        onSeek(relativeX * duration);
      }
    });

    return () => {
      ws.destroy();
    };
  }, [videoUrl]); // Don't re-create on duration change if already initialized

  useEffect(() => {
    if (!regionsRef.current || !duration || isUpdatingRef.current) return;

    const regions = regionsRef.current;
    regions.clearRegions();
    
    // Protect against invalid region
    if (trimEnd <= trimStart) return;

    const region = regions.addRegion({
      start: trimStart,
      end: trimEnd,
      color: "rgba(59, 130, 246, 0.3)",
      drag: true,
      resize: true,
    });

    region.on("update-end", () => {
      isUpdatingRef.current = true;
      onTrimChange(region.start, region.end);
      setTimeout(() => { isUpdatingRef.current = false; }, 50);
    });
  }, [trimStart, trimEnd, duration]);

  return (
    <div className="bg-white rounded-xl border p-4 shadow-sm">
      <div className="mb-2 flex justify-between text-xs text-gray-500 font-medium">
        <span>Timeline</span>
        <span>Drag handles to trim</span>
      </div>
      <div ref={containerRef} className="w-full relative" />
    </div>
  );
}
