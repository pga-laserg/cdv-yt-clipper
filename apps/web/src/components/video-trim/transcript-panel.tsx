"use client";

import { useEffect, useRef } from "react";
import { TranscriptSegment } from "@/lib/video/parse-srt";

export default function TranscriptPanel({
  transcript,
  currentTime,
  paddedStart, // video time 0 relative to actual SRT start
  onSeek,
  onSetStart,
  onSetEnd,
}: {
  transcript: TranscriptSegment[];
  currentTime: number;
  paddedStart: number;
  onSeek: (absoluteTime: number) => void;
  onSetStart: (absoluteTime: number) => void;
  onSetEnd: (absoluteTime: number) => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  
  // Convert current video time to absolute SRT time
  const currentAbsTime = currentTime + paddedStart;
  
  const activeIndex = transcript.findIndex(
    (s) => currentAbsTime >= s.start && currentAbsTime <= s.end
  );

  useEffect(() => {
    if (activeIndex >= 0 && scrollRef.current) {
      const activeEl = scrollRef.current.children[activeIndex] as HTMLElement;
      if (activeEl) {
        // Auto-scroll logic if out of view
        const containerTop = scrollRef.current.scrollTop;
        const containerBottom = containerTop + scrollRef.current.clientHeight;
        const elTop = activeEl.offsetTop;
        const elBottom = elTop + activeEl.clientHeight;

        if (elTop < containerTop || elBottom > containerBottom) {
          activeEl.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      }
    }
  }, [activeIndex]);

  return (
    <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-3">
      {transcript.map((seg, idx) => {
        const isActive = idx === activeIndex;
        return (
          <div
            key={seg.id || idx}
            className={`p-3 rounded-xl border transition-colors ${
              isActive
                ? "bg-blue-50 border-blue-200"
                : "bg-white border-gray-100 hover:border-gray-200"
            }`}
          >
            <button
              onClick={() => onSeek(seg.start)}
              className="text-left w-full block text-sm text-gray-800 leading-relaxed mb-3 hover:text-blue-600 transition-colors"
            >
              {seg.text}
            </button>
            <div className="flex items-center gap-2">
              <button
                onClick={() => onSetStart(seg.start)}
                className="text-xs px-2 py-1 rounded bg-gray-100 font-medium text-gray-600 hover:bg-gray-200"
              >
                Set start
              </button>
              <button
                onClick={() => onSetEnd(seg.end)}
                className="text-xs px-2 py-1 rounded bg-gray-100 font-medium text-gray-600 hover:bg-gray-200"
              >
                Set end
              </button>
              <div className="ml-auto text-xs text-gray-400 font-mono">
                {Math.floor(seg.start)}s
              </div>
            </div>
          </div>
        );
      })}
      
      {transcript.length === 0 && (
        <div className="text-center text-gray-400 text-sm mt-12">
          No transcript segments mapped to this window.
        </div>
      )}
    </div>
  );
}
