"use client";

import { Scissors } from "lucide-react";

export default function TrimToolbar({
  trimStart,
  trimEnd,
  maxDuration,
  isSubmitting,
  onSubmit,
}: {
  trimStart: number;
  trimEnd: number;
  maxDuration: number;
  isSubmitting: boolean;
  onSubmit: () => void;
}) {
  const duration = Math.max(0, trimEnd - trimStart);
  const isValid = duration > 0 && duration <= maxDuration;

  return (
    <div className="p-4 bg-white border rounded-xl flex items-center justify-between shadow-sm">
      <div className="flex flex-col">
        <span className="text-sm text-gray-500 font-medium">Selected Duration</span>
        <div className="flex items-baseline gap-2">
          <span className={`text-2xl font-bold tracking-tight ${!isValid ? 'text-red-500' : 'text-gray-900'}`}>
            {duration.toFixed(1)}s
          </span>
          <span className="text-sm text-gray-400">/ {maxDuration}s max</span>
        </div>
      </div>

      <button
        onClick={onSubmit}
        disabled={!isValid || isSubmitting}
        className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white font-medium rounded-xl hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-sm"
      >
        <Scissors size={20} />
        {isSubmitting ? "Queueing..." : "Submit Trim"}
      </button>
    </div>
  );
}
