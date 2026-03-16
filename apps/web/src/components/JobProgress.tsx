'use client';

import React from 'react';
import { getDetailedProgress } from '@/lib/progress';

import { AlertCircle } from 'lucide-react';
import { JobRecord } from '@/lib/api-types';

interface JobProgressProps {
  job: JobRecord;
}

export const JobProgress: React.FC<JobProgressProps> = ({ job }) => {
  const progress = getDetailedProgress(job);
  
  // Custom gradient based on stage
  const getGradient = (stage: string) => {
    switch (stage) {
      case 'Ingesting': return 'from-blue-400 to-blue-600';
      case 'Transcribing': return 'from-indigo-400 to-indigo-600';
      case 'Analyzing': return 'from-purple-400 to-purple-600';
      case 'Rendering': return 'from-pink-400 to-rose-600';
      case 'Saving': return 'from-cyan-400 to-teal-600';
      case 'Blog': return 'from-yellow-400 to-orange-500';
      case 'Delivery': return 'from-emerald-400 to-green-600';
      case 'Completed': return 'from-green-400 to-green-600';
      case 'Failed': return 'from-red-400 to-red-600';
      default: return 'from-gray-400 to-gray-600';
    }
  };

  const gradientClass = getGradient(progress.stage);

  return (
    <div className="w-full space-y-2">
      <div className="flex justify-between items-end">
        <div className="flex flex-col">
          <span className="text-[10px] uppercase tracking-widest font-bold text-gray-400 mb-0.5">
            {progress.stage}
          </span>
          <span className="text-sm font-medium text-gray-700 truncate max-w-[150px] sm:max-w-[200px]">
            {progress.subtask}
          </span>
        </div>
        <span className="text-sm font-bold text-gray-900">
          {Math.round(progress.value)}%
        </span>
      </div>
      
      <div className="relative h-2.5 w-full bg-gray-100 rounded-full overflow-hidden shadow-inner">
        {/* Animated Glow Effect */}
        <div 
          className={`absolute top-0 left-0 h-full bg-gradient-to-r ${gradientClass} transition-all duration-500 ease-out rounded-full shadow-[0_0_10px_rgba(0,0,0,0.1)]`}
          style={{ width: `${progress.value}%` }}
        >
          {/* Shimmer overlay for active stages */}
          {!['Completed', 'Failed', 'Queued'].includes(progress.stage) && (
            <div className="absolute inset-0 w-full h-full animate-[shimmer_2s_infinite] bg-gradient-to-r from-transparent via-white/30 to-transparent -translate-x-full" />
          )}
        </div>
      </div>
      
      {progress.isStale && (
        <div className="mt-3 flex items-start gap-2 p-3 bg-amber-50 border border-amber-200 rounded-xl text-amber-700 text-[11px] leading-relaxed animate-in fade-in slide-in-from-top-1 duration-300">
          <AlertCircle size={14} className="mt-0.5 shrink-0" />
          <p>{progress.staleMessage}</p>
        </div>
      )}

      <style jsx>{`
        @keyframes shimmer {
          100% {
            transform: translateX(100%);
          }
        }
      `}</style>
    </div>
  );
};
