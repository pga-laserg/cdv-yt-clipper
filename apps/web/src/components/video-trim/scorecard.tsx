"use client";

import React from "react";

interface ScoreBreakdown {
  hook_strength?: number;
  spiritual_impact?: number;
  shareability?: number;
  ending_completeness?: number;
  model_confidence?: number;
}

interface ScorecardProps {
  scoreBreakdown?: ScoreBreakdown;
  hookType?: string;
}

const ScoreBar = ({ label, value, color }: { label: string; value: number; color: string }) => {
  const percent = Math.round(value * 100);
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-[10px] uppercase tracking-wider font-semibold text-gray-500">
        <span>{label}</span>
        <span>{percent}%</span>
      </div>
      <div className="h-1.5 w-full bg-gray-100 rounded-full overflow-hidden">
        <div 
          className={`h-full ${color} transition-all duration-500 ease-out`} 
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
};

export default function Scorecard({ scoreBreakdown, hookType }: ScorecardProps) {
  if (!scoreBreakdown) return null;

  const hookStrength = scoreBreakdown.hook_strength ?? 0;
  const spiritualImpact = scoreBreakdown.spiritual_impact ?? 0;
  const shareability = scoreBreakdown.shareability ?? 0;
  const endingCompleteness = scoreBreakdown.ending_completeness ?? 0;

  // Calculate overall score (weighted)
  const overallScore = Math.round(
    (hookStrength * 0.20 + 
     spiritualImpact * 0.30 + 
     shareability * 0.15 + 
     endingCompleteness * 0.35) * 100
  );

  const getHookBadgeColor = (type?: string) => {
    switch (type) {
      case 'question': return 'bg-blue-100 text-blue-700 border-blue-200';
      case 'promise': return 'bg-green-100 text-green-700 border-green-200';
      case 'scripture': return 'bg-purple-100 text-purple-700 border-purple-200';
      case 'story': return 'bg-orange-100 text-orange-700 border-orange-200';
      case 'contrast': return 'bg-red-100 text-red-700 border-red-200';
      default: return 'bg-gray-100 text-gray-700 border-gray-200';
    }
  };

  return (
    <div className="p-4 bg-white border-b space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex flex-col">
          <span className="text-[10px] uppercase tracking-widest text-gray-400 font-bold">Virality Score</span>
          <span className="text-3xl font-black text-gray-900">{overallScore}<span className="text-sm font-normal text-gray-400">/100</span></span>
        </div>
        {hookType && hookType !== 'none' && (
          <div className={`px-2 py-1 rounded-md border text-[10px] font-bold uppercase tracking-tight ${getHookBadgeColor(hookType)}`}>
            {hookType} Hook
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 gap-3">
        <ScoreBar label="Hook Strength" value={hookStrength} color="bg-indigo-500" />
        <ScoreBar label="Spiritual Impact" value={spiritualImpact} color="bg-rose-500" />
        <ScoreBar label="Shareability" value={shareability} color="bg-amber-500" />
        <ScoreBar label="Ending Completeness" value={endingCompleteness} color="bg-emerald-500" />
      </div>
    </div>
  );
}
