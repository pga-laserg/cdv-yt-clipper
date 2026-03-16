export interface ProgressState {
  determinate: boolean;
  value: number;
  stage: string;
  subtask: string;
  color: string;
  isStale?: boolean;
  staleMessage?: string;
}

const STAGES = {
  PENDING: { start: 0, end: 0, label: 'Queued' },
  INGEST: { start: 0, end: 15, label: 'Ingesting' },
  TRANSCRIBE: { start: 15, end: 40, label: 'Transcribing' },
  ANALYZE: { start: 40, end: 60, label: 'Analyzing' },
  RENDER: { start: 60, end: 85, label: 'Rendering' },
  STORE: { start: 85, end: 90, label: 'Saving' },
  BLOG: { start: 90, end: 95, label: 'Blog' },
  DELIVERY: { start: 95, end: 99, label: 'Delivery' },
  COMPLETED: { start: 100, end: 100, label: 'Completed' },
  FAILED: { start: 0, end: 100, label: 'Failed' },
};

import { JobRecord } from './api-types';

export function getDetailedProgress(job: JobRecord): ProgressState {
  if (!job) {
    return { determinate: false, value: 0, stage: 'Unknown', subtask: 'Missing job data', color: 'bg-gray-400' };
  }
  const { status } = job;
  
  // Stale detection
  let isStale = false;
  let staleMessage = '';

  const now = new Date().getTime();
  
  if (status === 'pending') {
    const createdAt = new Date(job.created_at).getTime();
    if (now - createdAt > 60000) { // 60 seconds
      isStale = true;
      staleMessage = 'Starting slowly. Check if a worker is online.';
    }
    return { determinate: true, value: 0, stage: 'Queued', subtask: 'Waiting for worker...', color: 'bg-gray-400', isStale, staleMessage };
  }
  
  if (status.startsWith('processing')) {
    if (job.lease_expires_at) {
      const expiresAt = new Date(job.lease_expires_at).getTime();
      if (now > expiresAt) {
        isStale = true;
        staleMessage = 'Worker heartbeat lost. It might be offline or crashed.';
      }
    }
  }

  if (status === 'completed') {
    return { determinate: true, value: 100, stage: 'Completed', subtask: 'All tasks finished', color: 'bg-green-500' };
  }
  if (status === 'failed') {
    return { determinate: true, value: 100, stage: 'Failed', subtask: 'Error in pipeline', color: 'bg-red-500' };
  }

  // Helper to map 0-100 within a stage to the global 0-100
  const mapRange = (stageRange: { start: number, end: number }, percent: number) => {
    return stageRange.start + (percent / 100) * (stageRange.end - stageRange.start);
  };

  // Ingest: processing:ingest:stage:percent%
  if (status.startsWith('processing:ingest')) {
    const parts = status.split(':');
    const subtask = parts[2] || 'initializing';
    const percent = parseInt(parts[3] || '0');
    return {
      determinate: true,
      value: mapRange(STAGES.INGEST, percent),
      stage: 'Ingesting',
      subtask: `${subtask} ${percent}%`,
      color: 'bg-blue-500',
      isStale,
      staleMessage
    };
  }

  // Transcribe: processing:transcribe:current/total
  if (status.startsWith('processing:transcribe')) {
    if (status === 'processing:transcribe') {
      return { determinate: false, value: STAGES.TRANSCRIBE.start, stage: 'Transcribing', subtask: 'Starting STT...', color: 'bg-indigo-500' };
    }
    const match = status.match(/processing:transcribe:(\d+)\/(\d+)/);
    if (match) {
      const current = parseInt(match[1]);
      const total = parseInt(match[2]);
      const percent = (current / total) * 100;
      return {
        determinate: true,
        value: mapRange(STAGES.TRANSCRIBE, percent),
        stage: 'Transcribing',
        subtask: `${Math.round(percent)}% complete`,
        color: 'bg-indigo-500',
        isStale,
        staleMessage
      };
    }
    return { determinate: false, value: STAGES.TRANSCRIBE.start, stage: 'Transcribing', subtask: 'Processing audio...', color: 'bg-indigo-500', isStale, staleMessage };
  }

  // Analyze: processing:analyze
  if (status.startsWith('processing:analyze')) {
    return { determinate: false, value: STAGES.ANALYZE.start + 5, stage: 'Analyzing', subtask: 'GPT-4o selecting highlights...', color: 'bg-purple-500', isStale, staleMessage };
  }

  // Render: processing:render:stage:percent%
  if (status.startsWith('processing:render')) {
    const parts = status.split(':');
    const subtask = parts[2] || 'initializing';
    const percent = parseInt(parts[3] || '0');
    return {
      determinate: true,
      value: mapRange(STAGES.RENDER, percent),
      stage: 'Rendering',
      subtask: `${subtask} ${percent}%`,
      color: 'bg-pink-500',
      isStale,
      staleMessage
    };
  }

  // Store: processing:store:done/total
  if (status.startsWith('processing:store')) {
    const match = status.match(/processing:store:(\d+)\/(\d+)/);
    if (match) {
      const done = parseInt(match[1]);
      const total = parseInt(match[2]);
      const percent = (done / total) * 100;
      return {
        determinate: true,
        value: mapRange(STAGES.STORE, percent),
        stage: 'Saving',
        subtask: `Clip ${done} of ${total}`,
        color: 'bg-cyan-500',
        isStale,
        staleMessage
      };
    }
    return { determinate: false, value: STAGES.STORE.start, stage: 'Saving', subtask: 'Uploading assets...', color: 'bg-cyan-500', isStale, staleMessage };
  }

  // Blog: processing:blog
  if (status.startsWith('processing:blog')) {
    return { determinate: false, value: STAGES.BLOG.start + 2, stage: 'Blog', subtask: 'Generating social copy...', color: 'bg-yellow-500', isStale, staleMessage };
  }

  // Delivery: processing:delivery
  if (status.startsWith('processing:delivery')) {
    if (status.includes('complete')) return { determinate: true, value: STAGES.DELIVERY.end, stage: 'Delivery', subtask: 'Ready', color: 'bg-teal-500', isStale, staleMessage };
    const match = status.match(/encode:(\d+)%/);
    const percent = match ? parseInt(match[1]) : 50;
    return {
      determinate: true,
      value: mapRange(STAGES.DELIVERY, percent),
      stage: 'Delivery',
      subtask: 'Encoding delivery package...',
      color: 'bg-teal-500',
      isStale,
      staleMessage
    };
  }

  return { determinate: false, value: 10, stage: 'Processing', subtask: 'Running pipeline...', color: 'bg-gray-500' };
}
