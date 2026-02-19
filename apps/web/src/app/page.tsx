'use client';

import { useCallback, useEffect, useState } from 'react';
import { supabase } from '@/lib/supabase';
import Link from 'next/link';
import { Play, Clock, CheckCircle, AlertCircle } from 'lucide-react';

interface Job {
  id: string;
  title: string;
  status: string;
  created_at: string;
  youtube_url: string;
  video_url?: string;
}

function formatClock(totalSeconds: number): string {
  const seconds = Math.max(0, Math.floor(totalSeconds));
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
  return `${m}:${String(s).padStart(2, '0')}`;
}

function getProgressState(status: string): { determinate: boolean; value: number; text: string } {
  if (status === 'pending') return { determinate: true, value: 0, text: 'Queued' };
  if (status === 'completed') return { determinate: true, value: 100, text: '100%' };
  if (status === 'failed') return { determinate: true, value: 100, text: 'Failed' };

  if (status.startsWith('processing:transcribe:')) {
    const match = status.match(/processing:transcribe:(\d+)\/(\d+)/);
    if (!match) return { determinate: false, value: 45, text: 'Transcribing...' };
    const current = Number(match[1]);
    const total = Number(match[2]);
    if (!total) return { determinate: false, value: 45, text: 'Transcribing...' };
    const ratio = Math.min(1, current / total);
    return {
      determinate: true,
      value: Math.max(1, Math.round(ratio * 100)),
      text: `${formatClock(current)} / ${formatClock(total)}`
    };
  }

  if (status.startsWith('processing:store')) {
    const match = status.match(/processing:store:(\d+)\/(\d+)/);
    if (!match) return { determinate: false, value: 45, text: 'Saving outputs...' };
    const done = Number(match[1]);
    const total = Number(match[2]);
    if (!total) return { determinate: false, value: 45, text: 'Saving outputs...' };
    const ratio = Math.min(1, done / total);
    return {
      determinate: true,
      value: Math.max(1, Math.round(ratio * 100)),
      text: `${done}/${total} clips saved`
    };
  }

  if (status.startsWith('processing')) {
    return { determinate: false, value: 45, text: 'In progress...' };
  }

  return { determinate: false, value: 45, text: 'In progress...' };
}

function getStageLabel(status: string): string {
  if (status === 'pending') return 'Queued';
  if (status === 'completed') return 'Completed';
  if (status === 'failed') return 'Failed';
  if (status.startsWith('processing:transcribe:')) return 'Transcribing audio';
  if (status.startsWith('processing:ingest')) return 'Ingesting source';
  if (status.startsWith('processing:transcribe')) return 'Transcribing audio';
  if (status.startsWith('processing:analyze')) return 'Finding highlights';
  if (status.startsWith('processing:render')) return 'Rendering clips';
  if (status.startsWith('processing:store')) {
    const match = status.match(/processing:store:(\d+)\/(\d+)/);
    if (match) return `Saving clips (${match[1]}/${match[2]})`;
    return 'Saving outputs';
  }
  return status;
}

function getThumbnailUrl(job: Job): string | null {
  if (!job.video_url) return null;
  return job.video_url.replace(/\/sermon_horizontal\.mp4(\?.*)?$/, '/thumbnail.jpg');
}

export default function Home() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchJobs = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);
    const { data, error } = await supabase
      .from('jobs')
      .select('*')
      .order('created_at', { ascending: false });

    if (error) console.error('Error fetching jobs:', error);
    else setJobs(data || []);
    setLoading(false);
  }, []);

  useEffect(() => {
    const initial = setTimeout(() => {
      void fetchJobs();
    }, 0);
    const interval = setInterval(() => fetchJobs(true), 5000);
    return () => {
      clearTimeout(initial);
      clearInterval(interval);
    };
  }, [fetchJobs]);

  return (
    <div className="min-h-screen bg-white" suppressHydrationWarning>
      <main className="max-w-6xl mx-auto p-4 sm:p-8 text-black">
        <header className="flex flex-col gap-4 sm:flex-row sm:justify-between sm:items-center mb-8 sm:mb-12">
          <div>
            <h1 className="text-2xl sm:text-4xl font-bold text-gray-900 tracking-tight">Sermon Shorts</h1>
            <p className="text-gray-500 mt-1 sm:mt-2 text-sm sm:text-lg">Automation Pipeline Dashboard</p>
          </div>
          <div className="flex flex-col sm:flex-row gap-2 w-full sm:w-auto">
            <input
              id="youtube-url"
              type="text"
              placeholder="YouTube URL..."
              className="px-4 py-2 border rounded-full text-sm w-full sm:w-72"
            />
            <button
              onClick={async () => {
                const input = document.getElementById('youtube-url') as HTMLInputElement;
                const url = input?.value;
                if (!url) return;

                setLoading(true);
                const { error } = await supabase.from('jobs').insert({
                  youtube_url: url,
                  status: 'pending',
                  title: 'New Sermon Job'
                });

                if (error) alert(error.message);
                else {
                  input.value = '';
                  fetchJobs(true);
                }
                setLoading(false);
              }}
              className="bg-black text-white px-6 py-2 rounded-full font-medium hover:bg-gray-800 transition-colors shadow-lg w-full sm:w-auto"
            >
              Add Job
            </button>
          </div>
        </header>

        <section>
          <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">
            <Clock size={20} className="text-blue-600" />
            Recent Jobs
          </h2>

          <div className="space-y-4">
            {jobs.map((job) => (
              <Link
                key={job.id}
                href={`/jobs/${job.id}`}
                className="block p-4 sm:p-6 bg-white border border-gray-100 rounded-2xl shadow-sm hover:shadow-md hover:border-blue-100 transition-all group"
              >
                <div className="flex flex-col gap-4 sm:flex-row sm:justify-between sm:items-center">
                  <div className="flex items-center gap-4">
                    <div className="w-16 h-12 sm:w-20 sm:h-14 bg-blue-50 rounded-lg overflow-hidden flex items-center justify-center text-blue-600 shrink-0">
                      {getThumbnailUrl(job) ? (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img
                          src={getThumbnailUrl(job)!}
                          alt="Job thumbnail"
                          className="w-full h-full object-cover"
                          loading="lazy"
                        />
                      ) : (
                        <Play size={24} />
                      )}
                    </div>
                    <div className="min-w-0">
                      <h3 className="text-base sm:text-lg font-bold text-gray-900 group-hover:text-blue-600 transition-colors">
                        {job.title || 'Untitled Sermon'}
                      </h3>
                      <p className="text-xs sm:text-sm text-gray-500 font-mono break-all">{job.youtube_url}</p>
                    </div>
                  </div>

                  <div className="w-full sm:w-auto">
                    <div className="text-left sm:text-right">
                      <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${job.status === 'completed' ? 'bg-green-100 text-green-700' :
                        job.status === 'failed' ? 'bg-red-100 text-red-700' :
                          'bg-blue-100 text-blue-700 animate-pulse'
                        }`}>
                        {job.status === 'completed' && <CheckCircle size={12} />}
                        {job.status === 'failed' && <AlertCircle size={12} />}
                        {getStageLabel(job.status)}
                      </span>
                      {(() => {
                        const progress = getProgressState(job.status);
                        return (
                          <div className="mt-2 w-full sm:w-44 sm:ml-auto">
                            <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                              <div
                                className={`h-full transition-all ${job.status === 'failed' ? 'bg-red-500' : 'bg-blue-500'} ${progress.determinate ? '' : 'animate-pulse'}`}
                                style={{ width: `${progress.value}%` }}
                              />
                            </div>
                            <p className="text-xs text-gray-400 mt-1">{progress.text}</p>
                          </div>
                        );
                      })()}
                      <p className="text-xs text-gray-400 mt-1">
                        {new Date(job.created_at).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                </div>
              </Link>
            ))}

            {loading && <div className="p-12 text-center text-gray-400">Loading jobs...</div>}
            {!loading && jobs.length === 0 && (
              <div className="p-8 sm:p-12 text-center text-gray-400 border-2 border-dashed rounded-2xl">
                No jobs found. Start by creating a new one.
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
