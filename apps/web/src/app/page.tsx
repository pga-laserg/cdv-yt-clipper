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
}

function getProgress(status: string): number {
  if (status === 'pending') return 0;
  if (status === 'completed') return 100;
  if (status === 'failed') return 100;
  if (status.startsWith('processing:ingest')) return 15;
  if (status.startsWith('processing:transcribe')) return 35;
  if (status.startsWith('processing:analyze')) return 55;
  if (status.startsWith('processing:render')) return 75;
  if (status.startsWith('processing:store')) {
    const match = status.match(/processing:store:(\d+)\/(\d+)/);
    if (!match) return 90;
    const done = Number(match[1]);
    const total = Number(match[2]);
    if (!total) return 90;
    return Math.min(99, 90 + Math.floor((done / total) * 9));
  }
  if (status.startsWith('processing')) return 10;
  return 0;
}

function getStageLabel(status: string): string {
  if (status === 'pending') return 'Queued';
  if (status === 'completed') return 'Completed';
  if (status === 'failed') return 'Failed';
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
      <main className="max-w-6xl mx-auto p-8 text-black">
        <header className="flex justify-between items-center mb-12">
          <div>
            <h1 className="text-4xl font-bold text-gray-900 tracking-tight">Sermon Shorts</h1>
            <p className="text-gray-500 mt-2 text-lg">Automation Pipeline Dashboard</p>
          </div>
          <div className="flex gap-2">
            <input
              id="youtube-url"
              type="text"
              placeholder="YouTube URL..."
              className="px-4 py-2 border rounded-full text-sm w-64"
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
              className="bg-black text-white px-6 py-2 rounded-full font-medium hover:bg-gray-800 transition-colors shadow-lg"
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
                className="block p-6 bg-white border border-gray-100 rounded-2xl shadow-sm hover:shadow-md hover:border-blue-100 transition-all group"
              >
                <div className="flex justify-between items-center">
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 bg-blue-50 rounded-full flex items-center justify-center text-blue-600">
                      <Play size={24} />
                    </div>
                    <div>
                      <h3 className="text-lg font-bold text-gray-900 group-hover:text-blue-600 transition-colors">
                        {job.title || 'Untitled Sermon'}
                      </h3>
                      <p className="text-sm text-gray-500 font-mono">{job.youtube_url}</p>
                    </div>
                  </div>

                  <div className="flex items-center gap-6">
                    <div className="text-right">
                      <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${job.status === 'completed' ? 'bg-green-100 text-green-700' :
                        job.status === 'failed' ? 'bg-red-100 text-red-700' :
                          'bg-blue-100 text-blue-700 animate-pulse'
                        }`}>
                        {job.status === 'completed' && <CheckCircle size={12} />}
                        {job.status === 'failed' && <AlertCircle size={12} />}
                        {getStageLabel(job.status)}
                      </span>
                      <div className="mt-2 w-40 ml-auto">
                        <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                          <div
                            className={`h-full transition-all ${job.status === 'failed' ? 'bg-red-500' : 'bg-blue-500'}`}
                            style={{ width: `${getProgress(job.status)}%` }}
                          />
                        </div>
                        <p className="text-xs text-gray-400 mt-1">{getProgress(job.status)}%</p>
                      </div>
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
              <div className="p-12 text-center text-gray-400 border-2 border-dashed rounded-2xl">
                No jobs found. Start by creating a new one.
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
