'use client';

import { useCallback, useEffect, useState, use } from 'react';
import { supabase } from '@/lib/supabase';
import { ChevronLeft, Play, CheckCircle, XCircle, Download } from 'lucide-react';
import Link from 'next/link';

interface Clip {
    id: string;
    title: string;
    start_seconds: number;
    end_seconds: number;
    transcript_excerpt: string;
    status: string;
    video_url?: string;
}

interface Job {
    id: string;
    title: string;
    youtube_url: string;
    video_url?: string;
    srt_url?: string;
    status: string;
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

export default function JobDetails({ params: paramsPromise }: { params: Promise<{ id: string }> }) {
    const params = use(paramsPromise);
    const [job, setJob] = useState<Job | null>(null);
    const [clips, setClips] = useState<Clip[]>([]);
    const [loading, setLoading] = useState(true);

    const fetchJobAndClips = useCallback(async (silent = false) => {
        if (!silent) setLoading(true);
        const { data: jobData } = await supabase.from('jobs').select('*').eq('id', params.id).single();
        const { data: clipsData } = await supabase.from('clips').select('*').eq('job_id', params.id);

        if (jobData) setJob(jobData);
        if (clipsData) setClips(clipsData);
        setLoading(false);
    }, [params.id]);

    useEffect(() => {
        const initial = setTimeout(() => {
            void fetchJobAndClips();
        }, 0);
        const interval = setInterval(() => fetchJobAndClips(true), 5000);
        return () => {
            clearTimeout(initial);
            clearInterval(interval);
        };
    }, [fetchJobAndClips]);

    return (
        <div className="min-h-screen bg-gray-50 p-4 sm:p-8" suppressHydrationWarning>
            <header className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex items-center gap-3 sm:gap-4 min-w-0">
                    <Link href="/" className="p-2 rounded-full hover:bg-gray-200">
                        <ChevronLeft size={24} />
                    </Link>
                    <h1 className="text-xl sm:text-3xl font-bold truncate">{job?.title || 'Review Clips'}</h1>
                </div>

                {job?.srt_url && (
                    <button
                        className="flex items-center justify-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors shadow-sm w-full sm:w-auto"
                        onClick={() => window.open(job.srt_url)}
                    >
                        <Download size={18} />
                        Download SRT
                    </button>
                )}
                {job && (
                    <button
                        className="flex items-center justify-center gap-2 px-4 py-2 bg-gray-900 text-white rounded-lg hover:bg-black transition-colors shadow-sm w-full sm:w-auto"
                        onClick={() => window.open(`/api/jobs/${job.id}/download`)}
                    >
                        <Download size={18} />
                        Download Full Resolution
                    </button>
                )}
            </header>

            {job && (
                <section className="mb-8 bg-white p-5 rounded-2xl border border-gray-200 shadow-sm">
                    <div className="flex items-center justify-between mb-3">
                        <p className="text-sm font-semibold text-gray-700">Pipeline Progress</p>
                        <p className="text-sm text-gray-500">
                            {getProgressState(job.status).determinate ? `${getProgressState(job.status).value}%` : 'Active'}
                        </p>
                    </div>
                    <div className="h-3 bg-gray-100 rounded-full overflow-hidden">
                        <div
                            className={`h-full transition-all ${job.status === 'failed' ? 'bg-red-500' : 'bg-blue-500'} ${getProgressState(job.status).determinate ? '' : 'animate-pulse'}`}
                            style={{ width: `${getProgressState(job.status).value}%` }}
                        />
                    </div>
                    <p className="text-sm text-gray-600 mt-2">{getStageLabel(job.status)} · {getProgressState(job.status).text}</p>
                </section>
            )}

            {job?.video_url && (
                <section className="mb-12 bg-white p-6 rounded-2xl border border-gray-200 shadow-sm">
                    <h2 className="text-xl font-bold mb-4">Full Sermon (Horizontal)</h2>
                    <div className="aspect-video bg-black rounded-xl overflow-hidden shadow-inner max-w-4xl mx-auto">
                        <video src={job.video_url} controls className="w-full h-full" />
                    </div>
                </section>
            )}

            <h2 className="text-lg sm:text-xl font-bold mb-6">Generated Short Clips (Vertical)</h2>

            <main className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {clips.map((clip) => (
                    <div key={clip.id} className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden hover:shadow-md transition-shadow">
                        <div className="aspect-[9/16] bg-black flex items-center justify-center relative group">
                            {clip.video_url ? (
                                <video
                                    src={clip.video_url}
                                    controls
                                    className="w-full h-full object-cover"
                                />
                            ) : (
                                <>
                                    <Play size={48} className="text-white opacity-50 group-hover:opacity-100 transition-opacity" />
                                    <div className="absolute bottom-4 left-4 right-4 bg-black/60 p-2 rounded text-xs text-white">
                                        {Math.floor(clip.start_seconds)}s - {Math.floor(clip.end_seconds)}s
                                    </div>
                                </>
                            )}
                        </div>

                        <div className="p-6">
                            <h3 className="font-bold text-lg mb-2">{clip.title}</h3>
                            <p className="text-sm text-gray-600 line-clamp-3 mb-6 italic">
                                &quot;{clip.transcript_excerpt}&quot;
                            </p>

                            <div className="flex flex-col sm:flex-row gap-3">
                                <button className="flex-1 flex items-center justify-center gap-2 py-2.5 bg-green-600 text-white rounded-xl font-medium hover:bg-green-700 transition-colors">
                                    <CheckCircle size={18} />
                                    Approve
                                </button>
                                <button className="flex-1 flex items-center justify-center gap-2 py-2.5 bg-red-50 text-red-600 rounded-xl font-medium hover:bg-red-100 transition-colors">
                                    <XCircle size={18} />
                                    Reject
                                </button>
                            </div>
                        </div>
                    </div>
                ))}

                {clips.length === 0 && !loading && (
                    <div className="col-span-full p-16 text-center text-gray-400 bg-white rounded-3xl border-2 border-dashed border-gray-100">
                        No clips generated yet. Wait for the processing to finish.
                    </div>
                )}
            </main>
        </div>
    );
}
