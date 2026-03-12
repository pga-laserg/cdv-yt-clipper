'use client';

import { useCallback, useEffect, useState, use } from 'react';
import { supabase } from '@/lib/supabase';
import type { ClipRecord, JobDetailResponse, JobRecord } from '@/lib/api-types';
import { ChevronLeft, Play, CheckCircle, XCircle, Download, Scissors } from 'lucide-react';
import Link from 'next/link';

type Clip = ClipRecord;
type Job = JobRecord;

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

    if (status.startsWith('processing:blog:publish')) {
        return { determinate: false, value: 96, text: 'Publishing blog destinations...' };
    }

    if (status.startsWith('processing:blog')) {
        return { determinate: false, value: 92, text: 'Generating blog artifact...' };
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
    if (status.startsWith('processing:blog:generate')) return 'Generating blog draft';
    if (status.startsWith('processing:blog:persist')) return 'Saving blog draft';
    if (status.startsWith('processing:blog:sync')) return 'Syncing blog draft';
    if (status.startsWith('processing:blog:publish')) return 'Publishing blog destinations';
    if (status.startsWith('processing:blog')) return 'Generating blog artifact';
    return status;
}

async function buildAuthHeaders(includeJson = false): Promise<HeadersInit> {
    const headers: Record<string, string> = {};
    if (includeJson) headers['content-type'] = 'application/json';

    const { data } = await supabase.auth.getSession();
    const token = data.session?.access_token;
    if (token) headers.Authorization = `Bearer ${token}`;

    return headers;
}

function parseFilenameFromDisposition(disposition: string | null): string | null {
    if (!disposition) return null;
    const match = disposition.match(/filename="([^"]+)"/i) || disposition.match(/filename=([^;]+)/i);
    if (!match?.[1]) return null;
    return match[1].trim();
}

export default function JobDetails({ params: paramsPromise }: { params: Promise<{ id: string }> }) {
    const params = use(paramsPromise);
    const [job, setJob] = useState<Job | null>(null);
    const [clips, setClips] = useState<Clip[]>([]);
    const [loading, setLoading] = useState(true);
    const [apiError, setApiError] = useState('');
    const [downloading, setDownloading] = useState(false);

    const fetchJobAndClips = useCallback(async (silent = false) => {
        if (!silent) setLoading(true);
        try {
            const headers = await buildAuthHeaders();
            const response = await fetch(`/api/v1/jobs/${params.id}`, {
                headers,
                cache: 'no-store'
            });

            if (response.status === 401) {
                setApiError('Authentication required. Sign in to view this job.');
                setJob(null);
                setClips([]);
                setLoading(false);
                return;
            }

            if (!response.ok) {
                const body = await response.json().catch(() => ({ error: 'Failed to fetch job.' }));
                setApiError(body.error || 'Failed to fetch job.');
                setJob(null);
                setClips([]);
                setLoading(false);
                return;
            }

            const data = (await response.json()) as JobDetailResponse;
            setJob(data.job);
            setClips(data.clips || []);
            setApiError('');
        } catch (error) {
            setApiError(error instanceof Error ? error.message : 'Failed to fetch job.');
            setJob(null);
            setClips([]);
        } finally {
            setLoading(false);
        }
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

    const downloadFullRes = useCallback(async () => {
        if (!job || downloading) return;
        setDownloading(true);

        try {
            const headers = await buildAuthHeaders();
            const response = await fetch(`/api/jobs/${job.id}/download`, { headers });
            if (!response.ok) {
                const text = await response.text();
                throw new Error(text || 'Download failed.');
            }

            const blob = await response.blob();
            const objectUrl = URL.createObjectURL(blob);
            const filename =
                parseFilenameFromDisposition(response.headers.get('content-disposition')) ||
                `job-${job.id}-fullres.mp4`;

            const anchor = document.createElement('a');
            anchor.href = objectUrl;
            anchor.download = filename;
            document.body.appendChild(anchor);
            anchor.click();
            anchor.remove();
            setTimeout(() => URL.revokeObjectURL(objectUrl), 30_000);
        } catch (error) {
            const message = error instanceof Error ? error.message : 'Download failed.';
            alert(message);
        } finally {
            setDownloading(false);
        }
    }, [downloading, job]);

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
                        onClick={() => {
                            if (job.srt_url) window.open(job.srt_url);
                        }}
                    >
                        <Download size={18} />
                        Download SRT
                    </button>
                )}
                {job && (
                    <button
                        className="flex items-center justify-center gap-2 px-4 py-2 bg-gray-900 text-white rounded-lg hover:bg-black transition-colors shadow-sm w-full sm:w-auto disabled:opacity-60"
                        disabled={downloading}
                        onClick={() => void downloadFullRes()}
                    >
                        <Download size={18} />
                        {downloading ? 'Downloading...' : 'Download Full Resolution'}
                    </button>
                )}
            </header>

            {apiError && (
                <div className="mb-6 p-4 rounded-xl border border-red-200 bg-red-50 text-red-700 text-sm">
                    {apiError}
                </div>
            )}

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

                            <div className="flex flex-col sm:flex-row gap-2">
                                <Link
                                    href={`/clips/${clip.id}`}
                                    className="flex-1 flex items-center justify-center gap-2 py-2 bg-blue-50 text-blue-600 rounded-xl font-medium hover:bg-blue-100 transition-colors"
                                >
                                    <Scissors size={18} />
                                    Trim
                                </Link>
                                <button className="flex-1 flex items-center justify-center gap-2 py-2 bg-green-600 text-white rounded-xl font-medium hover:bg-green-700 transition-colors">
                                    <CheckCircle size={18} />
                                    Approve
                                </button>
                                <button className="flex-1 flex items-center justify-center gap-2 py-2 bg-red-50 text-red-600 rounded-xl font-medium hover:bg-red-100 transition-colors">
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
