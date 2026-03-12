import { supabase } from '../lib/supabase';
import fs from 'fs';
import path from 'path';

export class UploadFailedError extends Error {
    code: 'OBJECT_TOO_LARGE' | 'UPLOAD_FAILED';

    constructor(message: string, code: 'OBJECT_TOO_LARGE' | 'UPLOAD_FAILED') {
        super(message);
        this.code = code;
        this.name = 'UploadFailedError';
    }
}

function resolveContentType(filePath: string): string {
    const ext = path.extname(filePath).toLowerCase();
    return ext === '.mp4'
        ? 'video/mp4'
        : ext === '.jpg' || ext === '.jpeg'
        ? 'image/jpeg'
        : ext === '.png'
        ? 'image/png'
        : ext === '.srt'
        ? 'text/plain'
        : ext === '.json'
        ? 'application/json'
        : 'application/octet-stream';
}

function encodeStoragePath(pathname: string): string {
    return pathname
        .split('/')
        .filter((segment) => segment.length > 0)
        .map((segment) => encodeURIComponent(segment))
        .join('/');
}

async function uploadViaRestStream(filePath: string, bucket: string, destination: string, contentType: string): Promise<void> {
    const supabaseUrl = String(process.env.SUPABASE_URL || '').trim();
    const serviceRoleKey = String(process.env.SUPABASE_SERVICE_ROLE_KEY || '').trim();
    if (!supabaseUrl || !serviceRoleKey) {
        throw new Error('SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY are required for streamed storage uploads');
    }

    const encodedBucket = encodeURIComponent(bucket);
    const encodedDest = encodeStoragePath(destination);
    const endpoint = `${supabaseUrl.replace(/\/$/, '')}/storage/v1/object/${encodedBucket}/${encodedDest}`;

    const stat = await fs.promises.stat(filePath);
    const fileStream = fs.createReadStream(filePath);

    const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
            apikey: serviceRoleKey,
            Authorization: `Bearer ${serviceRoleKey}`,
            'content-type': contentType,
            'content-length': String(stat.size),
            'x-upsert': 'true'
        },
        body: fileStream as unknown as BodyInit,
        // Required by Node fetch when request body is a stream.
        duplex: 'half'
    } as RequestInit & { duplex: 'half' });

    if (!response.ok) {
        const text = await response.text().catch(() => '');
        throw new Error(text || `Storage REST upload failed (${response.status})`);
    }
}

export async function uploadFile(filePath: string, bucket: string, destination: string): Promise<string> {
    const contentType = resolveContentType(filePath);
    try {
        await uploadViaRestStream(filePath, bucket, destination, contentType);
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        const isTooLarge =
            message.toLowerCase().includes('maximum allowed size') ||
            message.toLowerCase().includes('too large') ||
            message.toLowerCase().includes('entity too large');
        if (isTooLarge) {
            throw new UploadFailedError(`Upload failed (object too large): ${message}`, 'OBJECT_TOO_LARGE');
        }
        throw new UploadFailedError(`Upload failed: ${message}`, 'UPLOAD_FAILED');
    }

    const { data: { publicUrl } } = supabase.storage
        .from(bucket)
        .getPublicUrl(destination);

    return publicUrl;
}
