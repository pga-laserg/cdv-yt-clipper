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

export async function uploadFile(filePath: string, bucket: string, destination: string): Promise<string> {
    const fileBuffer = fs.readFileSync(filePath);

    // Determine content type
    const ext = path.extname(filePath).toLowerCase();
    const contentType = ext === '.mp4' ? 'video/mp4' :
        ext === '.jpg' || ext === '.jpeg' ? 'image/jpeg' :
            ext === '.png' ? 'image/png' :
        ext === '.srt' ? 'text/plain' :
            ext === '.json' ? 'application/json' : 'application/octet-stream';

    const { data, error } = await supabase.storage
        .from(bucket)
        .upload(destination, fileBuffer, {
            contentType,
            upsert: true
        });

    if (error) {
        const isTooLarge = error.message.toLowerCase().includes('maximum allowed size');
        if (isTooLarge) {
            throw new UploadFailedError(`Upload failed (object too large): ${error.message}`, 'OBJECT_TOO_LARGE');
        }
        throw new UploadFailedError(`Upload failed: ${error.message}`, 'UPLOAD_FAILED');
    }

    const { data: { publicUrl } } = supabase.storage
        .from(bucket)
        .getPublicUrl(data.path);

    return publicUrl;
}
