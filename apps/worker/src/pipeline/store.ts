import { supabase } from '../lib/supabase';
import fs from 'fs';
import path from 'path';

export async function uploadFile(filePath: string, bucket: string, destination: string): Promise<string> {
    const fileBuffer = fs.readFileSync(filePath);

    // Determine content type
    const ext = path.extname(filePath).toLowerCase();
    const contentType = ext === '.mp4' ? 'video/mp4' :
        ext === '.srt' ? 'text/plain' :
            ext === '.json' ? 'application/json' : 'application/octet-stream';

    const { data, error } = await supabase.storage
        .from(bucket)
        .upload(destination, fileBuffer, {
            contentType,
            upsert: true
        });

    if (error) {
        throw new Error(`Upload failed: ${error.message}`);
    }

    const { data: { publicUrl } } = supabase.storage
        .from(bucket)
        .getPublicUrl(data.path);

    return publicUrl;
}
