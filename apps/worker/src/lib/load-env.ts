import dotenv from 'dotenv';
import fs from 'fs';
import path from 'path';

let loaded = false;

export function loadWorkerEnv(): void {
    if (loaded) return;
    loaded = true;

    const candidates = [
        // Preferred worker-local env
        path.resolve(__dirname, '../../.env.local'),
        path.resolve(process.cwd(), 'apps/worker/.env.local'),
        // Optional worker-scoped .env
        path.resolve(__dirname, '../../.env'),
        path.resolve(process.cwd(), 'apps/worker/.env'),
    ];

    for (const filePath of candidates) {
        if (!fs.existsSync(filePath)) continue;
        dotenv.config({ path: filePath, override: false });
    }
}
