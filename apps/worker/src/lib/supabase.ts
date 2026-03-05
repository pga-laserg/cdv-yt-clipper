import { createClient } from '@supabase/supabase-js';
import { loadWorkerEnv } from './load-env';

loadWorkerEnv();

const supabaseUrl = process.env.SUPABASE_URL || '';
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY || '';

if (!supabaseUrl || !supabaseServiceKey) {
    console.warn('Missing Supabase credentials. Database interactions will fail.');
}

export const supabase = createClient(supabaseUrl, supabaseServiceKey);
