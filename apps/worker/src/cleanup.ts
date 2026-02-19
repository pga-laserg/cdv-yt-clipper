import { supabase } from './lib/supabase';

async function cleanup() {
    console.log('Cleaning up database...');

    const { error: clipsError } = await supabase.from('clips').delete().neq('id', '00000000-0000-0000-0000-000000000000');
    if (clipsError) console.error('Error deleting clips:', clipsError);
    else console.log('Clips deleted.');

    const { error: jobsError } = await supabase.from('jobs').delete().neq('id', '00000000-0000-0000-0000-000000000000');
    if (jobsError) console.error('Error deleting jobs:', jobsError);
    else console.log('Jobs deleted.');

    process.exit(0);
}

cleanup();
