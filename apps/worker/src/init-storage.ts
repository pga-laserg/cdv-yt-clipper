import { supabase } from './lib/supabase';

async function initStorage() {
    console.log('Initializing storage buckets...');

    const { data: buckets, error: listError } = await supabase.storage.listBuckets();

    if (listError) {
        console.error('Failed to list buckets:', listError);
        return;
    }

    const required = ['assets'];

    for (const b of required) {
        if (!buckets.find(bucket => bucket.id === b)) {
            console.log(`Creating bucket: ${b}`);
            const { error } = await supabase.storage.createBucket(b, {
                public: true
            });
            if (error) console.error(`Error creating ${b}:`, error);
        } else {
            console.log(`Bucket ${b} already exists.`);
        }
    }
}

initStorage();
