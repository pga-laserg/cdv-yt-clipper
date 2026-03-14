import { supabase } from './lib/supabase';

async function seed() {
    console.log('Seeding demo data...');

    const { data: orgRow, error: orgError } = await supabase
        .from('organizations')
        .select('id')
        .eq('slug', 'default-org')
        .maybeSingle();
    if (orgError || !orgRow?.id) {
        console.error('Could not resolve default organization:', orgError?.message || 'missing default-org row');
        return;
    }
    const organizationId = orgRow.id as string;

    const { data: job, error: jobError } = await supabase.from('jobs').insert({
        organization_id: organizationId,
        source_url: 'https://www.youtube.com/watch?v=Jg51MCpDf0w',
        status: 'completed',
        sermon_start_seconds: 300,
        sermon_end_seconds: 4800,
        title: 'Sermon Feb 18 - Power of Faith'
    }).select().single();

    if (jobError) {
        console.error('Job seed failed:', jobError);
        return;
    }

    const clips = [
        {
            organization_id: organizationId,
            job_id: job.id,
            title: 'Finding Peace',
            start_seconds: 400,
            end_seconds: 450,
            transcript_excerpt: 'When you look at the stars, you see the hand of God guiding you through the night.',
            status: 'draft',
            confidence_score: 0.95
        },
        {
            organization_id: organizationId,
            job_id: job.id,
            title: 'Community Strength',
            start_seconds: 1200,
            end_seconds: 1255,
            transcript_excerpt: 'We are not alone in this journey. We walk together as one body in Christ.',
            status: 'draft',
            confidence_score: 0.88
        },
        {
            organization_id: organizationId,
            job_id: job.id,
            title: 'A Call to Action',
            start_seconds: 3500,
            end_seconds: 3550,
            transcript_excerpt: 'Go out today and be the light that someone needs to see.',
            status: 'draft',
            confidence_score: 0.92
        }
    ];

    const { error: clipError } = await supabase.from('clips').insert(clips);

    if (clipError) {
        console.error('Clip seed failed:', clipError);
    } else {
        console.log('Seed successful! Job ID:', job.id);
    }
}

seed();
