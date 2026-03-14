'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { supabase } from '@/lib/supabase';

export default function AdminLayout({ children }: { children: React.ReactNode }) {
    const router = useRouter();
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        supabase.auth.getSession().then(({ data: { session } }) => {
            if (!session) {
                router.push('/');
            } else {
                setLoading(false);
            }
        });
    }, [router]);

    if (loading) {
        return <div className="min-h-screen flex items-center justify-center p-8 bg-gray-50 text-gray-500">Loading admin context...</div>;
    }

    return <>{children}</>;
}
