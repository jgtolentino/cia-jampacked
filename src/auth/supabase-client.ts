import { createClient } from '@supabase/supabase-js';

// Initialize Supabase client
const supabaseUrl = process.env.SUPABASE_URL || 'https://cxzllzyxwpyptfretryc.supabase.co';
const supabaseAnonKey = process.env.SUPABASE_ANON_KEY || '';

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error('Missing Supabase environment variables');
}

export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
  auth: {
    autoRefreshToken: true,
    persistSession: true,
    detectSessionInUrl: true,
  },
});

// Auth helper functions
export const signIn = async (email: string, password: string) => {
  const { data, error } = await supabase.auth.signInWithPassword({
    email,
    password,
  });
  return { data, error };
};

export const signUp = async (email: string, password: string, metadata?: any) => {
  const { data, error } = await supabase.auth.signUp({
    email,
    password,
    options: {
      data: metadata,
    },
  });
  return { data, error };
};

export const signOut = async () => {
  const { error } = await supabase.auth.signOut();
  return { error };
};

export const getSession = async () => {
  const { data, error } = await supabase.auth.getSession();
  return { data, error };
};

export const getUser = async () => {
  const { data: { user }, error } = await supabase.auth.getUser();
  return { user, error };
};

// Role-based access control
export const checkRole = async (requiredRole: string) => {
  const { user, error } = await getUser();
  if (error || !user) return false;
  
  const userRole = user.user_metadata?.role || 'user';
  const roleHierarchy: Record<string, number> = {
    'user': 1,
    'analyst': 2,
    'manager': 3,
    'admin': 4,
  };
  
  return roleHierarchy[userRole] >= roleHierarchy[requiredRole];
};

// JWT token management
export const refreshToken = async () => {
  const { data, error } = await supabase.auth.refreshSession();
  return { data, error };
};

// Session management
export const onAuthStateChange = (callback: (event: string, session: any) => void) => {
  return supabase.auth.onAuthStateChange(callback);
};