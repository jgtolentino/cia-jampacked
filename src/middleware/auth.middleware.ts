import { Request, Response, NextFunction } from 'express';
import { supabase } from '../auth/supabase-client';

export interface AuthRequest extends Request {
  user?: any;
  session?: any;
}

// Verify JWT token middleware
export const verifyToken = async (
  req: AuthRequest,
  res: Response,
  next: NextFunction
) => {
  try {
    const authHeader = req.headers.authorization;
    
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return res.status(401).json({ error: 'No token provided' });
    }
    
    const token = authHeader.substring(7);
    
    // Verify the token with Supabase
    const { data: { user }, error } = await supabase.auth.getUser(token);
    
    if (error || !user) {
      return res.status(401).json({ error: 'Invalid token' });
    }
    
    // Attach user to request
    req.user = user;
    next();
  } catch (error) {
    console.error('Auth middleware error:', error);
    res.status(500).json({ error: 'Authentication failed' });
  }
};

// Role-based access control middleware
export const requireRole = (requiredRole: string) => {
  return async (req: AuthRequest, res: Response, next: NextFunction) => {
    try {
      if (!req.user) {
        return res.status(401).json({ error: 'Authentication required' });
      }
      
      const userRole = req.user.user_metadata?.role || 'user';
      const roleHierarchy: Record<string, number> = {
        'user': 1,
        'analyst': 2,
        'manager': 3,
        'admin': 4,
      };
      
      if (roleHierarchy[userRole] < roleHierarchy[requiredRole]) {
        return res.status(403).json({ error: 'Insufficient permissions' });
      }
      
      next();
    } catch (error) {
      console.error('Role check error:', error);
      res.status(500).json({ error: 'Authorization failed' });
    }
  };
};

// API key authentication middleware
export const verifyApiKey = async (
  req: AuthRequest,
  res: Response,
  next: NextFunction
) => {
  try {
    const apiKey = req.headers['x-api-key'] as string;
    
    if (!apiKey) {
      return res.status(401).json({ error: 'API key required' });
    }
    
    // Verify API key in database
    const { data, error } = await supabase
      .from('api_keys')
      .select('*')
      .eq('key', apiKey)
      .eq('is_active', true)
      .single();
    
    if (error || !data) {
      return res.status(401).json({ error: 'Invalid API key' });
    }
    
    // Update last used timestamp
    await supabase
      .from('api_keys')
      .update({ last_used: new Date().toISOString() })
      .eq('id', data.id);
    
    req.user = { apiKey: data };
    next();
  } catch (error) {
    console.error('API key verification error:', error);
    res.status(500).json({ error: 'API key verification failed' });
  }
};

// Session refresh middleware
export const refreshSession = async (
  req: AuthRequest,
  res: Response,
  next: NextFunction
) => {
  try {
    const refreshToken = req.headers['x-refresh-token'] as string;
    
    if (refreshToken) {
      const { data, error } = await supabase.auth.refreshSession({
        refresh_token: refreshToken,
      });
      
      if (!error && data.session) {
        res.setHeader('x-access-token', data.session.access_token);
        res.setHeader('x-refresh-token', data.session.refresh_token);
        req.session = data.session;
      }
    }
    
    next();
  } catch (error) {
    console.error('Session refresh error:', error);
    next();
  }
};