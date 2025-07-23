import rateLimit from 'express-rate-limit';
import helmet from 'helmet';
import { Request, Response } from 'express';
import * as redis from 'redis';

// Redis client for distributed rate limiting
const redisClient = redis.createClient({
  url: process.env.REDIS_URL || 'redis://localhost:6379',
});

redisClient.on('error', (err) => console.error('Redis Client Error', err));
redisClient.connect();

// Custom Redis store for rate limiting
class RedisStore {
  client: any;
  prefix: string;

  constructor(client: any, prefix = 'rl:') {
    this.client = client;
    this.prefix = prefix;
  }

  async increment(key: string): Promise<{ totalHits: number; resetTime: Date | undefined }> {
    const redisKey = `${this.prefix}${key}`;
    const ttl = 60 * 15; // 15 minutes
    
    const multi = this.client.multi();
    multi.incr(redisKey);
    multi.expire(redisKey, ttl);
    
    const results = await multi.exec();
    const totalHits = results[0][1];
    
    return {
      totalHits,
      resetTime: new Date(Date.now() + ttl * 1000),
    };
  }

  async decrement(key: string): Promise<void> {
    await this.client.decr(`${this.prefix}${key}`);
  }

  async resetKey(key: string): Promise<void> {
    await this.client.del(`${this.prefix}${key}`);
  }
}

// Rate limiting configurations
export const createRateLimiter = (options: {
  windowMs?: number;
  max?: number;
  message?: string;
  keyGenerator?: (req: Request) => string;
}) => {
  return rateLimit({
    windowMs: options.windowMs || 15 * 60 * 1000, // 15 minutes
    max: options.max || 100, // limit each IP to 100 requests per windowMs
    message: options.message || 'Too many requests, please try again later.',
    standardHeaders: true,
    legacyHeaders: false,
    store: new RedisStore(redisClient) as any,
    keyGenerator: options.keyGenerator || ((req) => req.ip || 'unknown'),
    handler: (req: Request, res: Response) => {
      res.status(429).json({
        error: 'Too many requests',
        message: options.message || 'Please try again later',
        retryAfter: res.getHeader('Retry-After'),
      });
    },
  });
};

// Different rate limiters for different endpoints
export const generalLimiter = createRateLimiter({
  windowMs: 15 * 60 * 1000,
  max: 100,
});

export const authLimiter = createRateLimiter({
  windowMs: 15 * 60 * 1000,
  max: 5,
  message: 'Too many authentication attempts',
});

export const apiLimiter = createRateLimiter({
  windowMs: 1 * 60 * 1000,
  max: 30,
  message: 'API rate limit exceeded',
});

export const agentLimiter = createRateLimiter({
  windowMs: 1 * 60 * 1000,
  max: 10,
  message: 'Agent creation rate limit exceeded',
  keyGenerator: (req) => req.user?.id || req.ip || 'unknown',
});

// Security headers middleware
export const securityHeaders = helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      scriptSrc: ["'self'", "'unsafe-inline'", "'unsafe-eval'"],
      imgSrc: ["'self'", "data:", "https:"],
      connectSrc: ["'self'", "https://api.supabase.co", "wss://"],
      fontSrc: ["'self'", "https:", "data:"],
      objectSrc: ["'none'"],
      mediaSrc: ["'self'"],
      frameSrc: ["'none'"],
    },
  },
  crossOriginEmbedderPolicy: false,
});

// API key rate limiting with higher limits
export const apiKeyLimiter = createRateLimiter({
  windowMs: 1 * 60 * 1000,
  max: 1000,
  keyGenerator: (req) => req.headers['x-api-key'] as string || 'unknown',
  message: 'API key rate limit exceeded',
});

// Request size limiting
export const requestSizeLimiter = (maxSize: string = '10mb') => {
  return (req: Request, res: Response, next: Function) => {
    const contentLength = req.headers['content-length'];
    if (contentLength) {
      const bytes = parseInt(contentLength);
      const maxBytes = parseSize(maxSize);
      if (bytes > maxBytes) {
        return res.status(413).json({
          error: 'Payload too large',
          message: `Request body must not exceed ${maxSize}`,
        });
      }
    }
    next();
  };
};

// Helper function to parse size strings
function parseSize(size: string): number {
  const units: Record<string, number> = {
    'b': 1,
    'kb': 1024,
    'mb': 1024 * 1024,
    'gb': 1024 * 1024 * 1024,
  };
  
  const match = size.toLowerCase().match(/^(\d+)([a-z]+)$/);
  if (!match) throw new Error('Invalid size format');
  
  const [, value, unit] = match;
  return parseInt(value) * (units[unit] || 1);
}

// IP whitelist/blacklist middleware
const blacklistedIPs = new Set<string>();
const whitelistedIPs = new Set<string>();

export const ipFilter = (req: Request, res: Response, next: Function) => {
  const clientIP = req.ip || 'unknown';
  
  if (blacklistedIPs.has(clientIP)) {
    return res.status(403).json({ error: 'Access denied' });
  }
  
  if (whitelistedIPs.size > 0 && !whitelistedIPs.has(clientIP)) {
    return res.status(403).json({ error: 'Access restricted' });
  }
  
  next();
};

// Export functions to manage IP lists
export const blacklistIP = (ip: string) => blacklistedIPs.add(ip);
export const whitelistIP = (ip: string) => whitelistedIPs.add(ip);
export const removeFromBlacklist = (ip: string) => blacklistedIPs.delete(ip);
export const removeFromWhitelist = (ip: string) => whitelistedIPs.delete(ip);