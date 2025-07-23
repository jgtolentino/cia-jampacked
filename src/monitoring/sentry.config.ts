import * as Sentry from '@sentry/node';
import { ProfilingIntegration } from '@sentry/profiling-node';
import { Application } from 'express';

export const initSentry = (app?: Application) => {
  Sentry.init({
    dsn: process.env.SENTRY_DSN || 'YOUR_SENTRY_DSN_HERE',
    environment: process.env.NODE_ENV || 'development',
    integrations: [
      // Enable HTTP calls tracing
      new Sentry.Integrations.Http({ tracing: true }),
      // Enable Express.js middleware tracing
      app && new Sentry.Integrations.Express({ app }),
      // Enable profiling
      new ProfilingIntegration(),
      // Additional integrations
      new Sentry.Integrations.Console(),
      new Sentry.Integrations.ContextLines(),
      new Sentry.Integrations.LinkedErrors(),
    ].filter(Boolean),
    
    // Performance Monitoring
    tracesSampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1.0,
    profilesSampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1.0,
    
    // Release tracking
    release: process.env.RELEASE_VERSION || 'jampacked@1.0.0',
    
    // Additional options
    attachStacktrace: true,
    autoSessionTracking: true,
    
    // Before send hook for filtering
    beforeSend(event, hint) {
      // Filter out specific errors
      if (event.exception?.values?.[0]?.type === 'NetworkError') {
        return null;
      }
      
      // Add user context if available
      if (event.request?.headers?.authorization) {
        event.user = event.user || {};
        event.user.id = 'authenticated_user';
      }
      
      // Sanitize sensitive data
      if (event.request?.data) {
        event.request.data = sanitizeData(event.request.data);
      }
      
      return event;
    },
    
    // Custom error filtering
    ignoreErrors: [
      'ResizeObserver loop limit exceeded',
      'Non-Error promise rejection captured',
      /Failed to fetch/,
    ],
  });
};

// Sanitize sensitive data from error reports
function sanitizeData(data: any): any {
  if (typeof data !== 'object' || data === null) return data;
  
  const sensitiveKeys = ['password', 'token', 'secret', 'api_key', 'apiKey', 'authorization'];
  const sanitized = { ...data };
  
  for (const key of Object.keys(sanitized)) {
    if (sensitiveKeys.some(sensitive => key.toLowerCase().includes(sensitive))) {
      sanitized[key] = '[REDACTED]';
    } else if (typeof sanitized[key] === 'object') {
      sanitized[key] = sanitizeData(sanitized[key]);
    }
  }
  
  return sanitized;
}

// Custom error classes for better tracking
export class AuthenticationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'AuthenticationError';
  }
}

export class ValidationError extends Error {
  constructor(message: string, public field?: string) {
    super(message);
    this.name = 'ValidationError';
  }
}

export class AgentError extends Error {
  constructor(message: string, public agentId?: string) {
    super(message);
    this.name = 'AgentError';
  }
}

export class DatabaseError extends Error {
  constructor(message: string, public query?: string) {
    super(message);
    this.name = 'DatabaseError';
  }
}

// Error reporting helpers
export const captureError = (error: Error, context?: any) => {
  Sentry.captureException(error, {
    contexts: {
      custom: context,
    },
  });
};

export const captureMessage = (message: string, level: Sentry.SeverityLevel = 'info') => {
  Sentry.captureMessage(message, level);
};

// Performance monitoring
export const startTransaction = (name: string, op: string) => {
  return Sentry.startTransaction({
    name,
    op,
  });
};

// User context
export const setUserContext = (user: { id: string; email?: string; role?: string }) => {
  Sentry.setUser({
    id: user.id,
    email: user.email,
    role: user.role,
  });
};

// Clear user context on logout
export const clearUserContext = () => {
  Sentry.setUser(null);
};

// Add breadcrumb for better error context
export const addBreadcrumb = (message: string, category: string, data?: any) => {
  Sentry.addBreadcrumb({
    message,
    category,
    level: 'info',
    data,
    timestamp: Date.now() / 1000,
  });
};

// Express error handler middleware
export const sentryErrorHandler = Sentry.Handlers.errorHandler({
  shouldHandleError(error) {
    // Capture all errors
    return true;
  },
});