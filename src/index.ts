import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import dotenv from 'dotenv';
import * as Sentry from '@sentry/node';
import { ProfilingIntegration } from '@sentry/profiling-node';
import { createClient } from 'redis';

// Load environment variables
dotenv.config();

// Import middleware
import { verifyToken, requireRole, verifyApiKey } from './middleware/auth.middleware';
import { 
  securityHeaders, 
  generalLimiter, 
  authLimiter, 
  apiLimiter,
  agentLimiter 
} from './middleware/security.middleware';
import { initSentry, sentryErrorHandler } from './monitoring/sentry.config';
import { setupSwagger } from './api/openapi';

// Import performance monitoring
const performanceMonitoring = require('./monitoring/performance-monitoring.js');

// Initialize Express app
const app = express();
const PORT = process.env.PORT || 3000;

// Initialize Sentry
initSentry(app);
app.use(Sentry.Handlers.requestHandler());
app.use(Sentry.Handlers.tracingHandler());

// Security middleware
app.use(helmet());
app.use(cors({
  origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
  credentials: true,
}));

// Body parsing middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Rate limiting
app.use('/api/', generalLimiter);
app.use('/api/auth/', authLimiter);
app.use('/api/agents/', agentLimiter);

// Initialize Redis
const redisClient = createClient({
  url: process.env.REDIS_URL || 'redis://localhost:6379',
});

redisClient.on('error', (err) => console.error('Redis Client Error', err));
redisClient.connect().then(() => {
  console.log('Connected to Redis');
});

// Performance monitoring middleware
app.use(performanceMonitoring.performanceMiddleware);

// API Documentation
setupSwagger(app);

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    memory: process.memoryUsage(),
  });
});

// Metrics endpoint for Prometheus
app.get('/metrics', async (req, res) => {
  try {
    const metrics = await performanceMonitoring.getMetrics();
    res.set('Content-Type', 'text/plain');
    res.send(metrics);
  } catch (error) {
    res.status(500).send('Error collecting metrics');
  }
});

// API Routes
app.use('/api/auth', require('./api/routes/auth.routes'));
app.use('/api/agents', verifyToken, require('./api/routes/agent.routes'));
app.use('/api/analysis', verifyToken, require('./api/routes/analysis.routes'));
app.use('/api/campaigns', verifyToken, require('./api/routes/campaign.routes'));

// Admin routes (require admin role)
app.use('/api/admin', verifyToken, requireRole('admin'), require('./api/routes/admin.routes'));

// API key protected routes
app.use('/api/v1', verifyApiKey, require('./api/routes/public.routes'));

// Error handling
app.use(sentryErrorHandler);

// Global error handler
app.use((err: any, req: express.Request, res: express.Response, next: express.NextFunction) => {
  console.error('Global error handler:', err);
  
  // Send error to Sentry
  Sentry.captureException(err);
  
  const status = err.status || 500;
  const message = err.message || 'Internal Server Error';
  
  res.status(status).json({
    error: message,
    ...(process.env.NODE_ENV === 'development' && { stack: err.stack }),
  });
});

// 404 handler
app.use((req, res) => {
  res.status(404).json({ error: 'Route not found' });
});

// Start server
const server = app.listen(PORT, () => {
  console.log(`🚀 JamPacked Creative Intelligence server running on port ${PORT}`);
  console.log(`📊 Metrics available at http://localhost:${PORT}/metrics`);
  console.log(`📚 API Documentation at http://localhost:${PORT}/api-docs`);
  console.log(`🏥 Health check at http://localhost:${PORT}/health`);
});

// Graceful shutdown
process.on('SIGTERM', async () => {
  console.log('SIGTERM received, shutting down gracefully...');
  
  server.close(() => {
    console.log('HTTP server closed');
  });
  
  await redisClient.quit();
  await Sentry.close(2000);
  
  process.exit(0);
});

export default app;