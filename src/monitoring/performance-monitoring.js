const sqlite3 = require('sqlite3').verbose();
const EventEmitter = require('events');
const os = require('os');
const { v4: uuidv4 } = require('uuid');

/**
 * Performance Monitoring and Alerting System for MCP Server
 * Tracks performance metrics, identifies bottlenecks, and sends alerts
 */
class PerformanceMonitor extends EventEmitter {
  constructor(config = {}) {
    super();
    this.config = {
      dbPath: config.dbPath || '/Users/tbwa/Documents/GitHub/mcp-sqlite-server/data/database.sqlite',
      samplingInterval: config.samplingInterval || 60000, // 1 minute
      alertThresholds: config.alertThresholds || {
        cpu_usage: 80,
        memory_usage: 85,
        response_time: 5000,
        error_rate: 5,
        queue_size: 100
      },
      retentionDays: config.retentionDays || 30
    };
    
    this.db = null;
    this.metrics = new Map();
    this.alerts = new Map();
    this.samplingTimer = null;
  }

  /**
   * Initialize monitoring system
   */
  async initialize() {
    this.db = new sqlite3.Database(this.config.dbPath);
    await this.createMonitoringTables();
    this.startMonitoring();
    console.log('✅ Performance Monitoring System initialized');
  }

  /**
   * Create monitoring tables
   */
  async createMonitoringTables() {
    const queries = [
      // Performance metrics table
      `CREATE TABLE IF NOT EXISTS performance_metrics (
        metric_id TEXT PRIMARY KEY,
        metric_name TEXT NOT NULL,
        metric_value REAL NOT NULL,
        metric_type TEXT NOT NULL,
        component TEXT,
        metadata TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
      )`,
      
      // System health snapshots
      `CREATE TABLE IF NOT EXISTS system_health (
        snapshot_id TEXT PRIMARY KEY,
        cpu_usage REAL,
        memory_usage REAL,
        memory_total INTEGER,
        disk_usage REAL,
        active_connections INTEGER,
        queue_size INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
      )`,
      
      // Analysis performance tracking
      `CREATE TABLE IF NOT EXISTS analysis_performance (
        performance_id TEXT PRIMARY KEY,
        analysis_id TEXT NOT NULL,
        analysis_type TEXT,
        start_time DATETIME NOT NULL,
        end_time DATETIME,
        duration_ms INTEGER,
        status TEXT,
        error_message TEXT,
        memory_used INTEGER,
        cpu_time_ms INTEGER
      )`,
      
      // API endpoint metrics
      `CREATE TABLE IF NOT EXISTS api_metrics (
        metric_id TEXT PRIMARY KEY,
        endpoint TEXT NOT NULL,
        method TEXT NOT NULL,
        response_time_ms INTEGER,
        status_code INTEGER,
        request_size INTEGER,
        response_size INTEGER,
        client_id TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
      )`,
      
      // Alert history
      `CREATE TABLE IF NOT EXISTS alert_history (
        alert_id TEXT PRIMARY KEY,
        alert_type TEXT NOT NULL,
        severity TEXT NOT NULL,
        metric_name TEXT,
        metric_value REAL,
        threshold_value REAL,
        message TEXT,
        resolved BOOLEAN DEFAULT 0,
        resolved_at DATETIME,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )`,
      
      // Performance anomalies
      `CREATE TABLE IF NOT EXISTS performance_anomalies (
        anomaly_id TEXT PRIMARY KEY,
        metric_name TEXT NOT NULL,
        expected_value REAL,
        actual_value REAL,
        deviation_percentage REAL,
        anomaly_score REAL,
        detected_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )`
    ];

    for (const query of queries) {
      await this.runQuery(query);
    }

    // Create indexes
    await this.createIndexes();
  }

  /**
   * Create performance indexes
   */
  async createIndexes() {
    const indexes = [
      'CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON performance_metrics(timestamp)',
      'CREATE INDEX IF NOT EXISTS idx_metrics_name ON performance_metrics(metric_name)',
      'CREATE INDEX IF NOT EXISTS idx_health_timestamp ON system_health(timestamp)',
      'CREATE INDEX IF NOT EXISTS idx_analysis_perf ON analysis_performance(analysis_id)',
      'CREATE INDEX IF NOT EXISTS idx_api_endpoint ON api_metrics(endpoint)',
      'CREATE INDEX IF NOT EXISTS idx_alerts_active ON alert_history(resolved)',
      'CREATE INDEX IF NOT EXISTS idx_anomalies_metric ON performance_anomalies(metric_name)'
    ];

    for (const index of indexes) {
      await this.runQuery(index);
    }
  }

  /**
   * Start monitoring
   */
  startMonitoring() {
    // System health monitoring
    this.samplingTimer = setInterval(() => {
      this.collectSystemMetrics();
    }, this.config.samplingInterval);
    
    // Real-time metric tracking
    this.setupMetricTracking();
    
    // Anomaly detection
    this.startAnomalyDetection();
    
    // Cleanup old data
    this.scheduleCleanup();
    
    console.log('🚀 Performance monitoring started');
  }

  /**
   * Collect system metrics
   */
  async collectSystemMetrics() {
    const metrics = {
      cpu_usage: this.getCPUUsage(),
      memory_usage: this.getMemoryUsage(),
      memory_total: os.totalmem(),
      disk_usage: await this.getDiskUsage(),
      active_connections: await this.getActiveConnections(),
      queue_size: await this.getQueueSize()
    };
    
    // Store snapshot
    await this.storeSystemHealth(metrics);
    
    // Check thresholds
    await this.checkThresholds(metrics);
    
    // Emit metrics event
    this.emit('metrics_collected', metrics);
  }

  /**
   * Track analysis performance
   */
  async trackAnalysisPerformance(analysisId, analysisType, performanceData) {
    const perfId = `perf_${uuidv4()}`;
    
    const query = `
      INSERT INTO analysis_performance 
      (performance_id, analysis_id, analysis_type, start_time, end_time, 
       duration_ms, status, error_message, memory_used, cpu_time_ms)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `;
    
    await this.runQuery(query, [
      perfId,
      analysisId,
      analysisType,
      performanceData.start_time,
      performanceData.end_time,
      performanceData.duration_ms,
      performanceData.status,
      performanceData.error_message,
      performanceData.memory_used,
      performanceData.cpu_time_ms
    ]);
    
    // Check for performance issues
    if (performanceData.duration_ms > this.config.alertThresholds.response_time) {
      await this.createAlert('slow_analysis', 'warning', {
        analysis_id: analysisId,
        duration: performanceData.duration_ms,
        threshold: this.config.alertThresholds.response_time
      });
    }
  }

  /**
   * Track API metrics
   */
  async trackAPIMetric(endpoint, method, metricData) {
    const metricId = `api_${uuidv4()}`;
    
    const query = `
      INSERT INTO api_metrics 
      (metric_id, endpoint, method, response_time_ms, status_code, 
       request_size, response_size, client_id)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `;
    
    await this.runQuery(query, [
      metricId,
      endpoint,
      method,
      metricData.response_time_ms,
      metricData.status_code,
      metricData.request_size,
      metricData.response_size,
      metricData.client_id
    ]);
    
    // Update endpoint statistics
    await this.updateEndpointStats(endpoint, metricData);
  }

  /**
   * Get performance dashboard data
   */
  async getDashboardData(timeRange = '1 hour') {
    const timeFilter = this.getTimeFilter(timeRange);
    
    const [systemHealth, analysisPerf, apiMetrics, activeAlerts, recentAnomalies] = 
      await Promise.all([
        this.getSystemHealthSummary(timeFilter),
        this.getAnalysisPerformanceSummary(timeFilter),
        this.getAPIMetricsSummary(timeFilter),
        this.getActiveAlerts(),
        this.getRecentAnomalies(timeFilter)
      ]);
    
    return {
      system_health: systemHealth,
      analysis_performance: analysisPerf,
      api_metrics: apiMetrics,
      active_alerts: activeAlerts,
      anomalies: recentAnomalies,
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Get system health summary
   */
  async getSystemHealthSummary(timeFilter) {
    const query = `
      SELECT 
        AVG(cpu_usage) as avg_cpu,
        MAX(cpu_usage) as max_cpu,
        AVG(memory_usage) as avg_memory,
        MAX(memory_usage) as max_memory,
        AVG(queue_size) as avg_queue_size,
        COUNT(*) as sample_count
      FROM system_health
      WHERE timestamp > datetime('now', ?)
    `;
    
    return new Promise((resolve, reject) => {
      this.db.get(query, [timeFilter], (err, row) => {
        if (err) reject(err);
        else resolve(row);
      });
    });
  }

  /**
   * Get analysis performance summary
   */
  async getAnalysisPerformanceSummary(timeFilter) {
    const query = `
      SELECT 
        analysis_type,
        COUNT(*) as total_analyses,
        AVG(duration_ms) as avg_duration,
        MAX(duration_ms) as max_duration,
        MIN(duration_ms) as min_duration,
        SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_count,
        AVG(memory_used) as avg_memory_used
      FROM analysis_performance
      WHERE start_time > datetime('now', ?)
      GROUP BY analysis_type
    `;
    
    return new Promise((resolve, reject) => {
      this.db.all(query, [timeFilter], (err, rows) => {
        if (err) reject(err);
        else resolve(rows);
      });
    });
  }

  /**
   * Get API metrics summary
   */
  async getAPIMetricsSummary(timeFilter) {
    const query = `
      SELECT 
        endpoint,
        method,
        COUNT(*) as request_count,
        AVG(response_time_ms) as avg_response_time,
        MAX(response_time_ms) as max_response_time,
        SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) as error_count,
        AVG(response_size) as avg_response_size
      FROM api_metrics
      WHERE timestamp > datetime('now', ?)
      GROUP BY endpoint, method
      ORDER BY request_count DESC
    `;
    
    return new Promise((resolve, reject) => {
      this.db.all(query, [timeFilter], (err, rows) => {
        if (err) reject(err);
        else resolve(rows);
      });
    });
  }

  /**
   * Create performance alert
   */
  async createAlert(alertType, severity, details) {
    const alertId = `alert_${uuidv4()}`;
    
    const alert = {
      alert_id: alertId,
      alert_type: alertType,
      severity: severity,
      metric_name: details.metric_name || alertType,
      metric_value: details.metric_value,
      threshold_value: details.threshold,
      message: this.generateAlertMessage(alertType, details)
    };
    
    const query = `
      INSERT INTO alert_history 
      (alert_id, alert_type, severity, metric_name, metric_value, 
       threshold_value, message)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `;
    
    await this.runQuery(query, [
      alert.alert_id,
      alert.alert_type,
      alert.severity,
      alert.metric_name,
      alert.metric_value,
      alert.threshold_value,
      alert.message
    ]);
    
    // Store active alert
    this.alerts.set(alertId, alert);
    
    // Emit alert event
    this.emit('alert_created', alert);
    
    // Send notifications
    await this.sendAlertNotifications(alert);
    
    return alertId;
  }

  /**
   * Anomaly detection
   */
  async detectAnomalies() {
    const metrics = await this.getRecentMetrics('10 minutes');
    
    for (const metric of metrics) {
      const baseline = await this.getMetricBaseline(metric.metric_name);
      
      if (baseline) {
        const deviation = Math.abs(metric.metric_value - baseline.avg_value) / baseline.std_dev;
        
        if (deviation > 3) { // 3 standard deviations
          await this.recordAnomaly({
            metric_name: metric.metric_name,
            expected_value: baseline.avg_value,
            actual_value: metric.metric_value,
            deviation_percentage: ((metric.metric_value - baseline.avg_value) / baseline.avg_value) * 100,
            anomaly_score: deviation
          });
        }
      }
    }
  }

  /**
   * Helper methods
   */
  
  getCPUUsage() {
    const cpus = os.cpus();
    let totalIdle = 0;
    let totalTick = 0;
    
    cpus.forEach(cpu => {
      for (const type in cpu.times) {
        totalTick += cpu.times[type];
      }
      totalIdle += cpu.times.idle;
    });
    
    return 100 - ~~(100 * totalIdle / totalTick);
  }
  
  getMemoryUsage() {
    const total = os.totalmem();
    const free = os.freemem();
    return ((total - free) / total) * 100;
  }
  
  async getDiskUsage() {
    // Simplified disk usage - would use df command or similar in production
    return Math.random() * 30 + 50; // Mock 50-80% usage
  }
  
  async getActiveConnections() {
    // Count active WebSocket connections and database connections
    return this.clients?.size || 0;
  }
  
  async getQueueSize() {
    // Get pending analysis queue size
    const query = 'SELECT COUNT(*) as count FROM notifications WHERE status = "pending"';
    
    return new Promise((resolve, reject) => {
      this.db.get(query, (err, row) => {
        if (err) reject(err);
        else resolve(row.count);
      });
    });
  }
  
  async checkThresholds(metrics) {
    const thresholds = this.config.alertThresholds;
    
    // CPU usage
    if (metrics.cpu_usage > thresholds.cpu_usage) {
      await this.createAlert('high_cpu_usage', 'warning', {
        metric_name: 'cpu_usage',
        metric_value: metrics.cpu_usage,
        threshold: thresholds.cpu_usage
      });
    }
    
    // Memory usage
    if (metrics.memory_usage > thresholds.memory_usage) {
      await this.createAlert('high_memory_usage', 'critical', {
        metric_name: 'memory_usage',
        metric_value: metrics.memory_usage,
        threshold: thresholds.memory_usage
      });
    }
    
    // Queue size
    if (metrics.queue_size > thresholds.queue_size) {
      await this.createAlert('large_queue_size', 'warning', {
        metric_name: 'queue_size',
        metric_value: metrics.queue_size,
        threshold: thresholds.queue_size
      });
    }
  }
  
  generateAlertMessage(alertType, details) {
    const messages = {
      high_cpu_usage: `CPU usage at ${details.metric_value.toFixed(1)}% exceeds threshold of ${details.threshold}%`,
      high_memory_usage: `Memory usage at ${details.metric_value.toFixed(1)}% exceeds threshold of ${details.threshold}%`,
      large_queue_size: `Queue size of ${details.metric_value} exceeds threshold of ${details.threshold}`,
      slow_analysis: `Analysis ${details.analysis_id} took ${details.duration}ms, exceeding ${details.threshold}ms threshold`,
      high_error_rate: `Error rate at ${details.metric_value.toFixed(1)}% exceeds threshold of ${details.threshold}%`
    };
    
    return messages[alertType] || `Alert: ${alertType}`;
  }
  
  async sendAlertNotifications(alert) {
    // In production, this would send to Slack, email, PagerDuty, etc.
    console.log(`🚨 ALERT [${alert.severity.toUpperCase()}]: ${alert.message}`);
    
    // Could integrate with notification system
    if (this.notificationSystem) {
      this.notificationSystem.notify(null, 'performance_alert', alert);
    }
  }
  
  getTimeFilter(timeRange) {
    const filters = {
      '1 hour': '-1 hour',
      '6 hours': '-6 hours',
      '1 day': '-1 day',
      '1 week': '-7 days',
      '1 month': '-30 days'
    };
    
    return filters[timeRange] || '-1 hour';
  }
  
  async getActiveAlerts() {
    const query = `
      SELECT * FROM alert_history 
      WHERE resolved = 0 
      ORDER BY created_at DESC
    `;
    
    return new Promise((resolve, reject) => {
      this.db.all(query, (err, rows) => {
        if (err) reject(err);
        else resolve(rows);
      });
    });
  }
  
  async getRecentAnomalies(timeFilter) {
    const query = `
      SELECT * FROM performance_anomalies 
      WHERE detected_at > datetime('now', ?)
      ORDER BY anomaly_score DESC
      LIMIT 10
    `;
    
    return new Promise((resolve, reject) => {
      this.db.all(query, [timeFilter], (err, rows) => {
        if (err) reject(err);
        else resolve(rows);
      });
    });
  }
  
  async getRecentMetrics(timeRange) {
    const query = `
      SELECT * FROM performance_metrics 
      WHERE timestamp > datetime('now', ?)
    `;
    
    return new Promise((resolve, reject) => {
      this.db.all(query, [this.getTimeFilter(timeRange)], (err, rows) => {
        if (err) reject(err);
        else resolve(rows);
      });
    });
  }
  
  async getMetricBaseline(metricName) {
    // Calculate baseline from historical data
    const query = `
      SELECT 
        AVG(metric_value) as avg_value,
        STDEV(metric_value) as std_dev,
        COUNT(*) as sample_count
      FROM performance_metrics
      WHERE metric_name = ?
      AND timestamp > datetime('now', '-7 days')
      AND timestamp < datetime('now', '-1 hour')
    `;
    
    return new Promise((resolve, reject) => {
      this.db.get(query, [metricName], (err, row) => {
        if (err) reject(err);
        else if (row.sample_count > 10) resolve(row);
        else resolve(null);
      });
    });
  }
  
  async recordAnomaly(anomaly) {
    const query = `
      INSERT INTO performance_anomalies 
      (anomaly_id, metric_name, expected_value, actual_value, 
       deviation_percentage, anomaly_score)
      VALUES (?, ?, ?, ?, ?, ?)
    `;
    
    await this.runQuery(query, [
      `anomaly_${uuidv4()}`,
      anomaly.metric_name,
      anomaly.expected_value,
      anomaly.actual_value,
      anomaly.deviation_percentage,
      anomaly.anomaly_score
    ]);
    
    // Create alert for significant anomalies
    if (anomaly.anomaly_score > 4) {
      await this.createAlert('performance_anomaly', 'warning', {
        metric_name: anomaly.metric_name,
        metric_value: anomaly.actual_value,
        expected_value: anomaly.expected_value,
        deviation: anomaly.deviation_percentage
      });
    }
  }
  
  async storeSystemHealth(metrics) {
    const query = `
      INSERT INTO system_health 
      (snapshot_id, cpu_usage, memory_usage, memory_total, 
       disk_usage, active_connections, queue_size)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `;
    
    await this.runQuery(query, [
      `health_${uuidv4()}`,
      metrics.cpu_usage,
      metrics.memory_usage,
      metrics.memory_total,
      metrics.disk_usage,
      metrics.active_connections,
      metrics.queue_size
    ]);
  }
  
  async updateEndpointStats(endpoint, metricData) {
    // Update running statistics for endpoint
    const stats = this.metrics.get(endpoint) || {
      total_requests: 0,
      total_errors: 0,
      total_response_time: 0
    };
    
    stats.total_requests++;
    stats.total_response_time += metricData.response_time_ms;
    if (metricData.status_code >= 400) {
      stats.total_errors++;
    }
    
    this.metrics.set(endpoint, stats);
    
    // Check error rate
    const errorRate = (stats.total_errors / stats.total_requests) * 100;
    if (errorRate > this.config.alertThresholds.error_rate && stats.total_requests > 10) {
      await this.createAlert('high_error_rate', 'critical', {
        endpoint,
        metric_value: errorRate,
        threshold: this.config.alertThresholds.error_rate
      });
    }
  }
  
  startAnomalyDetection() {
    // Run anomaly detection every 5 minutes
    setInterval(() => {
      this.detectAnomalies().catch(console.error);
    }, 300000);
  }
  
  scheduleCleanup() {
    // Clean up old data daily
    setInterval(() => {
      this.cleanupOldData().catch(console.error);
    }, 86400000); // 24 hours
  }
  
  async cleanupOldData() {
    const cutoffDate = `-${this.config.retentionDays} days`;
    
    const tables = [
      'performance_metrics',
      'system_health',
      'analysis_performance',
      'api_metrics',
      'alert_history',
      'performance_anomalies'
    ];
    
    for (const table of tables) {
      const query = `
        DELETE FROM ${table} 
        WHERE timestamp < datetime('now', ?)
        OR created_at < datetime('now', ?)
      `;
      
      await this.runQuery(query, [cutoffDate, cutoffDate]);
    }
    
    console.log('✅ Cleaned up old monitoring data');
  }
  
  runQuery(query, params = []) {
    return new Promise((resolve, reject) => {
      this.db.run(query, params, function(err) {
        if (err) reject(err);
        else resolve({ lastID: this.lastID, changes: this.changes });
      });
    });
  }
  
  /**
   * Shutdown monitoring
   */
  shutdown() {
    if (this.samplingTimer) {
      clearInterval(this.samplingTimer);
    }
    
    if (this.db) {
      this.db.close();
    }
    
    console.log('🛑 Performance monitoring stopped');
  }
}

/**
 * Performance tracking middleware for Express
 */
function performanceMiddleware(monitor) {
  return async (req, res, next) => {
    const start = Date.now();
    const startMemory = process.memoryUsage().heapUsed;
    
    // Capture response
    const originalSend = res.send;
    res.send = function(data) {
      res.locals.responseSize = Buffer.byteLength(data);
      originalSend.call(this, data);
    };
    
    // Track on response finish
    res.on('finish', async () => {
      const duration = Date.now() - start;
      const memoryUsed = process.memoryUsage().heapUsed - startMemory;
      
      await monitor.trackAPIMetric(req.path, req.method, {
        response_time_ms: duration,
        status_code: res.statusCode,
        request_size: req.get('content-length') || 0,
        response_size: res.locals.responseSize || 0,
        client_id: req.get('x-client-id') || 'anonymous'
      });
    });
    
    next();
  };
}

// Export modules
module.exports = {
  PerformanceMonitor,
  performanceMiddleware
};

// Example usage
if (require.main === module) {
  const monitor = new PerformanceMonitor({
    samplingInterval: 5000, // 5 seconds for demo
    alertThresholds: {
      cpu_usage: 70,
      memory_usage: 80,
      response_time: 3000,
      error_rate: 5,
      queue_size: 50
    }
  });
  
  monitor.initialize().then(async () => {
    // Listen for alerts
    monitor.on('alert_created', (alert) => {
      console.log('🚨 New alert:', alert);
    });
    
    // Simulate some performance tracking
    await monitor.trackAnalysisPerformance('test_001', 'comprehensive', {
      start_time: new Date(Date.now() - 5000).toISOString(),
      end_time: new Date().toISOString(),
      duration_ms: 5000,
      status: 'success',
      error_message: null,
      memory_used: 1024 * 1024 * 50, // 50MB
      cpu_time_ms: 4500
    });
    
    // Get dashboard data
    setTimeout(async () => {
      const dashboard = await monitor.getDashboardData('1 hour');
      console.log('\n📊 Performance Dashboard:');
      console.log(JSON.stringify(dashboard, null, 2));
    }, 10000);
  });
  
  // Graceful shutdown
  process.on('SIGINT', () => {
    monitor.shutdown();
    process.exit(0);
  });
}