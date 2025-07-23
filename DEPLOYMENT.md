# JamPacked Creative Intelligence - Deployment Guide

## 🚀 Overview

JamPacked Creative Intelligence can be deployed using Docker Compose for single-server deployments or Kubernetes for scalable cloud deployments.

## 📋 Prerequisites

### Required
- Docker 20.10+ and Docker Compose 2.0+
- 16GB RAM minimum (32GB recommended)
- 100GB available disk space
- Python 3.9+ (for local development)

### Optional
- NVIDIA GPU with CUDA 11.8+ (for accelerated multimodal analysis)
- Kubernetes 1.24+ (for cloud deployment)
- Helm 3.0+ (for Kubernetes package management)

## 🐳 Docker Compose Deployment

### Quick Start

```bash
# Clone the repository
git clone https://github.com/pulser/jampacked-creative-intelligence.git
cd jampacked-creative-intelligence

# Run deployment script
./deployment/scripts/deploy.sh docker production
```

### Manual Deployment

1. **Build Images**
```bash
# Build all Docker images
docker build -f deployment/Dockerfile.core -t jampacked/core:latest .
docker build -f deployment/Dockerfile.worker -t jampacked/pattern-worker:latest .
docker build -f deployment/Dockerfile.gpu -t jampacked/multimodal-worker:latest .
```

2. **Configure Environment**
```bash
# Set environment variables
export MCP_SERVER_PATH="/path/to/mcp-sqlite-server"
export GPU_ENABLED="true"  # or "false" if no GPU
export GRAFANA_PASSWORD="your-secure-password"
```

3. **Start Services**
```bash
cd deployment
docker-compose -f docker-compose.production.yml up -d
```

4. **Initialize Database**
```bash
# Run setup script to create JamPacked tables
docker exec jampacked-core python setup_mcp_integration.py
```

### Service URLs
- **API**: http://localhost:8080
- **Grafana**: http://localhost:3000 (admin/your-password)
- **Prometheus**: http://localhost:9091
- **MCP SQLite**: localhost:3333

## ☸️ Kubernetes Deployment

### Prerequisites
- Kubernetes cluster (GKE, EKS, AKS, or self-managed)
- kubectl configured with cluster access
- NVIDIA GPU operator (for GPU nodes)

### Deployment Steps

1. **Create Namespace**
```bash
kubectl create namespace jampacked
```

2. **Apply Configurations**
```bash
cd deployment/kubernetes
kubectl apply -f jampacked-deployment.yaml
```

3. **Monitor Deployment**
```bash
# Watch deployment progress
kubectl -n jampacked get pods -w

# Check service status
kubectl -n jampacked get all
```

### Scaling

```bash
# Scale workers
kubectl -n jampacked scale deployment pattern-discovery-worker --replicas=5

# Enable autoscaling
kubectl -n jampacked autoscale deployment jampacked-core \
  --min=3 --max=10 --cpu-percent=70
```

## 🔧 Configuration

### Core Service Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `JAMPACKED_ENV` | production | Environment (production/staging/development) |
| `WORKSPACE_ROOT` | /data/jampacked | Data storage location |
| `MCP_SQLITE_PATH` | /data/mcp/database.sqlite | MCP SQLite database path |
| `ENABLE_AUTONOMOUS` | true | Enable autonomous learning features |
| `LOG_LEVEL` | INFO | Logging level (DEBUG/INFO/WARNING/ERROR) |

### Worker Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKER_TYPE` | pattern_discovery | Worker type (pattern_discovery/cultural_analysis/multimodal_analysis) |
| `QUEUE_URL` | redis://redis:6379/0 | Redis queue URL |
| `GPU_ENABLED` | false | Enable GPU acceleration |
| `BATCH_SIZE` | 32 | Processing batch size |

## 📊 Monitoring

### Metrics Collection

JamPacked exposes Prometheus metrics on port 9090:

```bash
# View raw metrics
curl http://localhost:9090/metrics
```

### Available Metrics
- `jampacked_analysis_duration_seconds` - Analysis processing time
- `jampacked_pattern_discoveries_total` - Total patterns discovered
- `jampacked_active_campaigns` - Active campaign analyses
- `jampacked_gpu_utilization_percent` - GPU usage (if enabled)

### Grafana Dashboards

Pre-configured dashboards available:
1. **System Overview** - Overall health and performance
2. **Analysis Pipeline** - Campaign analysis metrics
3. **Pattern Discovery** - Novel pattern detection stats
4. **Resource Usage** - CPU, memory, GPU utilization

## 🔒 Security

### SSL/TLS Configuration

1. **Generate Certificates**
```bash
# Self-signed (development)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout deployment/nginx/ssl/key.pem \
  -out deployment/nginx/ssl/cert.pem

# Let's Encrypt (production)
certbot certonly --standalone -d jampacked.yourdomain.com
```

2. **Update Nginx Config**
```nginx
ssl_certificate /etc/nginx/ssl/cert.pem;
ssl_certificate_key /etc/nginx/ssl/key.pem;
```

### Authentication

Configure API authentication in `config/auth.yaml`:
```yaml
auth:
  type: jwt
  secret: ${JWT_SECRET}
  expiry: 3600
```

## 🔄 Backup & Recovery

### Database Backup

```bash
# Backup MCP SQLite database
docker exec jampacked-core sqlite3 /data/mcp/database.sqlite \
  ".backup /data/backup/jampacked_$(date +%Y%m%d_%H%M%S).db"

# Backup workspace data
tar -czf jampacked_workspace_$(date +%Y%m%d).tar.gz \
  /data/jampacked/
```

### Restore

```bash
# Restore database
docker exec jampacked-core sqlite3 /data/mcp/database.sqlite \
  ".restore /data/backup/jampacked_backup.db"

# Restore workspace
tar -xzf jampacked_workspace_backup.tar.gz -C /
```

## 🚨 Troubleshooting

### Common Issues

1. **GPU Not Detected**
```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base nvidia-smi

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
```

2. **Memory Issues**
```bash
# Increase Docker memory limit
# Docker Desktop: Preferences > Resources > Memory

# Kubernetes: Update resource limits
kubectl -n jampacked edit deployment jampacked-core
```

3. **Database Connection Failed**
```bash
# Check SQLite database permissions
docker exec jampacked-core ls -la /data/mcp/

# Verify MCP server is running
docker logs mcp-integration
```

### Logs

```bash
# View core service logs
docker logs -f jampacked-core

# View worker logs
docker logs -f pattern-discovery-worker

# Kubernetes logs
kubectl -n jampacked logs -f deployment/jampacked-core
```

## 📈 Performance Tuning

### Optimization Tips

1. **Database Performance**
   - Enable WAL mode for SQLite
   - Regular VACUUM operations
   - Index optimization

2. **Worker Scaling**
   - Scale based on queue depth
   - Separate CPU and GPU workloads
   - Use node affinity for GPU workers

3. **Caching**
   - Enable Redis persistence
   - Configure Nginx caching
   - Implement model caching

## 🔄 Updates & Maintenance

### Rolling Updates

```bash
# Docker Compose
docker-compose -f docker-compose.production.yml pull
docker-compose -f docker-compose.production.yml up -d

# Kubernetes
kubectl -n jampacked set image deployment/jampacked-core \
  jampacked-core=jampacked/core:v2.0.0
```

### Health Checks

```bash
# Run deployment health check
./deployment/scripts/deploy.sh health

# Manual health check
curl http://localhost:8080/health
```

## 📞 Support

For deployment issues:
1. Check logs for error messages
2. Verify all prerequisites are met
3. Review configuration settings
4. Contact support at support@jampacked.ai

## 📚 Additional Resources

- [Architecture Overview](./docs/architecture.md)
- [API Documentation](./docs/api.md)
- [Configuration Reference](./docs/configuration.md)
- [Performance Benchmarks](./docs/benchmarks.md)