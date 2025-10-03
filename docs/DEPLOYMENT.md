# Deployment Guide

This guide covers deploying the Shrimp Annotation Pipeline in production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Database Configuration](#database-configuration)
4. [Security Configuration](#security-configuration)
5. [Docker Deployment](#docker-deployment)
6. [Kubernetes Deployment](#kubernetes-deployment)
7. [Monitoring and Logging](#monitoring-and-logging)
8. [Backup and Recovery](#backup-and-recovery)
9. [Performance Tuning](#performance-tuning)
10. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

**Minimum Production Requirements:**
- **CPU**: 4 cores, 2.4 GHz
- **RAM**: 8 GB minimum, 16 GB recommended
- **Storage**: 100 GB SSD (database + logs + caches)
- **Network**: 1 Gbps network connection

**Recommended Production Requirements:**
- **CPU**: 8 cores, 3.0 GHz
- **RAM**: 32 GB
- **Storage**: 500 GB NVMe SSD
- **Load Balancer**: For multi-instance deployments

### Software Dependencies

- **Docker**: 20.10+ and Docker Compose 2.0+
- **PostgreSQL**: 14+ (production database)
- **Redis**: 6.0+ (caching and rate limiting)
- **Python**: 3.8+ (for development/debugging)
- **Node.js**: 18+ (for frontend builds)

### External Services

- **OpenAI API**: For LLM candidate generation
- **SMTP Server**: For email notifications (optional)
- **Prometheus/Grafana**: For monitoring (recommended)
- **Sentry**: For error tracking (optional)

## Environment Setup

### 1. Clone and Prepare Repository

```bash
# Clone the repository
git clone <repository-url>
cd shrimp-annotation-pipeline

# Create required directories
mkdir -p logs data/exports data/feedback
chmod 755 logs data
```

### 2. Environment Configuration

Create production environment file:

```bash
cp .env.example .env.production
```

**Required Environment Variables:**

```bash
# Environment
ENVIRONMENT=production
DEBUG=false

# Security - MUST BE CHANGED
JWT_SECRET_KEY=your-super-secure-jwt-secret-key-256-bits
DB_PASSWORD=your-secure-database-password

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=shrimp_annotation
DB_USER=shrimp_user

# Redis
REDIS_URL=redis://localhost:6379/0

# External APIs
OPENAI_API_KEY=your-openai-api-key

# Monitoring (Optional)
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project
PROMETHEUS_ENABLED=true

# Logging
LOG_LEVEL=INFO
JSON_LOGGING=true

# CORS
CORS_ORIGINS=["https://your-frontend-domain.com"]

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_PER_HOUR=1000
```

### 3. Security Checklist

**Before deploying:**

```bash
# Generate secure JWT secret (256-bit)
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate secure database password
python -c "import secrets; print(secrets.token_urlsafe(16))"

# Verify no default passwords remain
grep -r "dev_password_change_in_production" .
grep -r "development_secret_key" .
```

## Database Configuration

### 1. PostgreSQL Production Setup

```sql
-- Create database and user
CREATE USER shrimp_user WITH PASSWORD 'your-secure-password';
CREATE DATABASE shrimp_annotation OWNER shrimp_user;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE shrimp_annotation TO shrimp_user;

-- Connect to the database
\c shrimp_annotation

-- Grant schema permissions
GRANT ALL ON SCHEMA public TO shrimp_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO shrimp_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO shrimp_user;
```

### 2. Database Initialization

```bash
# Run database migrations
python scripts/manage_db.py init
python scripts/manage_db.py upgrade

# Verify database setup
python scripts/manage_db.py check
```

### 3. Database Performance Tuning

Add to PostgreSQL configuration (`postgresql.conf`):

```sql
# Memory settings
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB

# Connection settings
max_connections = 100

# Performance settings
random_page_cost = 1.1
effective_io_concurrency = 200

# WAL settings
wal_buffers = 16MB
checkpoint_completion_target = 0.9
```

## Security Configuration

### 1. TLS/SSL Setup

**Nginx Configuration** (`/etc/nginx/sites-available/shrimp-api`):

```nginx
server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;
    
    ssl_certificate /path/to/your/cert.pem;
    ssl_certificate_key /path/to/your/private.key;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeout settings
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name api.yourdomain.com;
    return 301 https://$server_name$request_uri;
}
```

### 2. Firewall Configuration

```bash
# UFW firewall rules
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP (for redirect)
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable

# Restrict database access
sudo ufw allow from 10.0.0.0/8 to any port 5432
sudo ufw allow from 172.16.0.0/12 to any port 5432
sudo ufw allow from 192.168.0.0/16 to any port 5432
```

### 3. System Security

```bash
# Create dedicated user
sudo useradd -r -s /bin/false shrimp-api
sudo usermod -a -G docker shrimp-api

# Set file permissions
sudo chown -R shrimp-api:shrimp-api /opt/shrimp-annotation-pipeline
sudo chmod -R 750 /opt/shrimp-annotation-pipeline
sudo chmod 600 /opt/shrimp-annotation-pipeline/.env.production
```

## Docker Deployment

### 1. Production Docker Compose

Create `docker-compose.production.yml`:

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.production
    ports:
      - "8000:8000"
    environment:
      - ENV_FILE=.env.production
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./.env.production:/app/.env
    depends_on:
      - postgres
      - redis
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G

  postgres:
    image: postgres:14-alpine
    environment:
      POSTGRES_DB: shrimp_annotation
      POSTGRES_USER: shrimp_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init-db.sql:/docker-entrypoint-initdb.d/init-db.sql
    ports:
      - "5432:5432"
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 2G

  redis:
    image: redis:6-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M

  frontend:
    build:
      context: ./ui
      dockerfile: Dockerfile.production
    ports:
      - "80:80"
    depends_on:
      - api
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:

networks:
  default:
    driver: bridge
```

### 2. Production Dockerfile

Create `Dockerfile.production`:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p logs data/exports data/feedback && \
    chown -R app:app /app

# Switch to app user
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["python", "scripts/start_server.py"]
```

### 3. Deployment Commands

```bash
# Build and start services
docker-compose -f docker-compose.production.yml up -d --build

# View logs
docker-compose -f docker-compose.production.yml logs -f

# Update deployment
docker-compose -f docker-compose.production.yml pull
docker-compose -f docker-compose.production.yml up -d --remove-orphans

# Backup database
docker-compose -f docker-compose.production.yml exec postgres \
    pg_dump -U shrimp_user shrimp_annotation > backup_$(date +%Y%m%d_%H%M%S).sql
```

## Kubernetes Deployment

### 1. Namespace and ConfigMap

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: shrimp-annotation

---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: shrimp-config
  namespace: shrimp-annotation
data:
  ENVIRONMENT: "production"
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  DB_HOST: "postgres-service"
  REDIS_URL: "redis://redis-service:6379/0"
```

### 2. Secrets

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: shrimp-secrets
  namespace: shrimp-annotation
type: Opaque
data:
  JWT_SECRET_KEY: <base64-encoded-secret>
  DB_PASSWORD: <base64-encoded-password>
  OPENAI_API_KEY: <base64-encoded-api-key>
```

### 3. API Deployment

```yaml
# api-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: shrimp-api
  namespace: shrimp-annotation
spec:
  replicas: 3
  selector:
    matchLabels:
      app: shrimp-api
  template:
    metadata:
      labels:
        app: shrimp-api
    spec:
      containers:
      - name: api
        image: your-registry/shrimp-api:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: shrimp-config
        - secretRef:
            name: shrimp-secrets
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: shrimp-api-service
  namespace: shrimp-annotation
spec:
  selector:
    app: shrimp-api
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
```

### 4. Database StatefulSet

```yaml
# postgres-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: shrimp-annotation
spec:
  serviceName: postgres-service
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:14-alpine
        env:
        - name: POSTGRES_DB
          value: shrimp_annotation
        - name: POSTGRES_USER
          value: shrimp_user
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: shrimp-secrets
              key: DB_PASSWORD
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi

---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: shrimp-annotation
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
  type: ClusterIP
```

## Monitoring and Logging

### 1. Prometheus Configuration

```yaml
# prometheus-config.yaml
global:
  scrape_interval: 15s

scrape_configs:
- job_name: 'shrimp-api'
  static_configs:
  - targets: ['shrimp-api-service:8000']
  metrics_path: '/metrics/prometheus'
  scrape_interval: 30s
```

### 2. Grafana Dashboard

Key metrics to monitor:

- **API Metrics**: Request rate, response time, error rate
- **System Metrics**: CPU, memory, disk usage
- **Application Metrics**: Queue size, annotation rate, user activity
- **Database Metrics**: Connection count, query performance
- **Cache Metrics**: Redis hit rate, memory usage

### 3. Log Aggregation

**ELK Stack Configuration:**

```yaml
# filebeat.yml
filebeat.inputs:
- type: log
  paths:
    - /app/logs/*.log
  json.keys_under_root: true
  json.add_error_key: true

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  index: "shrimp-api-%{+yyyy.MM.dd}"

logging.level: info
```

## Backup and Recovery

### 1. Database Backup Strategy

```bash
#!/bin/bash
# backup-db.sh

BACKUP_DIR="/backups/database"
DATE=$(date +%Y%m%d_%H%M%S)
DB_NAME="shrimp_annotation"
DB_USER="shrimp_user"

# Create backup directory
mkdir -p $BACKUP_DIR

# Create database backup
pg_dump -h localhost -U $DB_USER -d $DB_NAME > $BACKUP_DIR/shrimp_db_$DATE.sql

# Compress backup
gzip $BACKUP_DIR/shrimp_db_$DATE.sql

# Remove backups older than 30 days
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete

# Upload to cloud storage (optional)
# aws s3 cp $BACKUP_DIR/shrimp_db_$DATE.sql.gz s3://your-backup-bucket/
```

### 2. Application Data Backup

```bash
#!/bin/bash
# backup-data.sh

BACKUP_DIR="/backups/data"
DATE=$(date +%Y%m%d_%H%M%S)
APP_DIR="/opt/shrimp-annotation-pipeline"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup critical data directories
tar -czf $BACKUP_DIR/data_$DATE.tar.gz \
    $APP_DIR/data/gold \
    $APP_DIR/data/exports \
    $APP_DIR/logs

# Remove old backups
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
```

### 3. Recovery Procedures

**Database Recovery:**

```bash
# Restore from backup
gunzip -c backup_20240101_120000.sql.gz | psql -h localhost -U shrimp_user -d shrimp_annotation

# Verify restoration
python scripts/manage_db.py check
```

## Performance Tuning

### 1. API Performance

**Uvicorn Configuration:**

```python
# In production startup
uvicorn.run(
    "services.api.annotation_api:app",
    host="0.0.0.0",
    port=8000,
    workers=4,  # CPU cores
    worker_class="uvicorn.workers.UvicornWorker",
    max_requests=1000,
    max_requests_jitter=50,
    preload=True
)
```

### 2. Database Optimization

```sql
-- Create indexes for common queries
CREATE INDEX CONCURRENTLY idx_annotations_doc_id ON annotations(doc_id);
CREATE INDEX CONCURRENTLY idx_annotations_status ON annotations(status);
CREATE INDEX CONCURRENTLY idx_annotations_created_at ON annotations(created_at);
CREATE INDEX CONCURRENTLY idx_users_username ON users(username);

-- Analyze tables for query planning
ANALYZE annotations;
ANALYZE users;
ANALYZE documents;
```

### 3. Caching Strategy

```python
# Redis caching configuration
CACHE_SETTINGS = {
    "llm_responses": {"ttl": 3600},  # 1 hour
    "user_sessions": {"ttl": 1800},  # 30 minutes
    "api_responses": {"ttl": 300},   # 5 minutes
}
```

## Troubleshooting

### 1. Common Issues

**High Memory Usage:**
```bash
# Check memory usage
docker stats

# Adjust worker settings
API_WORKERS=2  # Reduce workers
```

**Database Connection Issues:**
```bash
# Check database connectivity
pg_isready -h localhost -p 5432

# Monitor connections
SELECT count(*) FROM pg_stat_activity;
```

**API Performance Issues:**
```bash
# Check response times
curl -w "@curl-format.txt" -o /dev/null -s "http://localhost:8000/health"

# Monitor queue status
curl http://localhost:8000/triage/statistics
```

### 2. Log Analysis

**API Logs:**
```bash
# Error patterns
grep "ERROR" logs/api_*.log | tail -20

# Slow requests
grep "duration_ms.*[0-9]{4,}" logs/api_*.log

# Authentication failures
grep "401\|403" logs/api_*.log
```

### 3. Health Checks

```bash
# System health
curl http://localhost:8000/health/detailed

# Circuit breaker status
curl http://localhost:8000/health | jq '.circuit_breakers'

# Metrics summary
curl http://localhost:8000/metrics/summary
```

## Deployment Checklist

### Pre-Deployment

- [ ] All environment variables configured
- [ ] Secrets properly generated and secured
- [ ] Database initialized and migrated
- [ ] SSL certificates installed
- [ ] Firewall rules configured
- [ ] Monitoring setup completed
- [ ] Backup procedures tested

### Post-Deployment

- [ ] Health checks passing
- [ ] Monitoring alerts configured
- [ ] Performance baselines established
- [ ] Documentation updated
- [ ] Team trained on operations
- [ ] Incident response procedures reviewed

### Rollback Plan

1. Stop current deployment
2. Restore previous container version
3. Restore database backup if needed
4. Verify system functionality
5. Update monitoring and logs

## Support and Maintenance

### Regular Maintenance Tasks

**Daily:**
- Check system health and alerts
- Monitor resource usage
- Review error logs

**Weekly:**
- Update dependencies
- Analyze performance metrics
- Test backup procedures

**Monthly:**
- Security updates
- Performance optimization
- Capacity planning review

### Emergency Contacts

- **System Administrator**: [Contact Info]
- **Database Administrator**: [Contact Info]
- **Development Team**: [Contact Info]
- **Security Team**: [Contact Info]

---

**Document Version**: 1.0  
**Last Updated**: 2024-01-01  
**Next Review**: 2024-04-01