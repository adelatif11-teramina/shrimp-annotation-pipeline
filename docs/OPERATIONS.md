# Operations Runbook

This document provides operational procedures for managing the Shrimp Annotation Pipeline in production.

## Table of Contents

1. [Daily Operations](#daily-operations)
2. [Monitoring and Alerting](#monitoring-and-alerting)
3. [Incident Response](#incident-response)
4. [Scaling Procedures](#scaling-procedures)
5. [Maintenance Tasks](#maintenance-tasks)
6. [Security Operations](#security-operations)
7. [Troubleshooting Guide](#troubleshooting-guide)

## Daily Operations

### Morning Health Check

```bash
#!/bin/bash
# daily-health-check.sh

echo "🌅 Daily Health Check - $(date)"
echo "=================================="

# 1. Check API health
echo "📡 API Health:"
curl -s http://localhost:8000/health | jq '.status'

# 2. Check database
echo "🗄️  Database:"
docker exec shrimp-postgres pg_isready -U shrimp_user

# 3. Check disk space
echo "💾 Disk Usage:"
df -h / | tail -1

# 4. Check memory
echo "🧠 Memory Usage:"
free -h | grep Mem

# 5. Check active alerts
echo "🚨 Active Alerts:"
curl -s http://localhost:8000/alerts | jq '.alert_count'

# 6. Check queue status
echo "📋 Queue Status:"
curl -s http://localhost:8000/triage/statistics | jq '.queue_size'

echo "=================================="
echo "✅ Health check complete"
```

### Log Review

**Check for errors in the last 24 hours:**

```bash
# API errors
grep "ERROR" logs/api_$(date +%Y%m%d).log | wc -l

# Authentication failures
grep "401\|403" logs/api_$(date +%Y%m%d).log | wc -l

# Database errors
docker logs shrimp-postgres 2>&1 | grep ERROR

# Redis errors
docker logs shrimp-redis 2>&1 | grep ERROR
```

## Monitoring and Alerting

### Key Metrics to Monitor

**API Metrics:**
- Request rate (requests/minute)
- Response time (p50, p95, p99)
- Error rate (4xx, 5xx)
- Authentication success rate

**System Metrics:**
- CPU usage (%)
- Memory usage (%)
- Disk usage (%)
- Network I/O

**Application Metrics:**
- Queue size
- Processing time per annotation
- LLM API response time
- Database query time

### Alert Thresholds

```yaml
# prometheus-alerts.yml
groups:
- name: shrimp-api
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"

  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High response time detected"

  - alert: DatabaseConnectionFailure
    expr: up{job="postgres"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Database connection failure"

  - alert: HighMemoryUsage
    expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage detected"
```

### Dashboard Queries

**Grafana Queries:**

```promql
# Request rate
rate(http_requests_total[5m])

# Error rate
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])

# Response time percentiles
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Active users
count(increase(user_login_total[1h]))

# Queue growth rate
rate(triage_queue_size[5m])
```

## Incident Response

### Severity Levels

**P0 - Critical (Response: Immediate)**
- API completely down
- Database unavailable
- Data corruption detected
- Security breach

**P1 - High (Response: 30 minutes)**
- High error rates (>10%)
- Significant performance degradation
- Authentication system failure
- Critical feature unavailable

**P2 - Medium (Response: 2 hours)**
- Minor performance issues
- Non-critical feature failure
- Queue backup

**P3 - Low (Response: Next business day)**
- UI glitches
- Documentation issues
- Enhancement requests

### Response Procedures

#### P0 - API Down

```bash
# 1. Check container status
docker ps | grep shrimp

# 2. Check logs for errors
docker logs shrimp-api --tail 50

# 3. Check database connectivity
docker exec shrimp-postgres pg_isready

# 4. Check resource usage
docker stats --no-stream

# 5. Restart if necessary
docker-compose restart api

# 6. Verify recovery
curl http://localhost:8000/health
```

#### P1 - High Error Rate

```bash
# 1. Check error logs
grep "ERROR\|CRITICAL" logs/api_$(date +%Y%m%d).log | tail -20

# 2. Check circuit breaker status
curl http://localhost:8000/health | jq '.circuit_breakers'

# 3. Check external dependencies
curl -I https://api.openai.com/v1/models

# 4. Check database performance
docker exec shrimp-postgres psql -U shrimp_user -d shrimp_annotation -c "\
SELECT query, mean_exec_time, calls \
FROM pg_stat_statements \
ORDER BY mean_exec_time DESC \
LIMIT 10;"

# 5. Scale if needed (see scaling procedures)
```

#### Database Issues

```bash
# Check database status
docker exec shrimp-postgres pg_isready -U shrimp_user

# Check connections
docker exec shrimp-postgres psql -U shrimp_user -d shrimp_annotation -c "\
SELECT count(*) as connections, state \
FROM pg_stat_activity \
GROUP BY state;"

# Check slow queries
docker exec shrimp-postgres psql -U shrimp_user -d shrimp_annotation -c "\
SELECT query, mean_exec_time, calls \
FROM pg_stat_statements \
WHERE mean_exec_time > 1000 \
ORDER BY mean_exec_time DESC;"

# Restart database if necessary
docker-compose restart postgres
```

### Communication Templates

**Incident Notification:**
```
🚨 INCIDENT ALERT

Severity: P1
Component: Shrimp Annotation API
Issue: High error rate detected (15% 5xx errors)
Started: 2024-01-01 14:30 UTC
Impact: Users experiencing login failures

Investigating: @oncall-engineer
Updates: Every 15 minutes

Status Page: https://status.company.com
```

**Resolution Notification:**
```
✅ INCIDENT RESOLVED

Severity: P1
Component: Shrimp Annotation API
Resolution: Database connection pool increased from 20 to 50
Duration: 45 minutes
Root Cause: Database connection exhaustion during peak usage

Post-mortem: Will be published within 72 hours
```

## Scaling Procedures

### Horizontal Scaling

**Docker Compose Scaling:**

```bash
# Scale API instances
docker-compose -f docker-compose.production.yml up -d --scale api=3

# Verify scaling
docker-compose ps
```

**Kubernetes Scaling:**

```bash
# Scale API deployment
kubectl scale deployment shrimp-api --replicas=5 -n shrimp-annotation

# Check pod status
kubectl get pods -n shrimp-annotation

# Monitor resource usage
kubectl top pods -n shrimp-annotation
```

### Vertical Scaling

**Resource Limits:**

```yaml
# docker-compose.production.yml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
```

### Database Scaling

**Read Replicas:**

```yaml
# docker-compose.production.yml
  postgres-replica:
    image: postgres:14-alpine
    environment:
      PGUSER: replicator
      POSTGRES_PASSWORD: ${REPLICA_PASSWORD}
      POSTGRES_MASTER_SERVICE: postgres
    command: |
      bash -c "
      pg_basebackup -h postgres -D /var/lib/postgresql/data -U replicator -v -P -W
      echo 'standby_mode = on' >> /var/lib/postgresql/data/recovery.conf
      echo 'primary_conninfo = host=postgres port=5432 user=replicator' >> /var/lib/postgresql/data/recovery.conf
      postgres
      "
```

### Auto-scaling Configuration

**Kubernetes HPA:**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: shrimp-api-hpa
  namespace: shrimp-annotation
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: shrimp-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Maintenance Tasks

### Weekly Maintenance

```bash
#!/bin/bash
# weekly-maintenance.sh

echo "🔧 Weekly Maintenance - $(date)"

# 1. Update system packages
sudo apt update && sudo apt upgrade -y

# 2. Clean up old Docker images
docker image prune -a -f

# 3. Clean up old logs
find logs/ -name "*.log" -mtime +30 -delete

# 4. Database maintenance
docker exec shrimp-postgres psql -U shrimp_user -d shrimp_annotation -c "VACUUM ANALYZE;"

# 5. Check backup integrity
python scripts/verify-backups.py

# 6. Update SSL certificates if needed
certbot renew --quiet

# 7. Security scan
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  -v /tmp:/tmp aquasec/trivy image shrimp-api:latest
```

### Monthly Maintenance

```bash
#!/bin/bash
# monthly-maintenance.sh

# 1. Dependency updates
pip list --outdated
npm audit

# 2. Database optimization
docker exec shrimp-postgres psql -U shrimp_user -d shrimp_annotation -c "REINDEX DATABASE shrimp_annotation;"

# 3. Performance review
python scripts/performance-report.py

# 4. Capacity planning
python scripts/capacity-report.py

# 5. Security audit
python scripts/security-audit.py
```

### Database Maintenance

```sql
-- Weekly database maintenance queries

-- Check database size
SELECT pg_size_pretty(pg_database_size('shrimp_annotation'));

-- Check table sizes
SELECT schemaname,tablename,pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Check index usage
SELECT schemaname, tablename, attname, n_distinct, correlation
FROM pg_stats
WHERE schemaname = 'public';

-- Update statistics
ANALYZE;

-- Vacuum old data
VACUUM (VERBOSE, ANALYZE);
```

## Security Operations

### Security Monitoring

**Daily Security Checks:**

```bash
# Check failed login attempts
grep "401" logs/api_$(date +%Y%m%d).log | wc -l

# Check suspicious API calls
grep -E "(DROP|DELETE|INSERT|UPDATE).*--" logs/api_$(date +%Y%m%d).log

# Check rate limiting violations
grep "rate limit exceeded" logs/api_$(date +%Y%m%d).log

# Check file integrity
find /opt/shrimp-annotation-pipeline -type f -name "*.py" -exec sha256sum {} \; > /tmp/current_hashes
diff /opt/backups/file_hashes.txt /tmp/current_hashes
```

### Access Management

**Add New User:**

```bash
# Create API user
curl -X POST http://localhost:8000/auth/users \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "new_user",
    "email": "user@company.com",
    "password": "secure_password",
    "role": "annotator"
  }'
```

**Revoke Access:**

```bash
# Disable user
curl -X PATCH http://localhost:8000/auth/users/123 \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"is_active": false}'
```

### Certificate Management

```bash
# Check certificate expiry
openssl x509 -in /etc/ssl/certs/api.crt -text -noout | grep "Not After"

# Renew certificates
certbot renew --force-renewal

# Update Docker secrets (Kubernetes)
kubectl create secret tls api-tls-cert \
  --cert=/etc/letsencrypt/live/api.company.com/fullchain.pem \
  --key=/etc/letsencrypt/live/api.company.com/privkey.pem \
  -n shrimp-annotation --dry-run=client -o yaml | kubectl apply -f -
```

## Troubleshooting Guide

### Common Issues and Solutions

#### High Memory Usage

**Symptoms:**
- OOM kills in logs
- Slow response times
- Container restarts

**Investigation:**
```bash
# Check memory usage by process
docker exec shrimp-api ps aux --sort=-%mem | head

# Check Python memory profiling
docker exec shrimp-api python -c "
import psutil
process = psutil.Process()
print(f'Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB')
"

# Check garbage collection
docker exec shrimp-api python -c "
import gc
print(f'Objects: {len(gc.get_objects())}')
gc.collect()
"
```

**Solutions:**
```bash
# Increase memory limits
# In docker-compose.yml:
# mem_limit: 4g

# Reduce worker processes
# API_WORKERS=2

# Add memory monitoring
# python scripts/add-memory-alerts.py
```

#### Database Connection Exhaustion

**Symptoms:**
- "too many connections" errors
- Connection timeouts
- Slow queries

**Investigation:**
```bash
# Check active connections
docker exec shrimp-postgres psql -U shrimp_user -c "
SELECT count(*), state 
FROM pg_stat_activity 
GROUP BY state;
"

# Check connection pool settings
grep -r "pool_size\|max_overflow" .
```

**Solutions:**
```bash
# Increase connection pool
# In settings.py:
# DB_POOL_SIZE=20
# DB_MAX_OVERFLOW=30

# Increase PostgreSQL max_connections
# In postgresql.conf:
# max_connections = 200
```

#### Queue Backup

**Symptoms:**
- Increasing queue size
- Processing delays
- User complaints about slow annotation

**Investigation:**
```bash
# Check queue statistics
curl http://localhost:8000/triage/statistics

# Check processing times
grep "candidate_generation" logs/api_$(date +%Y%m%d).log | grep "duration_ms"

# Check LLM API status
curl -I https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"
```

**Solutions:**
```bash
# Scale API instances
docker-compose up -d --scale api=3

# Clear stuck items
python scripts/clear-stuck-queue-items.py

# Increase batch sizes
# In candidate generator settings:
# BATCH_SIZE=20
```

### Performance Debugging

**Query Performance:**

```sql
-- Enable query logging
ALTER SYSTEM SET log_statement = 'all';
ALTER SYSTEM SET log_min_duration_statement = 1000; -- Log queries > 1s

-- Find slow queries
SELECT query, mean_exec_time, calls, total_exec_time
FROM pg_stat_statements
WHERE mean_exec_time > 100
ORDER BY mean_exec_time DESC
LIMIT 10;

-- Check for missing indexes
SELECT schemaname, tablename, attname, n_distinct, correlation
FROM pg_stats
WHERE schemaname = 'public'
AND n_distinct > 100
AND correlation < 0.1;
```

**API Performance:**

```bash
# Request profiling
python -m cProfile -o profile.stats scripts/test-api-performance.py

# Memory profiling
pip install memory-profiler
python -m memory_profiler scripts/test-memory-usage.py

# Load testing
ab -n 1000 -c 10 http://localhost:8000/health
```

### Log Analysis

**Error Pattern Analysis:**

```bash
# Most common errors
grep "ERROR" logs/api_$(date +%Y%m%d).log | \
awk '{print $NF}' | sort | uniq -c | sort -nr | head -10

# Error timeline
grep "ERROR" logs/api_$(date +%Y%m%d).log | \
awk '{print $2}' | cut -d: -f1-2 | sort | uniq -c

# User error patterns
grep "401\|403" logs/api_$(date +%Y%m%d).log | \
awk '{print $8}' | sort | uniq -c | sort -nr
```

### Emergency Procedures

#### Complete System Recovery

```bash
#!/bin/bash
# emergency-recovery.sh

echo "🚨 Emergency Recovery Procedure"

# 1. Stop all services
docker-compose down

# 2. Backup current state
cp -r data data.backup.$(date +%Y%m%d_%H%M%S)

# 3. Restore from latest backup
python scripts/restore-from-backup.py --latest

# 4. Start core services first
docker-compose up -d postgres redis

# 5. Wait for database
sleep 30

# 6. Run database migrations
python scripts/manage_db.py upgrade

# 7. Start API
docker-compose up -d api

# 8. Verify health
curl http://localhost:8000/health

echo "✅ Recovery complete"
```

#### Data Corruption Recovery

```bash
# 1. Stop application
docker-compose stop api

# 2. Backup corrupted database
pg_dump -h localhost -U shrimp_user shrimp_annotation > corrupted_db.sql

# 3. Restore from clean backup
psql -h localhost -U shrimp_user -d shrimp_annotation < latest_clean_backup.sql

# 4. Verify data integrity
python scripts/verify-data-integrity.py

# 5. Restart application
docker-compose start api
```

---

**Document Version**: 1.0  
**Last Updated**: 2024-01-01  
**Review Schedule**: Monthly