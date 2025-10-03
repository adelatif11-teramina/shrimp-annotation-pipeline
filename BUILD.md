# 🚀 Build & Deployment Guide

Complete guide for building, testing, and deploying the Shrimp Annotation Pipeline system.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8+ (tested with 3.11)
- **Node.js**: 16+ (for React UI)
- **Docker**: 20.10+ with Docker Compose
- **Memory**: 8GB+ RAM recommended
- **Storage**: 10GB+ free space

### API Keys (Optional)
```bash
# For LLM features
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

## 🔧 Quick Build

### Automated Setup (Recommended)
```bash
# Clone and build everything
cd shrimp-annotation-pipeline
python scripts/setup_project.py --full-build

# This script will:
# 1. Create Python virtual environment
# 2. Install all dependencies
# 3. Setup database
# 4. Build React frontend
# 5. Download required models
# 6. Run initial tests
```

### Manual Build Process

#### 1. Backend (Python)
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Verify installation
python -c "import services.api.annotation_api; print('✅ Backend OK')"
```

#### 2. Frontend (React)
```bash
cd ui/

# Install Node.js dependencies
npm install

# Build production version
npm run build

# Verify build
ls -la build/
```

#### 3. Database Setup
```bash
# Start PostgreSQL and Redis
docker-compose up -d postgres redis

# Run database migrations
python services/database/migrations.py

# Verify database
python -c "from services.database.models import Base; print('✅ Database OK')"
```

## 🧪 Testing

### Unit Tests
```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests
pytest tests/ -v --cov=services

# Run specific test modules
pytest tests/test_annotation_api.py -v
pytest tests/test_retraining_system.py -v
```

### Integration Tests
```bash
# Start all services
docker-compose up -d

# Run integration tests
pytest tests/integration/ -v

# API health check
curl http://localhost:8000/health
curl http://localhost:8000/retraining/health
```

### Frontend Tests
```bash
cd ui/

# Run React tests
npm test -- --coverage --watchAll=false

# Build test
npm run build
```

## 🐳 Docker Deployment

### Development Environment
```bash
# Start all services
docker-compose up -d

# Services will be available at:
# - API Server: http://localhost:8000
# - Label Studio: http://localhost:8080
# - PostgreSQL: localhost:5432
# - Redis: localhost:6379
```

### Production Environment
```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Deploy with production config
docker-compose -f docker-compose.prod.yml up -d

# Monitor logs
docker-compose logs -f api
```

### With Local LLM (Optional)
```bash
# Include Ollama service
docker-compose --profile with-local-llm up -d

# Download models
docker exec ollama ollama pull llama2
docker exec ollama ollama pull mistral
```

## ⚙️ Configuration

### Environment Variables
```bash
# Database
export DATABASE_URL="postgresql://annotator:password@localhost:5432/annotations"
export REDIS_URL="redis://localhost:6379"

# API Settings
export API_HOST="0.0.0.0"
export API_PORT="8000"
export DEBUG="false"

# LLM Configuration
export LLM_PROVIDER="openai"  # or "ollama"
export OPENAI_API_KEY="your-key"
export OLLAMA_BASE_URL="http://localhost:11434"

# Frontend
export REACT_APP_API_URL="http://localhost:8000"
```

### Configuration Files
```bash
# Update configuration
cp config/retraining_config.yaml.example config/retraining_config.yaml
# Edit config/retraining_config.yaml as needed

# Label Studio config
cp config/label_studio_config.xml.example config/label_studio_config.xml
```

## 🚀 Deployment Scenarios

### Scenario 1: Development Setup
```bash
# For development and testing
python scripts/setup_project.py --dev
source venv/bin/activate
python services/api/annotation_api.py  # Start API
cd ui && npm start  # Start React dev server
```

### Scenario 2: Production Docker
```bash
# For production deployment
docker-compose -f docker-compose.prod.yml up -d
```

### Scenario 3: Cloud Deployment
```bash
# Configure cloud settings
export DATABASE_URL="postgresql://user:pass@cloud-db:5432/annotations"
export REDIS_URL="redis://cloud-redis:6379"

# Build and deploy
docker build -t shrimp-annotation:latest .
docker push your-registry/shrimp-annotation:latest
```

## 🔍 Health Checks

### System Health
```bash
# API health
curl http://localhost:8000/health

# Database connectivity
curl http://localhost:8000/health/database

# Redis connectivity  
curl http://localhost:8000/health/redis

# Model retraining system
curl http://localhost:8000/retraining/health
```

### Performance Checks
```bash
# Check annotation throughput
curl http://localhost:8000/statistics/overview

# Check triage queue status
curl http://localhost:8000/triage/statistics

# Monitor retraining status
python scripts/manage_retraining.py status
```

## 🐛 Troubleshooting

### Common Issues

#### Backend Issues
```bash
# Module not found errors
source venv/bin/activate  # Ensure venv is activated
pip install -r requirements.txt

# Database connection errors
docker-compose up -d postgres
export DATABASE_URL="postgresql://annotator:secure_password_change_me@localhost:5432/annotations"

# spaCy model missing
python -m spacy download en_core_web_sm
```

#### Frontend Issues
```bash
# Node modules issues
cd ui/
rm -rf node_modules package-lock.json
npm install

# Build failures
npm run build -- --verbose
```

#### Docker Issues
```bash
# Port conflicts
docker-compose down
lsof -i :8000 -i :8080 -i :5432 -i :6379

# Volume issues
docker-compose down -v
docker-compose up -d
```

### Debug Mode
```bash
# Enable debug logging
export DEBUG="true"
export LOG_LEVEL="DEBUG"

# Start with verbose logging
python services/api/annotation_api.py --log-level DEBUG
```

## 📊 Performance Optimization

### Database Optimization
```bash
# Create indices for better performance
python services/database/create_indices.py

# Database connection pooling
export DB_POOL_SIZE="20"
export DB_MAX_OVERFLOW="30"
```

### Caching Configuration
```bash
# Redis caching for LLM responses
export ENABLE_LLM_CACHE="true"
export CACHE_TTL="3600"  # 1 hour
```

### Resource Limits
```bash
# For Docker deployment
docker-compose --scale api=2 up -d  # Scale API instances
```

## 📈 Monitoring

### Application Metrics
```bash
# View real-time metrics
curl http://localhost:8000/statistics/overview | jq

# Export metrics for monitoring
curl http://localhost:8000/metrics/prometheus
```

### Log Management
```bash
# View application logs
docker-compose logs -f api
docker-compose logs -f postgres
docker-compose logs -f redis

# Log rotation setup
# Configure logrotate for production
```

## 🔄 Continuous Integration

### CI Pipeline Example
```yaml
# .github/workflows/build.yml
name: Build and Test
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - uses: actions/setup-node@v2
        with:
          node-version: '18'
      - run: python scripts/setup_project.py --ci
      - run: pytest tests/ --cov=services
      - run: cd ui && npm test -- --coverage --watchAll=false
```

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] All tests passing
- [ ] Environment variables configured
- [ ] Database migrations applied
- [ ] Frontend built successfully
- [ ] Health checks passing
- [ ] Performance benchmarks met

### Post-Deployment
- [ ] Service health verified
- [ ] Database connectivity confirmed
- [ ] API endpoints responding
- [ ] Frontend loading correctly
- [ ] Monitoring systems active
- [ ] Backup systems verified

## 🔐 Security Considerations

### Production Security
```bash
# Use strong passwords
export POSTGRES_PASSWORD="$(openssl rand -hex 32)"

# Configure HTTPS
# Use nginx or traefik for SSL termination

# API rate limiting
export RATE_LIMIT_REQUESTS="100"
export RATE_LIMIT_WINDOW="60"
```

### Data Security
```bash
# Encrypt sensitive data
export ENCRYPT_ANNOTATIONS="true"
export ENCRYPTION_KEY="$(openssl rand -hex 32)"

# Regular backups
python scripts/backup_database.py --schedule daily
```

## 🆘 Support

### Getting Help
- **Documentation**: Check `docs/` directory
- **Issues**: Create GitHub issue with logs
- **Configuration**: Review `config/` examples
- **API**: View `/docs` endpoint for API documentation

### Log Collection
```bash
# Collect diagnostic information
python scripts/collect_diagnostics.py --output diagnostic_info.zip
# Include this file when reporting issues
```