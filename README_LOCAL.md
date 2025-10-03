# Local Development Mode - Complete Setup Guide

## 🎯 Overview

This local development mode allows you to run the Shrimp Annotation Pipeline **completely offline** with:
- ✅ **Zero external dependencies** (no internet required after setup)
- ✅ **SQLite database** (no PostgreSQL needed)
- ✅ **Local LLM with Ollama** (optional, free)
- ✅ **Simple authentication** (token-based)
- ✅ **File-based queuing** (no Redis required)
- ✅ **Complete annotation workflow**

Perfect for internal teams, testing, or development!

## 🚀 Quick Start (30 seconds)

```bash
# 1. Setup everything automatically
python3 scripts/setup_local.py

# 2. Start all services
./start_local.sh

# 3. Open browser
# http://localhost:3000
```

**That's it!** 🎉

## 📋 Prerequisites

- **Python 3.8+** (Check: `python3 --version`)
- **Node.js 16+** (Check: `node --version`)
- **npm** (Check: `npm --version`)

Optional but recommended:
- **Ollama** for local LLM ([Install here](https://ollama.ai))

## 🔧 Detailed Setup

### Option 1: Automatic Setup (Recommended)

```bash
# Run the setup script
python3 scripts/setup_local.py

# This will:
# ✅ Create virtual environment
# ✅ Install all dependencies  
# ✅ Setup SQLite database
# ✅ Create sample data
# ✅ Configure everything
# ✅ Download LLM model (if Ollama available)
```

### Option 2: Manual Setup

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install Python dependencies
pip install -r requirements.txt
pip install "numpy<2" --upgrade  # Fix compatibility

# 3. Install UI dependencies
cd ui && npm install && cd ..

# 4. Initialize database
python -c "
from services.database.local_models import init_db
init_db()
"

# 5. Optional: Install Ollama for LLM
# Download from https://ollama.ai
ollama pull llama3.2:3b
```

## 🏃‍♂️ Running the System

### Method 1: Start Script (Easiest)

```bash
./start_local.sh
```

This script:
- ✅ Checks all prerequisites
- ✅ Starts API server (port 8000)
- ✅ Starts UI server (port 3000)
- ✅ Starts Ollama (if available)
- ✅ Monitors all processes
- ✅ Provides helpful output

### Method 2: Manual Start

```bash
# Terminal 1: Start API
source venv/bin/activate
python services/api/local_annotation_api.py

# Terminal 2: Start UI
cd ui && npm start

# Terminal 3: Start Ollama (optional)
ollama serve
```

### Method 3: Docker (If you prefer containers)

```bash
# Start with Docker Compose
docker-compose -f docker-compose.local.yml up -d

# Check status
docker-compose -f docker-compose.local.yml ps

# View logs
docker-compose -f docker-compose.local.yml logs -f
```

## 🌐 Access Points

Once running, access these URLs:

| Service | URL | Description |
|---------|-----|-------------|
| **Main UI** | http://localhost:3000 | Annotation workspace |
| **API Server** | http://localhost:8000 | REST API |
| **API Docs** | http://localhost:8000/docs | Interactive API documentation |

## 🔑 Authentication

Simple token-based authentication with pre-configured users:

| Role | Username | Token | Permissions |
|------|----------|-------|-------------|
| **Admin** | admin | `local-admin-2024` | Full access |
| **Annotator** | annotator1 | `anno-team-001` | Annotate, view stats |
| **Annotator** | annotator2 | `anno-team-002` | Annotate, view stats |
| **Reviewer** | reviewer | `review-lead-003` | Review, export data |

### Using Authentication

In the UI, enter the token when prompted, or use the API:

```bash
# API request with authentication
curl -H "Authorization: Bearer local-admin-2024" \
     http://localhost:8000/statistics
```

## 📊 Using the System

### 1. Upload Documents

```bash
# Via API
curl -X POST http://localhost:8000/documents \
  -H "Authorization: Bearer local-admin-2024" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Disease Report 1",
    "text": "Penaeus vannamei infected with Vibrio parahaemolyticus showed 80% mortality.",
    "source": "manual"
  }'

# Or use the UI at http://localhost:3000
```

### 2. Generate Candidates

```bash
# Generate annotations for a sentence
curl -X POST http://localhost:8000/candidates/generate \
  -H "Authorization: Bearer local-admin-2024" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "abc123",
    "sent_id": "s0", 
    "text": "Penaeus vannamei infected with Vibrio parahaemolyticus."
  }'
```

### 3. Review in Triage Queue

```bash
# Get next item to review
curl -H "Authorization: Bearer anno-team-001" \
     http://localhost:8000/triage/next

# Make annotation decision
curl -X POST http://localhost:8000/annotations/decide \
  -H "Authorization: Bearer anno-team-001" \
  -H "Content-Type: application/json" \
  -d '{
    "candidate_id": 1,
    "decision": "accept",
    "confidence": 0.9
  }'
```

### 4. Export Data

```bash
# Export annotations
curl -H "Authorization: Bearer local-admin-2024" \
     "http://localhost:8000/export?format=json" > annotations.json
```

## 📁 File Structure

Local mode creates this structure:

```
shrimp-annotation-pipeline/
├── data/local/              # All local data
│   ├── annotations.db       # SQLite database
│   ├── documents/           # Uploaded documents
│   ├── candidates/          # Generated candidates
│   ├── gold/                # Gold annotations
│   ├── exports/             # Exported data
│   ├── logs/                # Application logs
│   ├── queue/               # File-based queue
│   └── llm_cache/           # LLM response cache
├── config/
│   └── local_config.yaml   # Local configuration
├── .env.local               # Environment variables
└── start_local.sh           # Start script
```

## 🤖 LLM Integration

### With Ollama (Recommended)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Download model (done automatically by setup script)
ollama pull llama3.2:3b

# Model will be used for entity/relation extraction
```

### Rules-Only Mode

If Ollama isn't available, the system uses rule-based patterns:

- **Species**: Penaeus vannamei, white-leg shrimp
- **Pathogens**: Vibrio parahaemolyticus, WSSV
- **Diseases**: AHPND, WSD, EMS
- **Measurements**: Temperature, salinity, pH

## 📈 Monitoring & Stats

```bash
# Get system statistics
curl -H "Authorization: Bearer local-admin-2024" \
     http://localhost:8000/statistics

# Response:
{
  "documents": 5,
  "sentences": 25,
  "candidates": 50,
  "gold_annotations": 30,
  "queue_size": 5,
  "completed_today": 10
}
```

## 🔧 Troubleshooting

### Common Issues

**Port already in use:**
```bash
# Find process using port
lsof -i :8000
lsof -i :3000

# Kill process
kill -9 <PID>
```

**Database issues:**
```bash
# Reset database
rm data/local/annotations.db
python -c "from services.database.local_models import init_db; init_db()"
```

**Missing dependencies:**
```bash
# Reinstall everything
rm -rf venv node_modules ui/node_modules
python3 scripts/setup_local.py
```

**Ollama not working:**
```bash
# Check Ollama status
ollama list
curl http://localhost:11434/api/tags

# Restart Ollama
pkill ollama
ollama serve
```

### Debug Mode

```bash
# Enable debug logging
export DEBUG=true
export LOG_LEVEL=debug

# View logs
tail -f data/local/logs/api.log
```

## 🚀 Performance Tips

### For Better Performance

```yaml
# Edit config/local_config.yaml
performance:
  max_workers: 4        # Increase for faster processing
  batch_size: 20        # Process more items at once
  cache_enabled: true   # Keep caching enabled

# Monitor memory usage
# Large datasets may need more RAM
```

### For Lower Resource Usage

```yaml
performance:
  max_workers: 1        # Single thread processing  
  batch_size: 5         # Smaller batches
```

## 🛑 Stopping the System

```bash
# If using start_local.sh:
# Press Ctrl+C

# Manual stop:
pkill -f "local_annotation_api"
pkill -f "npm start"
pkill ollama

# Docker stop:
docker-compose -f docker-compose.local.yml down
```

## 📊 Sample Workflow

```bash
# 1. Start system
./start_local.sh

# 2. Upload a document via UI or:
curl -X POST http://localhost:8000/documents \
  -H "Authorization: Bearer local-admin-2024" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Shrimp Disease Study",
    "text": "Penaeus vannamei cultured at 28°C showed AHPND symptoms when infected with Vibrio parahaemolyticus.",
    "source": "research"
  }'

# 3. Generate candidates for sentences
curl -X POST http://localhost:8000/candidates/generate \
  -H "Authorization: Bearer local-admin-2024" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "abc123",
    "sent_id": "s0",
    "text": "Penaeus vannamei cultured at 28°C showed AHPND symptoms."
  }'

# 4. Review in triage queue (via UI at localhost:3000)

# 5. Export results
curl -H "Authorization: Bearer local-admin-2024" \
     http://localhost:8000/export > final_annotations.json
```

## 🎯 Perfect for Internal Teams

This setup is ideal for:
- ✅ **Small teams** (5-20 users)
- ✅ **Internal projects** (no external hosting needed)
- ✅ **Quick experiments** (up and running in minutes)
- ✅ **Training/demos** (completely self-contained)
- ✅ **Offline environments** (no internet after setup)
- ✅ **Budget-conscious** (zero ongoing costs)

## 📞 Support

If you encounter issues:

1. **Check logs**: `data/local/logs/`
2. **Reset data**: `rm -rf data/local/`
3. **Reinstall**: `python3 scripts/setup_local.py`
4. **Check processes**: `ps aux | grep -E "(python|node|ollama)"`

The local mode is designed to be bulletproof and work completely offline! 🎉