# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a Human-in-the-Loop (HITL) annotation pipeline for shrimp aquaculture domain. It's a production-ready system for building high-quality Knowledge Graph and topic modeling training data using microservices architecture.

## Core Technologies
- **Backend**: Python 3.8+ with FastAPI, SQLAlchemy 2.0+, PostgreSQL, Redis
- **Frontend**: React 18.2 with Material-UI, React Query, Recharts
- **ML/NLP**: OpenAI API, Ollama, spaCy, Label Studio
- **Infrastructure**: Docker Compose

## Essential Commands

### Setup and Installation
```bash
# Automated full setup (recommended)
python scripts/setup_project.py --full-build

# Manual backend setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Frontend setup
cd ui && npm install && npm run build
```

### Development
```bash
# Start all services with Docker
docker-compose up -d

# Run individual services for development
python services/api/annotation_api.py  # API server (port 8000)
cd ui && npm start  # React dev server (port 3000)

# Generate LLM candidates
python services/candidates/llm_candidate_generator.py

# Export training data
python scripts/export_training_data.py --format scibert

# Export documents for RAG storage
python scripts/export_documents_for_rag.py --all  # Export all documents
python scripts/export_documents_for_rag.py --summary  # Show export summary
```

### Testing
```bash
# Backend tests with coverage
pytest tests/ -v --cov=services
pytest tests/test_specific.py::TestClass::test_method  # Run single test

# Frontend tests
cd ui && npm test -- --coverage --watchAll=false
cd ui && npm test -- --testNamePattern="ComponentName"  # Run specific test
```

### Code Quality
```bash
# Python formatting and linting
black services/ tests/
flake8 services/ tests/
mypy services/ --ignore-missing-imports

# Frontend linting
cd ui && npm run lint
```

## Architecture Overview

### Microservices (9 Core Services)
1. **Ingestion Service** (`services/ingestion/`) - Document processing and sentence segmentation
2. **LLM Candidate Generator** (`services/candidates/`) - OpenAI/Ollama integration with intelligent caching
3. **Rule Engine** (`services/rule_engine/`) - Pattern-based baseline annotations
4. **Triage Queue** (`services/queue/`) - Multi-factor prioritization system
5. **Annotation API** (`services/api/`) - FastAPI server connecting all services
6. **Gold Store** (`services/gold_store/`) - Versioned storage of validated annotations
7. **Quality Control** (`services/quality/`) - IAA metrics and performance monitoring
8. **ML Integration** (`services/ml_integration/`) - REST APIs and batch exports
9. **Auto-Accept** (`services/auto_accept/`) - Configurable automation rules

### Key Directories
- `shared/` - Domain ontology, schemas, and prompts shared across services
- `ui/` - React-based annotation workspace with custom components
- `data/` - Data storage (raw documents, candidates, gold annotations, exports)
- `config/` - YAML configurations for all services
- `tests/` - Comprehensive test suite for all components

### Database Models (SQLAlchemy)
Main models in `services/api/models.py`:
- `Document`, `Sentence` - Document structure
- `Annotation`, `Relation` - Core annotation data
- `QualityMetric`, `AnnotationStats` - Quality tracking
- `User`, `AnnotationSession` - User management

### Frontend Architecture
React app in `ui/src/`:
- `components/` - Reusable UI components (AnnotationWorkspace, EntityHighlighter, etc.)
- `services/` - API client and data services
- `hooks/` - Custom React hooks for data fetching
- `utils/` - Helper functions and constants

## Domain-Specific Configuration

### Entity Types (11)
SPECIES, PATHOGEN, DISEASE, SYMPTOM, CHEMICAL, EQUIPMENT, LOCATION, DATE, MEASUREMENT, PROCESS, FEED

### Relation Types (9)
causes, infected_by, infects, treated_with, prevents, located_in, measured_by, occurs_at, uses, affects

### Topic Categories (10)
T_DISEASE, T_TREATMENT, T_PREVENTION, T_DIAGNOSIS, T_NUTRITION, T_WATER_QUALITY, T_PRODUCTION, T_HARVEST, T_ECONOMICS, T_GENERAL

## Service Endpoints
- API Server: `http://localhost:8000`
- Frontend: `http://localhost:3000`
- Label Studio: `http://localhost:8080`
- PostgreSQL: `localhost:5432`
- Redis: `localhost:6379`

## Working with LLM Integration
The system supports both OpenAI and Ollama models. Configuration in `config/llm_config.yaml`:
- OpenAI: Requires API key in environment variable `OPENAI_API_KEY`
- Ollama: Requires local Ollama server running (can be started via docker-compose)

## Data Flow
1. Documents ingested via `services/ingestion/`
2. LLM generates candidate annotations
3. Candidates enter triage queue with priority scoring
4. Annotations reviewed in UI or Label Studio
5. Validated annotations stored in gold store
6. Export to ML-ready formats (SCIBERT, CoNLL, JSON)

## Common Development Tasks

### Adding New Entity/Relation Types
1. Update `shared/schemas/ontology.py`
2. Update prompts in `shared/prompts/`
3. Rebuild UI: `cd ui && npm run build`
4. Update database schema if needed

### Modifying LLM Prompts
Edit prompt templates in `shared/prompts/`:
- `ner_prompt.txt` - Entity recognition
- `re_prompt.txt` - Relation extraction
- `topic_prompt.txt` - Topic classification

### Database Migrations
```bash
# After modifying models.py
alembic revision --autogenerate -m "Description"
alembic upgrade head
```

## Performance Considerations
- LLM responses are cached in Redis to minimize API costs
- Batch processing recommended for large document sets
- Frontend uses React Query for efficient data fetching and caching
- Database indexes on frequently queried fields (document_id, annotation_type, status)
