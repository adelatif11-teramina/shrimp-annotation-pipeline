#!/bin/bash
# Enhanced Railway startup script with comprehensive error handling

set -e  # Exit on any error

# Set default port if not provided
PORT=${PORT:-8000}
echo "🚀 Railway Enhanced Startup Script"
echo "📍 Port: $PORT"
echo "📂 Working directory: $(pwd)"
echo "🐍 Python version: $(python --version)"

# Set required environment variables
export PYTHONPATH=/app:$PYTHONPATH
export ENVIRONMENT=production
export API_HOST=0.0.0.0
export DEBUG=false
export LOG_LEVEL=INFO

# Generate JWT secret if not provided
if [ -z "$JWT_SECRET_KEY" ]; then
    echo "⚠️ Warning: JWT_SECRET_KEY not set, generating temporary key"
    export JWT_SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
else
    echo "✅ JWT_SECRET_KEY configured"
fi

# Check OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️ Warning: OPENAI_API_KEY not set - triplets will use fallback mode"
else
    echo "✅ OPENAI_API_KEY configured: ${OPENAI_API_KEY:0:10}..."
fi

echo ""
echo "🔧 === DEPENDENCY INSTALLATION ==="

# Install critical dependencies with error handling
echo "📦 Installing production dependencies..."
pip install --quiet --no-warn-script-location \
    redis>=4.5 \
    pyjwt>=2.8 \
    bcrypt>=4.0 \
    python-multipart>=0.0.6 \
    uvicorn>=0.20.0 \
    fastapi>=0.95.0 \
    openai>=1.0.0 \
    asyncio-timeout \
    || echo "⚠️ Some dependencies failed to install but continuing..."

echo ""
echo "🏗️ === FRONTEND BUILD ==="

# Build React frontend with enhanced error handling
if [ -d "ui" ] && [ -f "ui/package.json" ]; then
    echo "📦 Building React frontend..."
    cd ui
    
    # Install dependencies with timeout
    echo "📥 Installing frontend dependencies..."
    timeout 300 npm install --silent --no-audit --no-fund || {
        echo "❌ Frontend dependency installation timed out, continuing..."
        cd ..
    }
    
    if [ -f "package.json" ]; then
        # Build the frontend with timeout
        echo "🔨 Building React app..."
        timeout 180 npm run build || {
            echo "❌ Frontend build failed, continuing without frontend..."
            cd ..
        }
    fi
    
    cd ..
    
    # Verify build
    if [ -d "ui/build" ] && [ -f "ui/build/index.html" ]; then
        echo "✅ React frontend built successfully"
        echo "📊 Build size: $(du -sh ui/build | cut -f1)"
    else
        echo "⚠️ React frontend build incomplete"
    fi
else
    echo "⚠️ No React frontend source found, API-only mode"
fi

echo ""
echo "🧪 === API COMPONENT TESTING ==="

# Test individual components
echo "🔍 Testing core imports..."

# Test basic imports
python -c "
import sys
print(f'✅ Python path: {sys.path[:3]}...')
try:
    import fastapi
    print('✅ FastAPI available')
except ImportError as e:
    print(f'❌ FastAPI missing: {e}')

try:
    import openai
    print('✅ OpenAI client available')
except ImportError as e:
    print(f'❌ OpenAI missing: {e}')
" || echo "⚠️ Import test had issues"

# Test LLM generator specifically
echo "🤖 Testing LLM candidate generator..."
python -c "
try:
    from services.candidates.llm_candidate_generator import LLMCandidateGenerator
    print('✅ LLM Generator import successful')
except ImportError as e:
    print(f'❌ LLM Generator import failed: {e}')
" || echo "❌ LLM Generator unavailable"

# Test triplet workflow
echo "🔗 Testing triplet workflow..."
python -c "
try:
    from services.candidates.triplet_workflow import TripletWorkflow
    print('✅ Triplet Workflow import successful')
except ImportError as e:
    print(f'❌ Triplet Workflow import failed: {e}')
" || echo "❌ Triplet Workflow unavailable"

# Test main annotation API
echo "🎯 Testing main annotation API..."
python -c "
try:
    from services.api.annotation_api import app
    print('✅ Main Annotation API import successful')
    FULL_API_AVAILABLE = True
except ImportError as e:
    print(f'❌ Main Annotation API import failed: {e}')
    FULL_API_AVAILABLE = False

# Write status to file for startup script
with open('/tmp/api_status.txt', 'w') as f:
    f.write('FULL' if FULL_API_AVAILABLE else 'MINIMAL')
" || echo "❌ Main API test failed"

echo ""
echo "🚀 === API SERVER STARTUP ==="

# Read API availability status
API_STATUS=$(cat /tmp/api_status.txt 2>/dev/null || echo "UNKNOWN")

if [ "$API_STATUS" = "FULL" ]; then
    echo "✅ Full annotation API available - starting production server"
    echo "🔧 Features: Triplet generation, LLM integration, rule engine, audit workflow"
    
    # Verify railway_production_api.py exists
    if [ -f "railway_production_api.py" ]; then
        echo "🚀 Starting railway_production_api.py with full features..."
        exec python railway_production_api.py
    else
        echo "❌ railway_production_api.py not found, falling back to uvicorn"
        exec python -m uvicorn services.api.annotation_api:app --host 0.0.0.0 --port $PORT --workers 1
    fi
    
else
    echo "⚠️ Full API not available (status: $API_STATUS)"
    echo "🔄 Starting minimal Railway API with enhanced fallback features"
    
    # Create enhanced minimal API if needed
    if [ ! -f "railway_api.py" ]; then
        echo "❌ railway_api.py not found, creating emergency API..."
        cat > emergency_api.py << 'EOF'
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="Emergency Railway API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/api/health")
async def health():
    return {"status": "healthy", "mode": "emergency"}

@app.post("/api/candidates/generate") 
async def emergency_candidates(request: dict):
    return {
        "candidates": {
            "triplets": [{
                "triplet_id": "emergency_1",
                "head": {"text": "System", "type": "STATUS"},
                "relation": "REPORTS", 
                "tail": {"text": "Emergency Mode", "type": "STATUS"},
                "evidence": "API running in emergency mode",
                "confidence": 1.0
            }],
            "metadata": {"audit_notes": "Emergency API active"}
        }
    }
EOF
        exec python -m uvicorn emergency_api:app --host 0.0.0.0 --port $PORT
    else
        exec python -m uvicorn railway_api:app --host 0.0.0.0 --port $PORT --workers 1
    fi
fi