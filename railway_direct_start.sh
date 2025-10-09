#!/bin/bash
# Direct Railway startup - no complex testing, just work!

set -e
PORT=${PORT:-8000}

echo "🚀 DIRECT Railway Startup - Port $PORT"
echo "📂 Working in: $(pwd)"
echo "🐍 Python: $(python --version)"

# Set essential environment
export PYTHONPATH=/app:/app/src:$PYTHONPATH
export ENVIRONMENT=production
export API_HOST=0.0.0.0
export DEBUG=false

# JWT secret
if [ -z "$JWT_SECRET_KEY" ]; then
    export JWT_SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
fi

# OpenAI key check
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️ No OPENAI_API_KEY - will use fallback triplets"
else
    echo "✅ OPENAI_API_KEY configured"
fi

# Install critical deps quickly
echo "📦 Installing key dependencies..."
python -m pip install -q fastapi uvicorn openai pyjwt bcrypt python-multipart pydantic-settings

# Build frontend quickly
echo "🔨 Building frontend..."
if [ -d "ui" ]; then
    cd ui && npm install --silent && npm run build && cd ..
    echo "✅ Frontend built"
fi

# Direct API startup - try production first, immediate fallback
echo "🎯 Starting API server..."

if python -c "
import sys
sys.path.insert(0, '/app')
try:
    from services.api.annotation_api import app
    print('✅ FULL API AVAILABLE')
    exit(0)
except Exception as e:
    print(f'❌ Full API failed: {e}')
    exit(1)
" 2>/dev/null; then
    echo "🚀 Starting FULL annotation API with triplet generation"
    exec python railway_production_api.py
else
    echo "🔄 Starting BULLETPROOF API with smart fallbacks"
    exec python railway_bulletproof_api.py
fi