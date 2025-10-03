#!/bin/bash
# Railway startup script

# Set default port if not provided
PORT=${PORT:-8000}

# Set required environment variables
export PYTHONPATH=/app
export ENVIRONMENT=production
export API_HOST=0.0.0.0
export DEBUG=false
export LOG_LEVEL=INFO

# Generate JWT secret if not provided
if [ -z "$JWT_SECRET_KEY" ]; then
    echo "Warning: JWT_SECRET_KEY not set, generating temporary key"
    export JWT_SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
fi

# Start the application (try full version first, fallback to minimal)
echo "Starting shrimp annotation pipeline on port $PORT"

# Check if React frontend was built
if [ -d "ui/build" ]; then
    echo "React frontend found at ui/build"
    ls -la ui/build/ | head -5
else
    echo "No React frontend build found"
fi

# Try the full API first (it has better frontend serving)
echo "Attempting to start full annotation API..."
if python -c "import services.api.annotation_api" 2>/dev/null; then
    echo "Starting full annotation API with frontend serving"
    python -m uvicorn services.api.annotation_api:app --host 0.0.0.0 --port $PORT --workers 1
else
    echo "Full API failed, starting minimal Railway-compatible API"
    python -m uvicorn railway_api:app --host 0.0.0.0 --port $PORT --workers 1
fi