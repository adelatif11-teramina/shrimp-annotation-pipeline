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

# Try the full API first
if python -c "import services.api.annotation_api" 2>/dev/null; then
    echo "Starting full annotation API"
    python -m uvicorn services.api.annotation_api:app --host 0.0.0.0 --port $PORT --workers 1
else
    echo "Starting minimal Railway-compatible API"
    python -m uvicorn railway_api:app --host 0.0.0.0 --port $PORT --workers 1
fi