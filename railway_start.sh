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

# Build React frontend if needed
if [ -d "ui" ] && [ -f "ui/package.json" ]; then
    echo "Building React frontend..."
    cd ui
    
    # Install dependencies if needed
    if [ ! -d "node_modules" ]; then
        echo "Installing frontend dependencies..."
        npm install --production --silent
    fi
    
    # Build if not exists or if in production
    if [ ! -d "build" ] || [ "$RAILWAY_ENVIRONMENT" = "production" ]; then
        echo "Building React app..."
        npm run build
    fi
    
    cd ..
    
    if [ -d "ui/build" ]; then
        echo "✅ React frontend built successfully"
        echo "Frontend files:"
        ls -la ui/build/ | head -5
    else
        echo "❌ React frontend build failed"
    fi
else
    echo "⚠️ No React frontend source found"
fi

# Try PostgreSQL production API first if DATABASE_URL is set
if [ -n "$DATABASE_URL" ]; then
    echo "DATABASE_URL detected, setting up PostgreSQL and starting production API..."
    
    # Setup database if needed
    python scripts/setup_railway_database.py
    
    if [ $? -eq 0 ]; then
        echo "✅ Database setup successful, starting PostgreSQL production API"
        exec python -m uvicorn services.api.production_api:app --host 0.0.0.0 --port $PORT --workers 1
    else
        echo "❌ PostgreSQL setup failed, falling back to in-memory API"
        exec python -m uvicorn railway_api:app --host 0.0.0.0 --port $PORT --workers 1
    fi
else
    echo "ℹ️ No DATABASE_URL found, starting Railway-compatible API"
    echo "To use PostgreSQL, add a PostgreSQL service and set DATABASE_URL"
    exec python -m uvicorn railway_api:app --host 0.0.0.0 --port $PORT --workers 1
fi