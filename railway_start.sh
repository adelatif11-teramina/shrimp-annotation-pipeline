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

# Build React frontend
echo "🔨 Building React frontend..."
if [ -d "ui" ] && [ -f "ui/package.json" ]; then
    cd ui
    
    # Install dependencies
    echo "📦 Installing frontend dependencies..."
    npm install --silent
    
    # Build the frontend
    echo "🏗️ Building React app..."
    npm run build
    
    cd ..
    
    # Verify build
    if [ -d "ui/build" ] && [ -f "ui/build/index.html" ]; then
        echo "✅ React frontend built successfully"
        echo "📁 Frontend files:"
        ls -la ui/build/ | head -10
        echo "📄 Index.html size: $(wc -c < ui/build/index.html) bytes"
    else
        echo "❌ React frontend build failed or incomplete"
        echo "📁 UI directory contents:"
        ls -la ui/ || echo "No ui directory"
        echo "📁 Build directory contents:"
        ls -la ui/build/ 2>/dev/null || echo "No build directory"
    fi
else
    echo "⚠️ No React frontend source found"
    echo "📁 Root directory contents:"
    ls -la | grep -E "(ui|package)"
fi

# Try full production API first
echo "🚀 Starting production API with full annotation features..."

# Install missing dependencies if needed
echo "📦 Installing production dependencies..."
pip install -q redis>=4.5 pyjwt>=2.8 bcrypt>=4.0 python-multipart>=0.0.6 2>/dev/null || echo "⚠️ Some dependencies may be missing"

# Try full annotation API with fallback
if python -c "import services.api.annotation_api" 2>/dev/null; then
    echo "✅ Full annotation API available, starting production server"
    exec python railway_production_api.py
else
    echo "⚠️ Full API not available, starting minimal Railway API"
    exec python -m uvicorn railway_api:app --host 0.0.0.0 --port $PORT --workers 1
fi