#!/bin/bash
# Start Local Development Environment

echo "🚀 Starting Shrimp Annotation Pipeline (Local Mode)"
echo "================================================"

# Activate virtual environment
source venv/bin/activate

# Export environment variables
export $(grep -v '^#' .env.local | xargs)

# Check if Ollama is running (optional)
if command -v ollama &> /dev/null; then
    echo "✓ Ollama detected"
    # Ensure Ollama is serving
    ollama serve > /dev/null 2>&1 &
    sleep 2
else
    echo "⚠ Ollama not found - using rules-only mode"
fi

# Start Backend API
echo "Starting API server..."
python services/api/mock_api.py &
API_PID=$!
echo "✓ API server started (PID: $API_PID)"

# Wait for API to be ready
sleep 3

# Start Frontend
echo "Starting UI server..."
cd ui && npm start &
UI_PID=$!
echo "✓ UI server started (PID: $UI_PID)"

echo ""
echo "================================================"
echo "✅ Local environment is running!"
echo ""
echo "Access points:"
echo "  📊 UI:       http://localhost:3000"
echo "  🔌 API:      http://localhost:8000"
echo "  📖 API Docs: http://localhost:8000/docs"
echo ""
echo "Default login tokens:"
echo "  Admin:      local-admin-2024"
echo "  Annotator:  anno-team-001"
echo "  Reviewer:   review-lead-003"
echo ""
echo "Press Ctrl+C to stop all services"
echo "================================================"

# Wait for interrupt
trap "kill $API_PID $UI_PID 2>/dev/null; echo 'Services stopped'" EXIT
wait
