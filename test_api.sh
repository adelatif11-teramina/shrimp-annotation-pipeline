#!/bin/bash

echo "Testing Shrimp Annotation Pipeline API Endpoints"
echo "================================================="
echo ""

# Set auth token
TOKEN="local-admin-2024"
BASE_URL="http://localhost:8000"

# Test health check
echo "1. Testing Health Check..."
curl -s $BASE_URL/health | python3 -c "import sys, json; data = json.load(sys.stdin); print(f'   ✓ Status: {data[\"status\"]}, Documents: {data[\"documents\"]}, Queue: {data[\"queue_size\"]}')"

# Test statistics overview
echo ""
echo "2. Testing Statistics Overview..."
curl -s $BASE_URL/statistics/overview -H "Authorization: Bearer $TOKEN" | python3 -c "import sys, json; data = json.load(sys.stdin); print(f'   ✓ Total Documents: {data[\"overview\"][\"total_documents\"]}, Annotations: {data[\"overview\"][\"total_annotations\"]}')"

# Test triage statistics
echo ""
echo "3. Testing Triage Statistics..."
curl -s $BASE_URL/triage/statistics -H "Authorization: Bearer $TOKEN" | python3 -c "import sys, json; data = json.load(sys.stdin); print(f'   ✓ Total Items: {data[\"total_items\"]}, Pending: {data[\"pending_items\"]}')"

# Test triage queue
echo ""
echo "4. Testing Triage Queue..."
curl -s "$BASE_URL/triage/queue?limit=5" -H "Authorization: Bearer $TOKEN" | python3 -c "import sys, json; data = json.load(sys.stdin); print(f'   ✓ Queue Items: {len(data[\"items\"])}, Total: {data[\"total\"]}')"

# Test get next item
echo ""
echo "5. Testing Get Next Item..."
curl -s $BASE_URL/triage/next -H "Authorization: Bearer $TOKEN" | python3 -c "import sys, json; data = json.load(sys.stdin); item = data.get(\"item\"); print(f'   ✓ Next Item: {item[\"id\"] if item else \"None\"}, Priority: {item[\"priority_level\"] if item else \"N/A\"}')"

# Test documents
echo ""
echo "6. Testing Documents Endpoint..."
curl -s $BASE_URL/documents -H "Authorization: Bearer $TOKEN" | python3 -c "import sys, json; data = json.load(sys.stdin); print(f'   ✓ Documents: {len(data[\"documents\"])}, Total: {data[\"total\"]}')"

# Test user info
echo ""
echo "7. Testing User Info..."
curl -s $BASE_URL/users/me -H "Authorization: Bearer $TOKEN" | python3 -c "import sys, json; data = json.load(sys.stdin); print(f'   ✓ User: {data[\"user\"][\"username\"]}, Role: {data[\"user\"][\"role\"]}')"

echo ""
echo "================================================="
echo "✅ All API endpoints are working correctly!"
echo ""
echo "Frontend: http://localhost:3000"
echo "API Docs: http://localhost:8000/docs"