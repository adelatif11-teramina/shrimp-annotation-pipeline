# API Documentation

This document provides comprehensive documentation for the Shrimp Annotation Pipeline REST API.

## Table of Contents

1. [Authentication](#authentication)
2. [Base URL and Versioning](#base-url-and-versioning)
3. [Rate Limiting](#rate-limiting)
4. [Response Format](#response-format)
5. [Error Handling](#error-handling)
6. [Endpoints](#endpoints)
7. [WebSocket Events](#websocket-events)
8. [SDK Examples](#sdk-examples)

## Authentication

The API uses JWT (JSON Web Token) authentication. All protected endpoints require a valid JWT token in the Authorization header.

### Login

```http
POST /auth/login
Content-Type: application/json

{
  "username": "your_username",
  "password": "your_password"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": {
    "id": 1,
    "username": "your_username",
    "email": "user@example.com",
    "role": "annotator"
  }
}
```

### Using Tokens

Include the access token in the Authorization header:

```http
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
```

### Refresh Token

When the access token expires, use the refresh token to get a new one:

```http
POST /auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

## Base URL and Versioning

**Base URL:** `https://api.yourdomain.com/`

**API Version:** v1 (current)

All endpoints are prefixed with the base URL. The API is currently unversioned but will use `/v1/` prefix in future versions.

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Per User**: 100 requests per minute, 1000 per hour
- **Per IP**: 200 requests per minute for unauthenticated requests
- **Burst**: Up to 20 requests in a 10-second window

Rate limit headers are included in responses:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

## Response Format

### Success Responses

All successful responses follow this structure:

```json
{
  "data": { ... },
  "message": "Success message",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Error Responses

Error responses follow this structure:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human readable error message",
    "details": { ... }
  },
  "timestamp": "2024-01-01T12:00:00Z",
  "path": "/api/endpoint"
}
```

## Error Handling

### HTTP Status Codes

- **200**: Success
- **201**: Created
- **400**: Bad Request
- **401**: Unauthorized
- **403**: Forbidden
- **404**: Not Found
- **409**: Conflict
- **422**: Validation Error
- **429**: Too Many Requests
- **500**: Internal Server Error
- **503**: Service Unavailable

### Common Error Codes

| Code | Description |
|------|-------------|
| `INVALID_CREDENTIALS` | Invalid username or password |
| `TOKEN_EXPIRED` | JWT token has expired |
| `INSUFFICIENT_PERMISSIONS` | User lacks required permissions |
| `VALIDATION_ERROR` | Request data validation failed |
| `RESOURCE_NOT_FOUND` | Requested resource doesn't exist |
| `RATE_LIMIT_EXCEEDED` | Too many requests |
| `SERVICE_UNAVAILABLE` | External service unavailable |

## Endpoints

### Health and Status

#### Get API Health

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0",
  "environment": "production",
  "services": {
    "llm_generator": true,
    "ingestion_service": true,
    "triage_engine": true,
    "rule_engine": true
  },
  "circuit_breakers": {
    "openai_api": {
      "state": "closed",
      "failure_count": 0,
      "last_failure_time": null
    }
  }
}
```

#### Get Detailed Health

```http
GET /health/detailed
```

Requires authentication.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "components": {
    "database": true,
    "redis": true,
    "openai_api": true
  },
  "system_metrics": {
    "cpu_usage": 25.5,
    "memory_usage": 67.2,
    "disk_usage": 45.1
  },
  "message": "System is healthy"
}
```

#### Get Metrics

```http
GET /metrics/summary
```

**Response:**
```json
{
  "counters": {
    "api_requests_total": 15847,
    "candidate_requests_total": 892,
    "llm_requests_total": 445
  },
  "gauges": {
    "active_users": 12,
    "queue_size": 25,
    "system_cpu_usage": 23.4
  },
  "histograms": {
    "request_duration_seconds": {
      "count": 15847,
      "sum": 3169.4,
      "avg": 0.2,
      "p50": 0.15,
      "p95": 0.8,
      "p99": 1.5
    }
  }
}
```

### Authentication Endpoints

#### Get Current User

```http
GET /auth/me
Authorization: Bearer <token>
```

**Response:**
```json
{
  "id": 1,
  "username": "annotator1",
  "email": "annotator1@example.com",
  "role": "annotator",
  "is_active": true,
  "created_at": "2024-01-01T00:00:00Z",
  "permissions": [
    "annotate:read",
    "annotate:write",
    "queue:read"
  ]
}
```

#### Verify Token

```http
GET /auth/verify-token
Authorization: Bearer <token>
```

**Response:**
```json
{
  "valid": true,
  "user_id": 1,
  "username": "annotator1",
  "role": "annotator",
  "permissions": ["annotate:read", "annotate:write"],
  "expires_at": "2024-01-01T13:00:00Z"
}
```

### Document Management

#### Ingest Document

```http
POST /documents/ingest
Authorization: Bearer <token>
Content-Type: application/json

{
  "doc_id": "doc_001",
  "text": "Shrimp aquaculture involves raising shrimp in controlled environments...",
  "title": "Introduction to Shrimp Farming",
  "source": "research_paper",
  "metadata": {
    "author": "Dr. Smith",
    "publication_year": 2024
  }
}
```

**Response:**
```json
{
  "doc_id": "doc_001",
  "sentence_count": 25,
  "message": "Document ingested successfully"
}
```

#### List Documents

```http
GET /documents
Authorization: Bearer <token>
```

**Response:**
```json
{
  "documents": [
    {
      "doc_id": "doc_001",
      "title": "Introduction to Shrimp Farming",
      "source": "research_paper",
      "status": "processed",
      "sentence_count": 25,
      "created_at": "2024-01-01T10:00:00Z"
    }
  ],
  "count": 1,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### Get Document

```http
GET /documents/{doc_id}
Authorization: Bearer <token>
```

### Candidate Generation

#### Generate Candidates for Sentence

```http
POST /candidates/generate
Authorization: Bearer <token>
Content-Type: application/json

{
  "doc_id": "doc_001",
  "sent_id": "sent_001",
  "text": "White spot syndrome virus affects Pacific white shrimp populations.",
  "title": "Disease Management"
}
```

**Response:**
```json
{
  "doc_id": "doc_001",
  "sent_id": "sent_001",
  "candidates": {
    "entities": [
      {
        "text": "White spot syndrome virus",
        "start": 0,
        "end": 25,
        "label": "PATHOGEN",
        "confidence": 0.95
      },
      {
        "text": "Pacific white shrimp",
        "start": 34,
        "end": 54,
        "label": "SPECIES",
        "confidence": 0.92
      }
    ],
    "relations": [
      {
        "head": {"text": "White spot syndrome virus", "start": 0, "end": 25},
        "tail": {"text": "Pacific white shrimp", "start": 34, "end": 54},
        "relation": "affects",
        "confidence": 0.88
      }
    ],
    "topics": [
      {
        "label": "T_DISEASE",
        "confidence": 0.91
      }
    ]
  },
  "rule_results": {
    "entities": [
      {
        "text": "virus",
        "start": 20,
        "end": 25,
        "label": "PATHOGEN",
        "source": "pathogen_keywords"
      }
    ]
  },
  "triage_score": 0.87,
  "processing_time": 1.23
}
```

#### Batch Generate Candidates

```http
POST /candidates/batch
Authorization: Bearer <token>
Content-Type: application/json

{
  "sentences": [
    {
      "doc_id": "doc_001",
      "sent_id": "sent_001",
      "text": "Sentence 1 text...",
      "title": "Document Title"
    },
    {
      "doc_id": "doc_001",
      "sent_id": "sent_002",
      "text": "Sentence 2 text...",
      "title": "Document Title"
    }
  ],
  "batch_size": 10
}
```

### Triage Queue Management

#### Get Triage Queue

```http
GET /triage/queue?limit=10&priority_filter=high
Authorization: Bearer <token>
```

**Query Parameters:**
- `limit` (integer): Number of items to return (1-100, default: 10)
- `priority_filter` (string): Filter by priority level (low, medium, high, critical)
- `annotator` (string): Filter by assigned annotator

**Response:**
```json
[
  {
    "item_id": "item_001",
    "doc_id": "doc_001",
    "sent_id": "sent_001",
    "item_type": "entity_annotation",
    "priority_score": 0.87,
    "priority_level": "high",
    "candidate_data": {
      "entities": [...],
      "relations": [...],
      "topics": [...]
    },
    "assigned_to": "annotator1",
    "sentence_text": "White spot syndrome virus affects Pacific white shrimp...",
    "document_title": "Disease Management"
  }
]
```

#### Get Queue Statistics

```http
GET /triage/statistics
Authorization: Bearer <token>
```

**Response:**
```json
{
  "queue_size": 156,
  "priority_distribution": {
    "critical": 5,
    "high": 23,
    "medium": 89,
    "low": 39
  },
  "annotator_workload": {
    "annotator1": 45,
    "annotator2": 38,
    "unassigned": 73
  },
  "avg_processing_time": 245.5,
  "completion_rate": 0.78
}
```

#### Populate Queue

```http
POST /triage/populate
Authorization: Bearer <token>
```

Processes uploaded documents and adds candidates to the triage queue.

### Annotation Decisions

#### Submit Annotation Decision

```http
POST /annotations/decisions
Authorization: Bearer <token>
Content-Type: application/json

{
  "item_id": "item_001",
  "decision": "accepted",
  "final_annotation": {
    "entities": [
      {
        "text": "White spot syndrome virus",
        "start": 0,
        "end": 25,
        "label": "PATHOGEN"
      }
    ],
    "relations": [
      {
        "head": {"text": "White spot syndrome virus", "start": 0, "end": 25},
        "tail": {"text": "Pacific white shrimp", "start": 34, "end": 54},
        "relation": "affects"
      }
    ],
    "topics": ["T_DISEASE"]
  },
  "annotator": "annotator1",
  "notes": "High confidence pathogen identification"
}
```

**Decision Types:**
- `accepted`: Accept candidate as-is
- `rejected`: Reject candidate completely
- `modified`: Accept with modifications

**Response:**
```json
{
  "status": "success",
  "message": "Decision recorded"
}
```

### Data Export

#### Export Gold Annotations

```http
POST /export/gold
Authorization: Bearer <token>
Content-Type: application/json

{
  "format": "jsonl",
  "doc_ids": ["doc_001", "doc_002"],
  "date_range": {
    "start": "2024-01-01",
    "end": "2024-01-31"
  },
  "annotator": "annotator1"
}
```

**Supported Formats:**
- `jsonl`: JSON Lines format
- `conll`: CoNLL format for NER
- `scibert`: SciBERT training format

**Response:**
```json
{
  "export_path": "/data/exports/gold_export_20240101_120000.jsonl",
  "item_count": 234,
  "format": "jsonl"
}
```

### ML Integration

#### Get Training Data

```http
GET /integration/training-data?format=scibert&min_annotations=5
Authorization: Bearer <token>
```

**Query Parameters:**
- `format` (string): Output format (default: "scibert")
- `split` (string): Data split (train/test/val)
- `min_annotations` (integer): Minimum annotations per document

**Response:**
```json
{
  "training_data": [
    {
      "text": "White spot syndrome virus affects Pacific white shrimp populations.",
      "entities": [
        {"start": 0, "end": 25, "label": "PATHOGEN"},
        {"start": 34, "end": 54, "label": "SPECIES"}
      ],
      "relations": [
        {
          "head": {"start": 0, "end": 25},
          "tail": {"start": 34, "end": 54},
          "relation": "affects"
        }
      ],
      "doc_id": "doc_001",
      "annotator": "annotator1",
      "timestamp": "2024-01-01T12:00:00Z"
    }
  ],
  "count": 234,
  "format": "scibert",
  "generation_time": "2024-01-01T12:00:00Z"
}
```

#### Submit Model Feedback

```http
POST /integration/model-feedback
Authorization: Bearer <token>
Content-Type: application/json

{
  "model_version": "v1.2.3",
  "performance_metrics": {
    "entity_f1": 0.87,
    "relation_f1": 0.82,
    "topic_accuracy": 0.91
  },
  "training_data_stats": {
    "total_examples": 1500,
    "entity_distribution": {
      "PATHOGEN": 234,
      "SPECIES": 456,
      "DISEASE": 189
    }
  },
  "recommendations": [
    "Need more CHEMICAL entity examples",
    "LOCATION entities showing low precision"
  ]
}
```

### System Statistics

#### Get Overview Statistics

```http
GET /statistics/overview
Authorization: Bearer <token>
```

**Response:**
```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "services": {
    "triage": {
      "queue_size": 156,
      "completion_rate": 0.78
    },
    "rules": {
      "patterns_matched": 1847,
      "accuracy": 0.73
    }
  },
  "gold_annotations": 2156
}
```

## WebSocket Events

For real-time updates, the API supports WebSocket connections at `/ws`.

### Connection

```javascript
const ws = new WebSocket('wss://api.yourdomain.com/ws');
ws.onopen = function() {
    // Send authentication
    ws.send(JSON.stringify({
        type: 'auth',
        token: 'your-jwt-token'
    }));
};
```

### Events

#### Queue Updates

```json
{
  "type": "queue_update",
  "data": {
    "queue_size": 157,
    "new_items": 1,
    "completed_items": 0
  }
}
```

#### Annotation Progress

```json
{
  "type": "annotation_progress",
  "data": {
    "doc_id": "doc_001",
    "progress": 0.75,
    "annotated_sentences": 18,
    "total_sentences": 24
  }
}
```

## SDK Examples

### Python SDK

```python
import requests
from typing import Dict, List, Optional

class ShrimpAnnotationAPI:
    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.token = self._login(username, password)
        self.session.headers.update({
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        })
    
    def _login(self, username: str, password: str) -> str:
        response = requests.post(
            f'{self.base_url}/auth/login',
            json={'username': username, 'password': password}
        )
        response.raise_for_status()
        return response.json()['access_token']
    
    def ingest_document(self, doc_id: str, text: str, title: str = None, 
                       source: str = 'api', metadata: Dict = None) -> Dict:
        data = {
            'doc_id': doc_id,
            'text': text,
            'title': title,
            'source': source,
            'metadata': metadata or {}
        }
        response = self.session.post(f'{self.base_url}/documents/ingest', json=data)
        response.raise_for_status()
        return response.json()
    
    def generate_candidates(self, doc_id: str, sent_id: str, text: str, 
                          title: str = None) -> Dict:
        data = {
            'doc_id': doc_id,
            'sent_id': sent_id,
            'text': text,
            'title': title
        }
        response = self.session.post(f'{self.base_url}/candidates/generate', json=data)
        response.raise_for_status()
        return response.json()
    
    def get_queue(self, limit: int = 10, priority_filter: str = None) -> List[Dict]:
        params = {'limit': limit}
        if priority_filter:
            params['priority_filter'] = priority_filter
        
        response = self.session.get(f'{self.base_url}/triage/queue', params=params)
        response.raise_for_status()
        return response.json()
    
    def submit_decision(self, item_id: str, decision: str, 
                       final_annotation: Dict = None, annotator: str = None,
                       notes: str = None) -> Dict:
        data = {
            'item_id': item_id,
            'decision': decision,
            'final_annotation': final_annotation,
            'annotator': annotator,
            'notes': notes
        }
        response = self.session.post(f'{self.base_url}/annotations/decisions', json=data)
        response.raise_for_status()
        return response.json()

# Usage example
api = ShrimpAnnotationAPI('https://api.yourdomain.com', 'username', 'password')

# Ingest a document
result = api.ingest_document(
    doc_id='test_doc',
    text='Shrimp farming requires careful water quality management.',
    title='Aquaculture Best Practices'
)

# Get candidates for annotation
candidates = api.generate_candidates(
    doc_id='test_doc',
    sent_id='sent_001',
    text='Shrimp farming requires careful water quality management.'
)

# Get queue items
queue_items = api.get_queue(limit=5, priority_filter='high')

# Submit annotation decision
decision = api.submit_decision(
    item_id='item_001',
    decision='accepted',
    final_annotation=candidates['candidates'],
    annotator='python_user'
)
```

### JavaScript SDK

```javascript
class ShrimpAnnotationAPI {
    constructor(baseUrl, username, password) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.token = null;
        this.login(username, password);
    }
    
    async login(username, password) {
        const response = await fetch(`${this.baseUrl}/auth/login`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({username, password})
        });
        
        if (!response.ok) throw new Error('Login failed');
        
        const data = await response.json();
        this.token = data.access_token;
    }
    
    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        const headers = {
            'Authorization': `Bearer ${this.token}`,
            'Content-Type': 'application/json',
            ...options.headers
        };
        
        const response = await fetch(url, {
            ...options,
            headers
        });
        
        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }
        
        return response.json();
    }
    
    async ingestDocument(docId, text, title = null, source = 'api', metadata = {}) {
        return this.request('/documents/ingest', {
            method: 'POST',
            body: JSON.stringify({
                doc_id: docId,
                text,
                title,
                source,
                metadata
            })
        });
    }
    
    async generateCandidates(docId, sentId, text, title = null) {
        return this.request('/candidates/generate', {
            method: 'POST',
            body: JSON.stringify({
                doc_id: docId,
                sent_id: sentId,
                text,
                title
            })
        });
    }
    
    async getQueue(limit = 10, priorityFilter = null) {
        const params = new URLSearchParams({limit});
        if (priorityFilter) params.append('priority_filter', priorityFilter);
        
        return this.request(`/triage/queue?${params}`);
    }
    
    async submitDecision(itemId, decision, finalAnnotation = null, annotator = null, notes = null) {
        return this.request('/annotations/decisions', {
            method: 'POST',
            body: JSON.stringify({
                item_id: itemId,
                decision,
                final_annotation: finalAnnotation,
                annotator,
                notes
            })
        });
    }
}

// Usage example
const api = new ShrimpAnnotationAPI('https://api.yourdomain.com', 'username', 'password');

// Ingest document
const ingestResult = await api.ingestDocument(
    'test_doc',
    'Shrimp farming requires careful water quality management.',
    'Aquaculture Best Practices'
);

// Generate candidates
const candidates = await api.generateCandidates(
    'test_doc',
    'sent_001',
    'Shrimp farming requires careful water quality management.'
);

// Get queue items
const queueItems = await api.getQueue(5, 'high');

// Submit decision
const decision = await api.submitDecision(
    'item_001',
    'accepted',
    candidates.candidates,
    'js_user'
);
```

---

**API Version**: 1.0  
**Last Updated**: 2024-01-01  
**Interactive Docs**: Available at `/docs` when server is running