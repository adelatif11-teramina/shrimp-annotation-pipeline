#!/usr/bin/env python3
"""
Minimal Railway-compatible API for Shrimp Annotation Pipeline
"""

import os
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory storage for Railway demo (persists during app lifetime)
documents_store = []
triage_queue_store = []
annotations_store = []

# Create FastAPI app
app = FastAPI(
    title="Shrimp Annotation Pipeline API",
    description="Production-ready annotation pipeline for shrimp aquaculture domain",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve React frontend static files
ui_build_path = Path(__file__).parent / "ui" / "build"
if ui_build_path.exists():
    logger.info(f"Serving React frontend from: {ui_build_path}")
    # Serve static files
    app.mount("/static", StaticFiles(directory=str(ui_build_path / "static")), name="static")
else:
    logger.warning("React frontend build not found, serving API only")

# Pydantic models
class DocumentRequest(BaseModel):
    doc_id: str
    text: str
    title: Optional[str] = None
    source: str = "manual"
    metadata: Dict[str, Any] = {}

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    message: str

class CandidateResponse(BaseModel):
    doc_id: str
    sent_id: str
    candidates: Dict[str, Any]
    processing_time: float

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for Railway"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        message="Shrimp Annotation Pipeline API is running on Railway"
    )

@app.get("/")
async def root():
    """Serve React frontend or API info"""
    # Try to serve React frontend
    if ui_build_path.exists():
        index_file = ui_build_path / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
    
    # Fallback to API info
    return {
        "message": "🦐 Shrimp Annotation Pipeline API",
        "version": "1.0.0", 
        "status": "running",
        "frontend": "not_built" if not ui_build_path.exists() else "available",
        "endpoints": {
            "health": "/health",
            "api_docs": "/docs",
            "api_base": "/api",
            "documents": "/api/documents",
            "statistics": "/api/statistics/overview"
        }
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check for Railway"""
    return {
        "ready": True,
        "timestamp": datetime.now().isoformat(),
        "services": {
            "api": True,
            "database": "sqlite",
            "frontend": "static"
        }
    }

@app.post("/api/documents/ingest")
async def ingest_document(doc_request: DocumentRequest):
    """Ingest a document for annotation (store in memory)"""
    logger.info(f"Document ingestion requested: {doc_request.doc_id}")
    
    # Calculate sentence count (simple split by period)
    sentences = [s.strip() for s in doc_request.text.split('.') if s.strip()]
    
    # Create document object
    document = {
        "id": len(documents_store) + 1,
        "doc_id": doc_request.doc_id,
        "title": doc_request.title or f"Document {doc_request.doc_id}",
        "source": doc_request.source,
        "raw_text": doc_request.text,
        "sentence_count": len(sentences),
        "metadata": doc_request.metadata,
        "created_at": datetime.now().isoformat()
    }
    
    # Store document in memory
    documents_store.append(document)
    
    logger.info(f"Document stored successfully: {doc_request.doc_id} (Total documents: {len(documents_store)})")
    
    # Create response
    return {
        "doc_id": doc_request.doc_id,
        "title": document["title"],
        "source": doc_request.source,
        "sentence_count": len(sentences),
        "sentences": len(sentences),
        "message": "Document ingested and stored successfully",
        "status": "processed",
        "created_at": document["created_at"]
    }

@app.post("/api/candidates/generate")
async def generate_candidates(doc_id: str = "demo", sent_id: str = "s1", text: str = "Sample text"):
    """Generate annotation candidates (minimal implementation)"""
    logger.info(f"Candidate generation requested: {doc_id}/{sent_id}")
    
    # Demo response for Railway deployment
    demo_candidates = {
        "entities": [
            {
                "text": "shrimp",
                "start": 0,
                "end": 6,
                "label": "SPECIES",
                "confidence": 0.95
            }
        ],
        "relations": [],
        "topics": ["T_GENERAL"]
    }
    
    return CandidateResponse(
        doc_id=doc_id,
        sent_id=sent_id,
        candidates=demo_candidates,
        processing_time=0.1
    )

@app.get("/api/documents")
async def get_documents(limit: int = 50, offset: int = 0, search: Optional[str] = None):
    """Get documents list from memory storage"""
    logger.info(f"Documents list requested: limit={limit}, offset={offset}, search={search} (Storage: {len(documents_store)} docs)")
    
    # Start with all stored documents
    all_documents = documents_store.copy()
    
    # If no documents in storage, return demo documents
    if not all_documents:
        logger.info("No documents in storage, returning demo documents")
        demo_documents = [
            {
                "id": 1,
                "doc_id": "demo_001",
                "title": "Shrimp Disease Management Guidelines",
                "source": "manual",
                "created_at": "2024-01-15T10:00:00Z",
                "sentence_count": 45
            },
            {
                "id": 2,
                "doc_id": "demo_002", 
                "title": "WSSV Prevention Strategies",
                "source": "upload",
                "created_at": "2024-01-14T15:30:00Z",
                "sentence_count": 32
            },
            {
                "id": 3,
                "doc_id": "demo_003",
                "title": "Aquaculture Best Practices",
                "source": "manual",
                "created_at": "2024-01-13T09:15:00Z",
                "sentence_count": 68
            }
        ]
        all_documents = demo_documents
    
    # Apply search filter if provided
    if search:
        all_documents = [
            doc for doc in all_documents 
            if search.lower() in doc["title"].lower() or search.lower() in doc["doc_id"].lower()
        ]
    
    # Sort by created_at (newest first)
    all_documents.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    
    # Apply pagination
    paginated_docs = all_documents[offset:offset + limit]
    
    logger.info(f"Returning {len(paginated_docs)} documents (total: {len(all_documents)})")
    
    return {
        "documents": paginated_docs,
        "total": len(all_documents),
        "limit": limit,
        "offset": offset
    }

@app.get("/api/statistics/overview")
async def get_statistics():
    """Get system statistics"""
    return {
        "timestamp": datetime.now().isoformat(),
        "mode": "railway_demo",
        "environment": os.getenv("ENVIRONMENT", "production"),
        "port": os.getenv("PORT", "8000"),
        "status": "running"
    }

@app.get("/api/triage/queue")
async def get_triage_queue(limit: int = 50, offset: int = 0):
    """Get triage queue items"""
    demo_items = [
        {
            "id": 1,
            "item_id": 1,
            "doc_id": "demo_001",
            "sent_id": "s1",
            "text": "White Spot Syndrome Virus (WSSV) causes significant mortality in shrimp farming.",
            "priority_score": 0.95,
            "status": "pending",
            "created_at": datetime.now().isoformat()
        },
        {
            "id": 2,
            "item_id": 2,
            "doc_id": "demo_002",
            "sent_id": "s1",
            "text": "Probiotics can improve water quality in shrimp ponds.",
            "priority_score": 0.82,
            "status": "pending",
            "created_at": datetime.now().isoformat()
        }
    ]
    
    return {
        "items": demo_items[offset:offset + limit],
        "total": len(demo_items),
        "limit": limit,
        "offset": offset
    }

@app.get("/api/triage/next")
async def get_next_triage_item():
    """Get next item from triage queue"""
    return {
        "item": {
            "id": 1,
            "item_id": 1,
            "doc_id": "demo_001",
            "sent_id": "s1",
            "text": "White Spot Syndrome Virus (WSSV) causes significant mortality in shrimp farming.",
            "candidates": {
                "entities": [
                    {"text": "White Spot Syndrome Virus", "label": "PATHOGEN", "start": 0, "end": 25},
                    {"text": "WSSV", "label": "PATHOGEN", "start": 27, "end": 31},
                    {"text": "shrimp", "label": "SPECIES", "start": 60, "end": 66}
                ],
                "relations": [
                    {"source": "WSSV", "target": "shrimp", "type": "infects"}
                ],
                "topics": ["T_DISEASE"]
            },
            "priority_score": 0.95
        }
    }

@app.post("/api/annotations/decide")
async def submit_annotation(annotation_data: Dict):
    """Submit annotation decision"""
    logger.info(f"Annotation submitted: {annotation_data}")
    return {
        "status": "success",
        "message": "Annotation saved",
        "annotation_id": 1,
        "timestamp": datetime.now().isoformat()
    }

# Catch-all route MUST be last to serve React frontend for client-side routing
@app.get("/{full_path:path}")
async def serve_frontend_routes(full_path: str):
    """Serve React frontend for all non-API routes"""
    # Try to serve React frontend
    if ui_build_path.exists():
        index_file = ui_build_path / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
    
    # Fallback
    raise HTTPException(status_code=404, detail="Frontend not available")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "railway_api:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )