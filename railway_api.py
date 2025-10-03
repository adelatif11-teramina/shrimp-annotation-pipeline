#!/usr/bin/env python3
"""
Minimal Railway-compatible API for Shrimp Annotation Pipeline
"""

import os
import logging
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
            "docs": "/docs",
            "openapi": "/openapi.json"
        }
    }

@app.get("/{full_path:path}")
async def serve_frontend_routes(full_path: str):
    """Serve React frontend for all non-API routes"""
    # Skip API routes
    if full_path.startswith(("health", "docs", "openapi", "candidates", "documents", "statistics", "ready")):
        raise HTTPException(status_code=404, detail="API endpoint not found")
    
    # Try to serve React frontend
    if ui_build_path.exists():
        index_file = ui_build_path / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
    
    # Fallback
    raise HTTPException(status_code=404, detail="Frontend not available")

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

@app.post("/documents/ingest")
async def ingest_document(doc_request: DocumentRequest):
    """Ingest a document for annotation (minimal implementation)"""
    logger.info(f"Document ingestion requested: {doc_request.doc_id}")
    
    # Minimal implementation for Railway demo
    return {
        "doc_id": doc_request.doc_id,
        "sentence_count": len(doc_request.text.split('.')),
        "message": "Document received successfully (Railway demo mode)",
        "status": "processed"
    }

@app.post("/candidates/generate")
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

@app.get("/statistics/overview")
async def get_statistics():
    """Get system statistics"""
    return {
        "timestamp": datetime.now().isoformat(),
        "mode": "railway_demo",
        "environment": os.getenv("ENVIRONMENT", "production"),
        "port": os.getenv("PORT", "8000"),
        "status": "running"
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "railway_api:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )