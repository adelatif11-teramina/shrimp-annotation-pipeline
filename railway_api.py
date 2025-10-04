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
    """Ingest a document for annotation (store in memory and create triage items)"""
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
    
    # Create triage queue items for each sentence
    triage_items_created = 0
    for i, sentence in enumerate(sentences):
        if sentence.strip():  # Only process non-empty sentences
            # Calculate priority score based on content
            priority_score = 0.5  # Default priority
            if any(keyword in sentence.lower() for keyword in ['disease', 'virus', 'pathogen', 'mortality', 'infection']):
                priority_score = 0.9
            elif any(keyword in sentence.lower() for keyword in ['treatment', 'prevention', 'antibiotic', 'vaccine']):
                priority_score = 0.8
            elif any(keyword in sentence.lower() for keyword in ['shrimp', 'aquaculture', 'farming', 'pond']):
                priority_score = 0.7
            
            triage_item = {
                "id": len(triage_queue_store) + 1,
                "item_id": len(triage_queue_store) + 1,
                "doc_id": doc_request.doc_id,
                "sent_id": f"s{i+1}",
                "sentence_id": i + 1,
                "text": sentence,
                "priority_score": priority_score,
                "priority_level": "high" if priority_score > 0.8 else "medium" if priority_score > 0.6 else "low",
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "candidates": {
                    "entities": [],
                    "relations": [],
                    "topics": []
                }
            }
            
            triage_queue_store.append(triage_item)
            triage_items_created += 1
    
    logger.info(f"Document stored successfully: {doc_request.doc_id} (Total documents: {len(documents_store)})")
    logger.info(f"Created {triage_items_created} triage queue items (Total queue items: {len(triage_queue_store)})")
    
    # Create response
    return {
        "doc_id": doc_request.doc_id,
        "title": document["title"],
        "source": doc_request.source,
        "sentence_count": len(sentences),
        "sentences": len(sentences),
        "triage_items_created": triage_items_created,
        "message": f"Document ingested with {triage_items_created} sentences added to triage queue",
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
async def get_triage_queue(
    limit: int = 50, 
    offset: int = 0, 
    status: Optional[str] = None,
    sort_by: Optional[str] = None
):
    """Get triage queue items from storage"""
    logger.info(f"Triage queue requested: limit={limit}, offset={offset}, status={status}, sort_by={sort_by} (Storage: {len(triage_queue_store)} items)")
    
    # Start with all stored triage items
    all_items = triage_queue_store.copy()
    
    # If no items in storage, return demo items
    if not all_items:
        logger.info("No triage items in storage, returning demo items")
        demo_items = [
            {
                "id": 1,
                "item_id": 1,
                "doc_id": "demo_001",
                "sent_id": "s1",
                "text": "White Spot Syndrome Virus (WSSV) causes significant mortality in shrimp farming.",
                "priority_score": 0.95,
                "priority_level": "high",
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "candidates": {
                    "entities": [
                        {"text": "White Spot Syndrome Virus", "label": "PATHOGEN", "start": 0, "end": 25},
                        {"text": "WSSV", "label": "PATHOGEN", "start": 27, "end": 31},
                        {"text": "shrimp", "label": "SPECIES", "start": 60, "end": 66}
                    ],
                    "relations": [],
                    "topics": ["T_DISEASE"]
                }
            },
            {
                "id": 2,
                "item_id": 2,
                "doc_id": "demo_002",
                "sent_id": "s1",
                "text": "Probiotics can improve water quality in shrimp ponds.",
                "priority_score": 0.82,
                "priority_level": "high",
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "candidates": {
                    "entities": [
                        {"text": "Probiotics", "label": "CHEMICAL", "start": 0, "end": 10},
                        {"text": "shrimp", "label": "SPECIES", "start": 42, "end": 48}
                    ],
                    "relations": [],
                    "topics": ["T_TREATMENT"]
                }
            }
        ]
        all_items = demo_items
    
    # Apply status filter if provided (ignore "undefined" values from frontend)
    if status and status not in ["all", "undefined", "null"]:
        all_items = [item for item in all_items if item.get("status") == status]
        logger.info(f"Filtered by status '{status}': {len(all_items)} items")
    
    # Apply sorting
    if sort_by == "priority":
        all_items.sort(key=lambda x: -x.get("priority_score", 0))
    elif sort_by == "created_at":
        all_items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    else:
        # Default sort: by priority score (highest first) then by created_at
        all_items.sort(key=lambda x: (-x.get("priority_score", 0), x.get("created_at", "")))
    
    # Apply pagination
    paginated_items = all_items[offset:offset + limit]
    
    logger.info(f"Returning {len(paginated_items)} triage items (total: {len(all_items)})")
    
    return {
        "items": paginated_items,
        "total": len(all_items),
        "limit": limit,
        "offset": offset
    }

@app.get("/api/triage/next")
async def get_next_triage_item():
    """Get next item from triage queue"""
    logger.info(f"Next triage item requested (Storage: {len(triage_queue_store)} items)")
    
    # Find highest priority pending item
    pending_items = [item for item in triage_queue_store if item.get("status") == "pending"]
    
    if pending_items:
        # Sort by priority score (highest first)
        pending_items.sort(key=lambda x: -x.get("priority_score", 0))
        next_item = pending_items[0]
        logger.info(f"Returning next item: {next_item['doc_id']}/{next_item['sent_id']}")
        return {"item": next_item}
    
    # Fallback to demo item if no pending items
    logger.info("No pending items in storage, returning demo item")
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
            "priority_score": 0.95,
            "priority_level": "high",
            "status": "pending",
            "created_at": datetime.now().isoformat()
        }
    }

@app.post("/api/annotations/decide")
async def submit_annotation(annotation_data: Dict):
    """Submit annotation decision and return next item"""
    global triage_queue_store, annotations_store
    
    logger.info(f"Annotation submitted: {annotation_data}")
    
    # Extract key fields
    item_id = annotation_data.get("item_id") or annotation_data.get("candidate_id")
    decision = annotation_data.get("decision", "accept")
    user_id = annotation_data.get("user_id", 1)
    
    # Find and update the triage item
    current_item = None
    for item in triage_queue_store:
        if item.get("item_id") == item_id or item.get("id") == item_id:
            current_item = item
            item["status"] = "completed"
            item["decision"] = decision
            item["completed_at"] = datetime.now().isoformat()
            break
    
    if not current_item:
        logger.warning(f"Triage item not found for item_id: {item_id}")
    
    # Store the annotation
    annotation = {
        "id": len(annotations_store) + 1,
        "item_id": item_id,
        "user_id": user_id,
        "decision": decision,
        "entities": annotation_data.get("entities", []),
        "relations": annotation_data.get("relations", []),
        "topics": annotation_data.get("topics", []),
        "confidence": annotation_data.get("confidence", 0.8),
        "notes": annotation_data.get("notes", ""),
        "created_at": datetime.now().isoformat()
    }
    annotations_store.append(annotation)
    
    # Find next pending item
    next_item = None
    pending_items = [item for item in triage_queue_store if item.get("status") == "pending"]
    
    if pending_items:
        # Sort by priority score (highest first)
        pending_items.sort(key=lambda x: -x.get("priority_score", 0))
        next_item = pending_items[0]
        logger.info(f"Next item: {next_item['doc_id']}/{next_item['sent_id']}")
    else:
        logger.info("No more pending items in queue")
    
    logger.info(f"Annotation saved for item {item_id}, decision: {decision}")
    
    return {
        "status": "success",
        "message": f"Annotation {decision}ed successfully",
        "annotation_id": annotation["id"],
        "next_item": next_item,
        "queue_remaining": len(pending_items),
        "timestamp": datetime.now().isoformat()
    }

@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and all its associated triage items"""
    global documents_store, triage_queue_store
    
    logger.info(f"Delete requested for document: {doc_id}")
    
    # Find and remove the document
    initial_doc_count = len(documents_store)
    documents_store = [doc for doc in documents_store if doc.get("doc_id") != doc_id]
    docs_deleted = initial_doc_count - len(documents_store)
    
    # Remove all associated triage items
    initial_triage_count = len(triage_queue_store)
    triage_queue_store = [item for item in triage_queue_store if item.get("doc_id") != doc_id]
    items_deleted = initial_triage_count - len(triage_queue_store)
    
    logger.info(f"Deleted document {doc_id}: {docs_deleted} document(s), {items_deleted} triage item(s) removed")
    
    if docs_deleted > 0:
        return {
            "status": "success",
            "message": f"Document {doc_id} deleted successfully",
            "documents_deleted": docs_deleted,
            "triage_items_deleted": items_deleted
        }
    else:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

@app.delete("/api/triage/clear")
async def clear_triage_queue():
    """Clear all items from the triage queue"""
    global triage_queue_store
    
    items_count = len(triage_queue_store)
    triage_queue_store = []
    
    logger.info(f"Cleared triage queue: {items_count} items removed")
    
    return {
        "status": "success",
        "message": f"Triage queue cleared successfully",
        "items_deleted": items_count
    }

@app.delete("/api/triage/document/{doc_id}")
async def clear_document_triage_items(doc_id: str):
    """Clear triage items for a specific document"""
    global triage_queue_store
    
    initial_count = len(triage_queue_store)
    triage_queue_store = [item for item in triage_queue_store if item.get("doc_id") != doc_id]
    items_deleted = initial_count - len(triage_queue_store)
    
    logger.info(f"Cleared triage items for document {doc_id}: {items_deleted} items removed")
    
    return {
        "status": "success",
        "message": f"Triage items for document {doc_id} cleared",
        "items_deleted": items_deleted
    }

@app.post("/api/reset-all")
async def reset_all_data():
    """Reset all data stores (useful for starting fresh)"""
    global documents_store, triage_queue_store, annotations_store
    
    doc_count = len(documents_store)
    triage_count = len(triage_queue_store)
    anno_count = len(annotations_store)
    
    documents_store = []
    triage_queue_store = []
    annotations_store = []
    
    logger.info(f"Reset all data stores: {doc_count} docs, {triage_count} triage items, {anno_count} annotations")
    
    return {
        "status": "success",
        "message": "All data stores reset successfully",
        "deleted": {
            "documents": doc_count,
            "triage_items": triage_count,
            "annotations": anno_count
        }
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