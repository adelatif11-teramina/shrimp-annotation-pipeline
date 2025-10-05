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
    # Serve static files only if directory exists
    static_path = ui_build_path / "static"
    if static_path.exists():
        app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
        logger.info("Static files mounted successfully")
    else:
        logger.warning(f"Static directory not found: {static_path}")
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

# Health check endpoints for Railway
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for Railway"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        message="Shrimp Annotation Pipeline API is running on Railway"
    )

@app.get("/api/health", response_model=HealthResponse)
async def api_health_check():
    """API health check endpoint for Railway"""
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
            return FileResponse(str(index_file), media_type="text/html")
    
    # Fallback to API info
    return {
        "message": "🦐 Shrimp Annotation Pipeline API",
        "version": "1.0.0", 
        "status": "running",
        "frontend": "not_built" if not ui_build_path.exists() else "index_missing",
        "frontend_path": str(ui_build_path),
        "index_exists": (ui_build_path / "index.html").exists() if ui_build_path.exists() else False,
        "endpoints": {
            "health": "/health",
            "api_docs": "/docs",
            "api_base": "/api",
            "documents": "/api/documents",
            "annotations": "/api/annotations",
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
        "status": "pending",
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
                "sentence_count": 45,
                "status": "pending"
            },
            {
                "id": 2,
                "doc_id": "demo_002", 
                "title": "WSSV Prevention Strategies",
                "source": "upload",
                "created_at": "2024-01-14T15:30:00Z",
                "sentence_count": 32,
                "status": "pending"
            },
            {
                "id": 3,
                "doc_id": "demo_003",
                "title": "Aquaculture Best Practices",
                "source": "manual",
                "created_at": "2024-01-13T09:15:00Z",
                "sentence_count": 68,
                "status": "pending"
            }
        ]
        all_documents = demo_documents
    
    # Apply search filter if provided
    if search:
        all_documents = [
            doc for doc in all_documents 
            if search.lower() in doc["title"].lower() or search.lower() in doc["doc_id"].lower()
        ]
    
    # Add processing statistics to each document
    for doc in all_documents:
        doc_id = doc.get("doc_id")
        if doc_id:
            # Count triage items and annotations for this document
            doc_triage_items = [item for item in triage_queue_store if item.get("doc_id") == doc_id]
            doc_annotations = [anno for anno in annotations_store if str(anno.get("item_id", "")).startswith(str(doc_id))]
            
            total_sentences = doc.get("sentence_count", 0)
            completed_sentences = len([item for item in doc_triage_items if item.get("status") == "completed"])
            progress_percentage = round(completed_sentences / total_sentences * 100, 1) if total_sentences > 0 else 0
            
            doc["progress"] = {
                "total_sentences": total_sentences,
                "completed_sentences": completed_sentences,
                "progress_percentage": progress_percentage,
                "remaining_sentences": total_sentences - completed_sentences
            }
    
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
    
    # If no items in storage at all, create demo items (one time only)
    if len(triage_queue_store) == 0:
        logger.info("No triage items in storage, creating persistent demo items")
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
        
        # Store demo items persistently so they can be annotated
        triage_queue_store.extend(demo_items)
        all_items = triage_queue_store.copy()
    
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
    global triage_queue_store
    
    logger.info(f"Next triage item requested (Storage: {len(triage_queue_store)} items)")
    
    # If no items at all, create demo items first
    if len(triage_queue_store) == 0:
        logger.info("Creating demo items for next item request")
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
                    "relations": [
                        {"source": "WSSV", "target": "shrimp", "type": "infects"}
                    ],
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
        triage_queue_store.extend(demo_items)
    
    # Find highest priority pending item
    pending_items = [item for item in triage_queue_store if item.get("status") == "pending"]
    
    if pending_items:
        # Sort by priority score (highest first)
        pending_items.sort(key=lambda x: -x.get("priority_score", 0))
        next_item = pending_items[0]
        logger.info(f"Returning next item: {next_item['doc_id']}/{next_item['sent_id']}")
        return {"item": next_item}
    
    # No pending items at all
    logger.info("No pending items available")
    return {"item": None}

@app.post("/api/annotations/decide")
async def submit_annotation(annotation_data: Dict):
    """Submit annotation decision and return next item"""
    global triage_queue_store, annotations_store, documents_store
    
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
    
    # Store the annotation with full context
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
        "created_at": datetime.now().isoformat(),
        # Add context fields
        "sentence_text": current_item.get("text", "") if current_item else "",
        "doc_id": current_item.get("doc_id", "") if current_item else "",
        "sent_id": current_item.get("sent_id", "") if current_item else "",
        "document_title": "",  # Will be populated below
        "priority_score": current_item.get("priority_score", 0.0) if current_item else 0.0,
        "time_spent": annotation_data.get("time_spent", 0),
        "source": current_item.get("source", "unknown") if current_item else "unknown"
    }
    
    # Add document title for context
    if current_item and current_item.get("doc_id"):
        for doc in documents_store:
            if doc.get("doc_id") == current_item.get("doc_id"):
                annotation["document_title"] = doc.get("title", "")
                break
    annotations_store.append(annotation)
    
    # Check if document is fully completed
    if current_item:
        doc_id = current_item.get("doc_id")
        if doc_id:
            # Count total items and completed items for this document
            doc_items = [item for item in triage_queue_store if item.get("doc_id") == doc_id]
            completed_items = [item for item in doc_items if item.get("status") == "completed"]
            
            logger.info(f"Document {doc_id} progress: {len(completed_items)}/{len(doc_items)} items completed")
            
            # If all items are completed, mark document as processed and remove from queue
            if len(completed_items) == len(doc_items):
                logger.info(f"Document {doc_id} fully completed! Removing from queue and marking as processed")
                
                # Remove all triage items for this document
                triage_queue_store = [item for item in triage_queue_store if item.get("doc_id") != doc_id]
                
                # Mark document as processed
                for doc in documents_store:
                    if doc.get("doc_id") == doc_id:
                        doc["status"] = "processed"
                        doc["completed_at"] = datetime.now().isoformat()
                        doc["total_annotations"] = len(completed_items)
                        break
                
                logger.info(f"Document {doc_id} processing complete: {len(completed_items)} annotations saved, removed from queue")
    
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

@app.get("/api/documents/completed")
async def get_completed_documents():
    """Get list of completed/processed documents"""
    completed_docs = [doc for doc in documents_store if doc.get("status") == "processed"]
    
    # Calculate statistics for each completed document
    for doc in completed_docs:
        doc_id = doc.get("doc_id")
        doc_annotations = [anno for anno in annotations_store if anno.get("item_id", "").startswith(doc_id)]
        
        # Count decision types
        accepted = len([anno for anno in doc_annotations if anno.get("decision") == "accept"])
        rejected = len([anno for anno in doc_annotations if anno.get("decision") == "reject"])
        skipped = len([anno for anno in doc_annotations if anno.get("decision") == "skip"])
        
        doc["annotation_stats"] = {
            "total_annotations": len(doc_annotations),
            "accepted": accepted,
            "rejected": rejected,
            "skipped": skipped,
            "acceptance_rate": round(accepted / len(doc_annotations) * 100, 1) if doc_annotations else 0
        }
    
    logger.info(f"Returning {len(completed_docs)} completed documents")
    
    return {
        "documents": completed_docs,
        "total": len(completed_docs)
    }

@app.get("/api/statistics/documents")
async def get_document_statistics():
    """Get document processing statistics"""
    total_docs = len(documents_store)
    processed_docs = len([doc for doc in documents_store if doc.get("status") == "processed"])
    pending_docs = total_docs - processed_docs
    
    total_annotations = len(annotations_store)
    accepted_annotations = len([anno for anno in annotations_store if anno.get("decision") == "accept"])
    rejected_annotations = len([anno for anno in annotations_store if anno.get("decision") == "reject"])
    
    return {
        "documents": {
            "total": total_docs,
            "processed": processed_docs,
            "pending": pending_docs,
            "completion_rate": round(processed_docs / total_docs * 100, 1) if total_docs > 0 else 0
        },
        "annotations": {
            "total": total_annotations,
            "accepted": accepted_annotations,
            "rejected": rejected_annotations,
            "acceptance_rate": round(accepted_annotations / total_annotations * 100, 1) if total_annotations > 0 else 0
        },
        "queue": {
            "pending_items": len([item for item in triage_queue_store if item.get("status") == "pending"]),
            "total_items": len(triage_queue_store)
        }
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

@app.get("/api/annotations")
async def get_annotations(
    decision: Optional[str] = None,
    doc_id: Optional[str] = None,
    user_id: Optional[int] = None,
    limit: int = 50,
    offset: int = 0,
    sort_by: Optional[str] = "created_at"
):
    """Get annotations with filtering capabilities"""
    logger.info(f"Annotations requested: decision={decision}, doc_id={doc_id}, user_id={user_id}, limit={limit}, offset={offset}")
    
    # Start with all annotations
    filtered_annotations = annotations_store.copy()
    
    # Apply filters
    if decision and decision != "all":
        filtered_annotations = [anno for anno in filtered_annotations if anno.get("decision") == decision]
    
    if doc_id:
        filtered_annotations = [anno for anno in filtered_annotations if anno.get("doc_id") == doc_id]
    
    if user_id:
        filtered_annotations = [anno for anno in filtered_annotations if anno.get("user_id") == user_id]
    
    # Apply sorting
    if sort_by == "created_at":
        filtered_annotations.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    elif sort_by == "priority_score":
        filtered_annotations.sort(key=lambda x: x.get("priority_score", 0), reverse=True)
    elif sort_by == "confidence":
        filtered_annotations.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    elif sort_by == "time_spent":
        filtered_annotations.sort(key=lambda x: x.get("time_spent", 0), reverse=True)
    
    # Apply pagination
    total_count = len(filtered_annotations)
    paginated_annotations = filtered_annotations[offset:offset + limit]
    
    # Add derived fields for better display
    for anno in paginated_annotations:
        # Truncate long sentence text for list view
        if anno.get("sentence_text"):
            anno["sentence_preview"] = anno["sentence_text"][:100] + "..." if len(anno["sentence_text"]) > 100 else anno["sentence_text"]
        
        # Add entity/relation counts
        anno["entity_count"] = len(anno.get("entities", []))
        anno["relation_count"] = len(anno.get("relations", []))
        anno["topic_count"] = len(anno.get("topics", []))
        
        # Format time spent
        time_spent = anno.get("time_spent", 0)
        if time_spent > 0:
            if time_spent >= 60:
                anno["time_spent_formatted"] = f"{time_spent // 60}m {time_spent % 60}s"
            else:
                anno["time_spent_formatted"] = f"{time_spent}s"
        else:
            anno["time_spent_formatted"] = "Unknown"
    
    logger.info(f"Returning {len(paginated_annotations)} annotations (total: {total_count})")
    
    return {
        "annotations": paginated_annotations,
        "total": total_count,
        "limit": limit,
        "offset": offset,
        "filters": {
            "decision": decision,
            "doc_id": doc_id,
            "user_id": user_id,
            "sort_by": sort_by
        }
    }

@app.get("/api/annotations/statistics")
async def get_annotation_statistics():
    """Get comprehensive annotation statistics"""
    logger.info("Annotation statistics requested")
    
    total_annotations = len(annotations_store)
    
    if total_annotations == 0:
        return {
            "summary": {
                "total_annotations": 0,
                "accepted": 0,
                "rejected": 0,
                "skipped": 0,
                "modified": 0,
                "acceptance_rate": 0,
                "avg_time_per_annotation": 0
            },
            "by_decision": [],
            "by_document": [],
            "by_user": [],
            "by_confidence": [],
            "by_date": [],
            "entity_stats": [],
            "relation_stats": [],
            "topic_stats": []
        }
    
    # Decision breakdown
    decisions = {}
    for anno in annotations_store:
        decision = anno.get("decision", "unknown")
        decisions[decision] = decisions.get(decision, 0) + 1
    
    accepted = decisions.get("accept", 0)
    rejected = decisions.get("reject", 0)
    skipped = decisions.get("skip", 0)
    modified = decisions.get("modified", 0)
    
    # Time statistics
    time_values = [anno.get("time_spent", 0) for anno in annotations_store if anno.get("time_spent", 0) > 0]
    avg_time = sum(time_values) / len(time_values) if time_values else 0
    
    # Document breakdown
    doc_stats = {}
    for anno in annotations_store:
        doc_id = anno.get("doc_id", "unknown")
        if doc_id not in doc_stats:
            doc_stats[doc_id] = {"total": 0, "accepted": 0, "rejected": 0, "skipped": 0}
        doc_stats[doc_id]["total"] += 1
        if anno.get("decision") == "accept":
            doc_stats[doc_id]["accepted"] += 1
        elif anno.get("decision") == "reject":
            doc_stats[doc_id]["rejected"] += 1
        elif anno.get("decision") == "skip":
            doc_stats[doc_id]["skipped"] += 1
    
    # Add document titles
    doc_breakdown = []
    for doc_id, stats in doc_stats.items():
        doc_title = doc_id
        for doc in documents_store:
            if doc.get("doc_id") == doc_id:
                doc_title = doc.get("title", doc_id)
                break
        
        acceptance_rate = round(stats["accepted"] / stats["total"] * 100, 1) if stats["total"] > 0 else 0
        doc_breakdown.append({
            "doc_id": doc_id,
            "doc_title": doc_title,
            "total": stats["total"],
            "accepted": stats["accepted"],
            "rejected": stats["rejected"],
            "skipped": stats["skipped"],
            "acceptance_rate": acceptance_rate
        })
    
    # User breakdown
    user_stats = {}
    for anno in annotations_store:
        user_id = anno.get("user_id", "unknown")
        if user_id not in user_stats:
            user_stats[user_id] = {"total": 0, "accepted": 0, "time_spent": 0}
        user_stats[user_id]["total"] += 1
        if anno.get("decision") == "accept":
            user_stats[user_id]["accepted"] += 1
        user_stats[user_id]["time_spent"] += anno.get("time_spent", 0)
    
    user_breakdown = []
    for user_id, stats in user_stats.items():
        acceptance_rate = round(stats["accepted"] / stats["total"] * 100, 1) if stats["total"] > 0 else 0
        avg_time_per_user = round(stats["time_spent"] / stats["total"], 1) if stats["total"] > 0 else 0
        user_breakdown.append({
            "user_id": user_id,
            "total": stats["total"],
            "accepted": stats["accepted"],
            "acceptance_rate": acceptance_rate,
            "avg_time_per_annotation": avg_time_per_user
        })
    
    # Confidence breakdown
    confidence_ranges = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
    for anno in annotations_store:
        conf = anno.get("confidence", 0)
        if conf >= 0.8:
            confidence_ranges["high"] += 1
        elif conf >= 0.6:
            confidence_ranges["medium"] += 1
        elif conf > 0:
            confidence_ranges["low"] += 1
        else:
            confidence_ranges["unknown"] += 1
    
    # Entity type statistics
    entity_stats = {}
    for anno in annotations_store:
        for entity in anno.get("entities", []):
            entity_type = entity.get("label", "unknown")
            if entity_type not in entity_stats:
                entity_stats[entity_type] = 0
            entity_stats[entity_type] += 1
    
    entity_breakdown = [{"type": k, "count": v} for k, v in sorted(entity_stats.items(), key=lambda x: x[1], reverse=True)]
    
    # Relation type statistics
    relation_stats = {}
    for anno in annotations_store:
        for relation in anno.get("relations", []):
            rel_type = relation.get("type", "unknown")
            if rel_type not in relation_stats:
                relation_stats[rel_type] = 0
            relation_stats[rel_type] += 1
    
    relation_breakdown = [{"type": k, "count": v} for k, v in sorted(relation_stats.items(), key=lambda x: x[1], reverse=True)]
    
    # Topic statistics
    topic_stats = {}
    for anno in annotations_store:
        for topic in anno.get("topics", []):
            topic_name = topic if isinstance(topic, str) else topic.get("name", "unknown")
            if topic_name not in topic_stats:
                topic_stats[topic_name] = 0
            topic_stats[topic_name] += 1
    
    topic_breakdown = [{"topic": k, "count": v} for k, v in sorted(topic_stats.items(), key=lambda x: x[1], reverse=True)]
    
    # Date breakdown (by day)
    from collections import defaultdict
    date_stats = defaultdict(lambda: {"total": 0, "accepted": 0})
    for anno in annotations_store:
        created_at = anno.get("created_at", "")
        if created_at:
            date = created_at.split("T")[0]  # Extract date part
            date_stats[date]["total"] += 1
            if anno.get("decision") == "accept":
                date_stats[date]["accepted"] += 1
    
    date_breakdown = []
    for date, stats in sorted(date_stats.items()):
        acceptance_rate = round(stats["accepted"] / stats["total"] * 100, 1) if stats["total"] > 0 else 0
        date_breakdown.append({
            "date": date,
            "total": stats["total"],
            "accepted": stats["accepted"],
            "acceptance_rate": acceptance_rate
        })
    
    return {
        "summary": {
            "total_annotations": total_annotations,
            "accepted": accepted,
            "rejected": rejected,
            "skipped": skipped,
            "modified": modified,
            "acceptance_rate": round(accepted / total_annotations * 100, 1) if total_annotations > 0 else 0,
            "avg_time_per_annotation": round(avg_time, 1)
        },
        "by_decision": [{"decision": k, "count": v, "percentage": round(v / total_annotations * 100, 1)} for k, v in decisions.items()],
        "by_document": doc_breakdown,
        "by_user": user_breakdown,
        "by_confidence": [{"level": k, "count": v, "percentage": round(v / total_annotations * 100, 1)} for k, v in confidence_ranges.items()],
        "by_date": date_breakdown,
        "entity_stats": entity_breakdown,
        "relation_stats": relation_breakdown,
        "topic_stats": topic_breakdown
    }

@app.get("/api/annotations/export")
async def export_annotations(
    format: str = "json",
    decision: Optional[str] = None,
    doc_id: Optional[str] = None,
    user_id: Optional[int] = None
):
    """Export annotations in various formats"""
    logger.info(f"Export requested: format={format}, decision={decision}, doc_id={doc_id}, user_id={user_id}")
    
    # Filter annotations based on parameters
    filtered_annotations = annotations_store.copy()
    
    if decision and decision != "all":
        filtered_annotations = [anno for anno in filtered_annotations if anno.get("decision") == decision]
    
    if doc_id:
        filtered_annotations = [anno for anno in filtered_annotations if anno.get("doc_id") == doc_id]
    
    if user_id:
        filtered_annotations = [anno for anno in filtered_annotations if anno.get("user_id") == user_id]
    
    # Sort by creation time
    filtered_annotations.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    
    if format.lower() == "json":
        from fastapi.responses import JSONResponse
        
        export_data = {
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "total_annotations": len(filtered_annotations),
                "filters_applied": {
                    "decision": decision,
                    "doc_id": doc_id,
                    "user_id": user_id
                },
                "format": "json"
            },
            "annotations": filtered_annotations
        }
        
        return JSONResponse(
            content=export_data,
            headers={
                "Content-Disposition": f"attachment; filename=annotations_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            }
        )
    
    elif format.lower() == "csv":
        import csv
        import io
        from fastapi.responses import StreamingResponse
        
        # Prepare CSV data
        output = io.StringIO()
        writer = csv.writer(output)
        
        # CSV headers
        headers = [
            "annotation_id", "item_id", "user_id", "decision", "doc_id", "sent_id",
            "sentence_text", "document_title", "priority_score", "confidence",
            "entity_count", "relation_count", "topic_count", "time_spent",
            "created_at", "notes", "source"
        ]
        writer.writerow(headers)
        
        # CSV rows
        for anno in filtered_annotations:
            writer.writerow([
                anno.get("id", ""),
                anno.get("item_id", ""),
                anno.get("user_id", ""),
                anno.get("decision", ""),
                anno.get("doc_id", ""),
                anno.get("sent_id", ""),
                anno.get("sentence_text", "").replace("\n", " ").replace("\r", " ")[:200],  # Truncate and clean
                anno.get("document_title", ""),
                anno.get("priority_score", ""),
                anno.get("confidence", ""),
                len(anno.get("entities", [])),
                len(anno.get("relations", [])),
                len(anno.get("topics", [])),
                anno.get("time_spent", ""),
                anno.get("created_at", ""),
                anno.get("notes", "").replace("\n", " ").replace("\r", " ")[:100],  # Truncate and clean
                anno.get("source", "")
            ])
        
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode('utf-8')),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=annotations_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            }
        )
    
    elif format.lower() == "scibert":
        # Export in SciBERT training format (CoNLL-like)
        import io
        from fastapi.responses import StreamingResponse
        
        output = io.StringIO()
        
        # Group annotations by document
        doc_annotations = {}
        for anno in filtered_annotations:
            doc_id = anno.get("doc_id", "unknown")
            if doc_id not in doc_annotations:
                doc_annotations[doc_id] = []
            doc_annotations[doc_id].append(anno)
        
        # Generate SciBERT format
        for doc_id, annotations in doc_annotations.items():
            for anno in annotations:
                if anno.get("decision") == "accept" and anno.get("entities"):
                    sentence_text = anno.get("sentence_text", "")
                    if sentence_text:
                        # Convert to token-label format
                        tokens = sentence_text.split()
                        labels = ["O"] * len(tokens)
                        
                        # Map entities to BIO labels
                        for entity in anno.get("entities", []):
                            entity_text = entity.get("text", "")
                            entity_label = entity.get("label", "")
                            
                            if entity_text and entity_label:
                                # Simple token matching (this could be improved)
                                entity_tokens = entity_text.split()
                                for i, token in enumerate(tokens):
                                    if token.startswith(entity_tokens[0] if entity_tokens else ""):
                                        labels[i] = f"B-{entity_label}"
                                        # Mark continuation tokens
                                        for j in range(1, len(entity_tokens)):
                                            if i + j < len(labels):
                                                labels[i + j] = f"I-{entity_label}"
                        
                        # Write tokens and labels
                        for token, label in zip(tokens, labels):
                            output.write(f"{token}\t{label}\n")
                        output.write("\n")  # Sentence separator
        
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode('utf-8')),
            media_type="text/plain",
            headers={
                "Content-Disposition": f"attachment; filename=annotations_scibert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            }
        )
    
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported export format: {format}. Supported formats: json, csv, scibert")

@app.get("/api/annotations/{annotation_id}")
async def get_annotation_detail(annotation_id: int):
    """Get detailed annotation information by ID"""
    logger.info(f"Annotation detail requested: {annotation_id}")
    
    # Find annotation by ID
    annotation = None
    for anno in annotations_store:
        if anno.get("id") == annotation_id:
            annotation = anno.copy()
            break
    
    if not annotation:
        raise HTTPException(status_code=404, detail=f"Annotation {annotation_id} not found")
    
    # Add additional context information
    doc_id = annotation.get("doc_id")
    if doc_id:
        # Find document information
        for doc in documents_store:
            if doc.get("doc_id") == doc_id:
                annotation["document_info"] = {
                    "title": doc.get("title", ""),
                    "source": doc.get("source", ""),
                    "created_at": doc.get("created_at", ""),
                    "sentence_count": doc.get("sentence_count", 0),
                    "status": doc.get("status", "")
                }
                break
        
        # Find related annotations from same document
        related_annotations = [
            {
                "id": anno.get("id"),
                "sent_id": anno.get("sent_id"),
                "decision": anno.get("decision"),
                "confidence": anno.get("confidence"),
                "created_at": anno.get("created_at")
            }
            for anno in annotations_store 
            if anno.get("doc_id") == doc_id and anno.get("id") != annotation_id
        ]
        annotation["related_annotations"] = related_annotations[:10]  # Limit to 10
    
    # Add formatted fields for display
    annotation["created_at_formatted"] = annotation.get("created_at", "").replace("T", " ").split(".")[0] if annotation.get("created_at") else "Unknown"
    
    # Format time spent
    time_spent = annotation.get("time_spent", 0)
    if time_spent > 0:
        if time_spent >= 60:
            annotation["time_spent_formatted"] = f"{time_spent // 60}m {time_spent % 60}s"
        else:
            annotation["time_spent_formatted"] = f"{time_spent}s"
    else:
        annotation["time_spent_formatted"] = "Unknown"
    
    # Add entity/relation/topic counts
    annotation["entity_count"] = len(annotation.get("entities", []))
    annotation["relation_count"] = len(annotation.get("relations", []))
    annotation["topic_count"] = len(annotation.get("topics", []))
    
    logger.info(f"Returning annotation detail for ID {annotation_id}")
    
    return {
        "annotation": annotation,
        "context": {
            "doc_id": doc_id,
            "related_count": len(annotation.get("related_annotations", [])),
            "has_document_info": "document_info" in annotation
        }
    }

@app.post("/api/demo/restart")
async def restart_demo():
    """Restart demo with fresh demo items for testing"""
    global triage_queue_store
    
    # Remove only demo items
    initial_count = len(triage_queue_store)
    triage_queue_store = [item for item in triage_queue_store if not item.get("doc_id", "").startswith("demo_")]
    demo_removed = initial_count - len(triage_queue_store)
    
    logger.info(f"Restarted demo: removed {demo_removed} demo items")
    
    return {
        "status": "success",
        "message": f"Demo restarted - removed {demo_removed} demo items",
        "remaining_items": len(triage_queue_store)
    }

# Catch-all route MUST be last to serve React frontend for client-side routing
@app.get("/{full_path:path}")
async def serve_frontend_routes(full_path: str):
    """Serve React frontend for all non-API routes"""
    # Don't serve frontend for API routes
    if full_path.startswith("api/") or full_path.startswith("docs") or full_path.startswith("health"):
        raise HTTPException(status_code=404, detail="API endpoint not found")
    
    # Try to serve React frontend
    if ui_build_path.exists():
        index_file = ui_build_path / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file), media_type="text/html")
    
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