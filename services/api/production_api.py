#!/usr/bin/env python3
"""
Production API with PostgreSQL support and error recovery endpoints

Designed for Railway deployment with 3 concurrent users.
Includes draft management and network recovery features.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Database imports
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

# Import existing models
from services.database.models import Base, Document, Sentence, GoldAnnotation, TriageItem, Candidate

logger = logging.getLogger(__name__)

# Pydantic models for API
class AnnotationDraft(BaseModel):
    item_id: str
    draft_data: Dict[str, Any]
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    database_status: str

# Database setup for PostgreSQL
def get_database_url():
    """Get database URL for Railway or local development"""
    if os.getenv('RAILWAY_ENVIRONMENT'):
        # Railway PostgreSQL
        return os.getenv('DATABASE_URL')
    else:
        # Local PostgreSQL for development
        user = os.getenv('POSTGRES_USER', 'postgres')
        password = os.getenv('POSTGRES_PASSWORD', 'postgres')
        host = os.getenv('POSTGRES_HOST', 'localhost')
        port = os.getenv('POSTGRES_PORT', '5432')
        db = os.getenv('POSTGRES_DB', 'shrimp_annotation')
        return f"postgresql://{user}:{password}@{host}:{port}/{db}"

# Create engine with connection pooling for concurrent users
DATABASE_URL = get_database_url()
engine = None
SessionLocal = None

def initialize_database():
    """Initialize database connection - called on first request"""
    global engine, SessionLocal
    
    if engine is not None:
        return engine is not None
    
    if not DATABASE_URL:
        logger.error("No database URL configured")
        return False
    
    try:
        engine = create_engine(
            DATABASE_URL,
            poolclass=QueuePool,
            pool_size=10,  # Handle 3 users + some overhead
            max_overflow=20,
            pool_pre_ping=True,  # Verify connections before use
            pool_recycle=3600,   # Recycle connections every hour
            echo=False
        )
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        # Create tables
        Base.metadata.create_all(bind=engine)
        logger.info(f"Connected to PostgreSQL: {DATABASE_URL[:50]}...")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        engine = None
        SessionLocal = None
        return False

# FastAPI app
app = FastAPI(
    title="Shrimp Annotation API",
    description="Production API for shrimp aquaculture annotation with error recovery",
    version="2.0.0"
)

# CORS for Railway deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Railway handles domain restrictions
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get database session
def get_db():
    if not initialize_database():
        raise HTTPException(status_code=500, detail="Database not configured")
    
    if not SessionLocal:
        raise HTTPException(status_code=500, detail="Database session not available")
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# In-memory draft storage (use Redis in production for scaling)
draft_storage: Dict[str, Dict] = {}

@app.get("/api/health", response_model=HealthResponse)
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint for network recovery testing"""
    try:
        # Test database connection
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "error"
        
    return HealthResponse(
        status="ok" if db_status == "connected" else "degraded",
        timestamp=datetime.now().isoformat(),
        database_status=db_status
    )

@app.post("/api/annotations/draft")
async def save_draft(draft: AnnotationDraft, db: Session = Depends(get_db)):
    """Save annotation draft for error recovery"""
    try:
        # Store in memory (use Redis for production scaling)
        draft_storage[draft.item_id] = {
            "data": draft.draft_data,
            "timestamp": draft.timestamp,
            "saved_at": datetime.now().isoformat()
        }
        
        logger.info(f"Saved draft for item {draft.item_id}")
        return {"status": "saved", "item_id": draft.item_id}
        
    except Exception as e:
        logger.error(f"Failed to save draft: {e}")
        raise HTTPException(status_code=500, detail="Failed to save draft")

@app.get("/api/annotations/draft/{item_id}")
async def get_draft(item_id: str):
    """Retrieve annotation draft"""
    if item_id in draft_storage:
        return {
            "item_id": item_id,
            "draft_data": draft_storage[item_id]["data"],
            "timestamp": draft_storage[item_id]["timestamp"]
        }
    else:
        raise HTTPException(status_code=404, detail="No draft found")

@app.delete("/api/annotations/draft")
async def clear_draft(request: Dict[str, str]):
    """Clear annotation draft after successful submission"""
    item_id = request.get("item_id")
    if item_id and item_id in draft_storage:
        del draft_storage[item_id]
        logger.info(f"Cleared draft for item {item_id}")
        return {"status": "cleared", "item_id": item_id}
    return {"status": "not_found", "item_id": item_id}

@app.get("/api/triage/queue")
async def get_triage_queue(limit: int = 50, offset: int = 0, db: Session = Depends(get_db)):
    """Get triage queue with PostgreSQL support"""
    try:
        items = db.query(TriageItem).filter(
            TriageItem.status == "pending"
        ).order_by(
            TriageItem.priority_score.desc()
        ).offset(offset).limit(limit).all()
        
        # Convert to API format
        queue_items = []
        for item in items:
            # Get sentence and document info
            sentence = db.query(Sentence).filter(Sentence.id == item.sentence_id).first()
            document = db.query(Document).filter(Document.id == sentence.document_id).first() if sentence else None
            candidate = db.query(Candidate).filter(Candidate.id == item.candidate_id).first()
            
            queue_item = {
                "id": item.item_id,
                "item_id": item.item_id,
                "priority_score": item.priority_score,
                "priority_level": item.priority_level,
                "text": sentence.text if sentence else "",
                "doc_id": document.doc_id if document else "",
                "sent_id": sentence.sent_id if sentence else "",
                "title": document.title if document else "",
                "entities": [],
                "relations": [],
                "topics": [],
                "created_at": item.created_at.isoformat()
            }
            
            # Add candidate data if available
            if candidate:
                if candidate.candidate_type == "entity":
                    queue_item["entities"] = [{
                        "id": str(candidate.id),
                        "text": candidate.text,
                        "label": candidate.label,
                        "start": candidate.start_offset,
                        "end": candidate.end_offset,
                        "confidence": candidate.confidence
                    }]
            
            queue_items.append(queue_item)
        
        return {"items": queue_items, "total": len(queue_items)}
        
    except Exception as e:
        logger.error(f"Failed to get triage queue: {e}")
        raise HTTPException(status_code=500, detail="Failed to get triage queue")

@app.get("/api/triage/next")
async def get_next_item(db: Session = Depends(get_db)):
    """Get next item from triage queue"""
    try:
        item = db.query(TriageItem).filter(
            TriageItem.status == "pending"
        ).order_by(
            TriageItem.priority_score.desc()
        ).first()
        
        if not item:
            return {"item": None}
        
        # Get associated data
        sentence = db.query(Sentence).filter(Sentence.id == item.sentence_id).first()
        document = db.query(Document).filter(Document.id == sentence.document_id).first() if sentence else None
        
        return {
            "item": {
                "id": item.item_id,
                "item_id": item.item_id,
                "text": sentence.text if sentence else "",
                "doc_id": document.doc_id if document else "",
                "sent_id": sentence.sent_id if sentence else "",
                "title": document.title if document else "",
                "priority_score": item.priority_score,
                "entities": [],
                "relations": [],
                "topics": []
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get next item: {e}")
        raise HTTPException(status_code=500, detail="Failed to get next item")

@app.post("/api/annotations/decide")
async def submit_annotation(annotation: Dict[str, Any], background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Submit annotation decision with PostgreSQL storage"""
    try:
        # Find the triage item
        item_id = annotation.get("item_id") or annotation.get("candidate_id")
        triage_item = db.query(TriageItem).filter(
            TriageItem.item_id == str(item_id)
        ).first()
        
        if not triage_item:
            raise HTTPException(status_code=404, detail="Triage item not found")
        
        # Get sentence and document
        sentence = db.query(Sentence).filter(Sentence.id == triage_item.sentence_id).first()
        document = db.query(Document).filter(Document.id == sentence.document_id).first() if sentence else None
        
        if not sentence or not document:
            raise HTTPException(status_code=404, detail="Associated data not found")
        
        # Create gold annotation
        gold_annotation = GoldAnnotation(
            document_id=document.id,
            sentence_id=sentence.id,
            entities=annotation.get("entities", []),
            relations=annotation.get("relations", []),
            topics=annotation.get("topics", []),
            annotator_email=annotation.get("annotator", "unknown@example.com"),
            status=annotation.get("decision", "accepted"),
            confidence_level=annotation.get("confidence", "high"),
            notes=annotation.get("notes", ""),
            decision_method="manual",
            source_candidate_id=triage_item.candidate_id
        )
        
        db.add(gold_annotation)
        
        # Update triage item status
        triage_item.status = "completed"
        triage_item.completed_at = datetime.now()
        
        db.commit()
        
        # Clear any draft for this item
        if item_id in draft_storage:
            del draft_storage[item_id]
        
        # Get next item
        next_item = db.query(TriageItem).filter(
            TriageItem.status == "pending"
        ).order_by(
            TriageItem.priority_score.desc()
        ).first()
        
        next_item_data = None
        if next_item:
            next_sentence = db.query(Sentence).filter(Sentence.id == next_item.sentence_id).first()
            next_document = db.query(Document).filter(Document.id == next_sentence.document_id).first() if next_sentence else None
            
            next_item_data = {
                "id": next_item.item_id,
                "item_id": next_item.item_id,
                "text": next_sentence.text if next_sentence else "",
                "doc_id": next_document.doc_id if next_document else "",
                "sent_id": next_sentence.sent_id if next_sentence else "",
                "title": next_document.title if next_document else "",
                "entities": [],
                "relations": [],
                "topics": []
            }
        
        logger.info(f"Annotation submitted for item {item_id}")
        
        return {
            "status": "success",
            "annotation_id": str(gold_annotation.id),
            "next_item": next_item_data
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to submit annotation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit annotation: {str(e)}")

@app.get("/api/statistics/overview")
async def get_statistics(db: Session = Depends(get_db)):
    """Get system statistics"""
    try:
        # Get basic counts
        total_items = db.query(TriageItem).count()
        pending_items = db.query(TriageItem).filter(TriageItem.status == "pending").count()
        completed_items = db.query(TriageItem).filter(TriageItem.status == "completed").count()
        total_annotations = db.query(GoldAnnotation).count()
        
        return {
            "total_items": total_items,
            "pending_items": pending_items,
            "completed_items": completed_items,
            "total_annotations": total_annotations,
            "completion_rate": completed_items / total_items if total_items > 0 else 0,
            "active_drafts": len(draft_storage)
        }
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for better error responses"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": str(type(exc).__name__)}
    )

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "production_api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )