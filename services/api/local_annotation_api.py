"""
Local Development API Server
Works completely offline with SQLite and optional LLM
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Import path fix for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Local imports
try:
    from services.database.local_models import (
        get_db, init_db, cache, queue,
        Document, Sentence, Candidate, GoldAnnotation, TriageItem, User,
        SessionLocal
    )
except ImportError:
    # Fallback import
    import sqlite3
    from pathlib import Path
    
    # Simple database connection for fallback
    db_path = Path("./data/local/annotations.db")
    def get_simple_db():
        return sqlite3.connect(str(db_path))
    
    # Placeholder classes
    class SimpleDB:
        def __init__(self):
            self.db_path = db_path
    
    Document = Sentence = Candidate = GoldAnnotation = TriageItem = User = SimpleDB
    SessionLocal = SimpleDB
    cache = {}
    queue = []
    
    def get_db():
        yield SimpleDB()
    
    def init_db():
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.close()
        print("✓ Simple database initialized")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config_path = Path(__file__).parent.parent.parent / "config" / "local_config.yaml"
if config_path.exists():
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
else:
    config = {"api": {"host": "127.0.0.1", "port": 8000}}

# Pydantic models
class DocumentInput(BaseModel):
    title: str
    text: str
    source: str = "manual"

class AnnotationDecision(BaseModel):
    candidate_id: int
    decision: str  # accept, reject, modify
    modified_data: Optional[Dict] = None
    confidence: float = 0.8
    notes: Optional[str] = None

class CandidateRequest(BaseModel):
    doc_id: str
    sent_id: str
    text: str

# Simple authentication
async def verify_token(authorization: Optional[str] = Header(None)):
    """Simple token verification"""
    if not config.get("auth", {}).get("enabled", True):
        return {"username": "local_user", "role": "admin"}
        
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
        
    token = authorization.replace("Bearer ", "")
    
    # Check against configured tokens
    for user in config.get("auth", {}).get("tokens", []):
        if user["token"] == token:
            return {"username": user["name"], "role": user["role"]}
            
    raise HTTPException(status_code=401, detail="Invalid token")

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Local Annotation API...")
    init_db()
    logger.info("✓ Database initialized")
    
    # Load rule engine patterns
    global rule_patterns
    rule_patterns = load_rule_patterns()
    logger.info("✓ Rule patterns loaded")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")

# Create FastAPI app
app = FastAPI(
    title="Shrimp Annotation Pipeline - Local Mode",
    description="Completely offline annotation system",
    version="1.0.0-local",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load rule patterns
def load_rule_patterns():
    """Load domain-specific patterns for rule-based annotation"""
    return {
        "entities": {
            "SPECIES": [
                (r"Penaeus\s+vannamei", "Penaeus vannamei", 0.95),
                (r"Litopenaeus\s+vannamei", "Litopenaeus vannamei", 0.95),
                (r"white[\s-]?leg\s+shrimp", "Litopenaeus vannamei", 0.90),
                (r"Pacific\s+white\s+shrimp", "Litopenaeus vannamei", 0.90),
            ],
            "PATHOGEN": [
                (r"Vibrio\s+parahaemolyticus", "Vibrio parahaemolyticus", 0.95),
                (r"Vibrio\s+harveyi", "Vibrio harveyi", 0.95),
                (r"WSSV", "White Spot Syndrome Virus", 0.95),
                (r"White\s+Spot\s+Syndrome\s+Virus", "White Spot Syndrome Virus", 0.95),
            ],
            "DISEASE": [
                (r"AHPND", "Acute Hepatopancreatic Necrosis Disease", 0.95),
                (r"WSD", "White Spot Disease", 0.95),
                (r"EMS", "Early Mortality Syndrome", 0.95),
            ],
            "MEASUREMENT": [
                (r"\d+\.?\d*\s*°C", "temperature", 0.90),
                (r"\d+\.?\d*\s*ppt", "salinity", 0.90),
                (r"\d+\.?\d*\s*mg/L", "concentration", 0.90),
                (r"\d+\.?\d*\s*%", "percentage", 0.90),
            ],
        }
    }

# LLM fallback to rules
async def generate_candidates(text: str, doc_id: str, sent_id: str) -> Dict:
    """Generate candidates using rules (LLM optional)"""
    candidates = {
        "entities": [],
        "relations": [],
        "topics": []
    }
    
    # Try Ollama if available
    if config.get("llm", {}).get("provider") == "ollama":
        try:
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2:3b",
                    "prompt": f"Extract entities from: {text}",
                    "stream": False
                },
                timeout=5
            )
            if response.status_code == 200:
                # Parse Ollama response
                pass
        except:
            logger.info("Ollama not available, using rules only")
    
    # Always apply rules
    import re
    for entity_type, patterns in rule_patterns["entities"].items():
        for pattern, canonical, confidence in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                candidates["entities"].append({
                    "text": match.group(),
                    "label": entity_type,
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": confidence,
                    "canonical": canonical
                })
    
    return candidates

# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "mode": "local",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/documents")
async def create_document(
    doc: DocumentInput,
    current_user: Dict = Depends(verify_token),
    db: SessionLocal = Depends(get_db)
):
    """Upload and process a new document"""
    try:
        # Generate doc ID
        import hashlib
        doc_id = hashlib.md5(f"{doc.title}:{doc.text[:100]}".encode()).hexdigest()[:12]
        
        # Create document
        db_doc = Document(
            doc_id=doc_id,
            title=doc.title,
            source=doc.source,
            raw_text=doc.text,
            metadata={"uploaded_by": current_user["username"]}
        )
        db.add(db_doc)
        
        # Segment into sentences
        sentences = doc.text.split('. ')
        for i, sent_text in enumerate(sentences):
            if sent_text.strip():
                sent = Sentence(
                    sent_id=f"s{i}",
                    doc_id=doc_id,
                    text=sent_text.strip() + '.',
                    start_offset=i*100,
                    end_offset=(i+1)*100
                )
                db.add(sent)
        
        db.commit()
        
        return {
            "doc_id": doc_id,
            "title": doc.title,
            "sentences": len(sentences),
            "status": "processed"
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Document creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents(
    current_user: Dict = Depends(verify_token),
    db: SessionLocal = Depends(get_db)
):
    """List all documents"""
    docs = db.query(Document).all()
    return [
        {
            "doc_id": d.doc_id,
            "title": d.title,
            "source": d.source,
            "created_at": d.created_at.isoformat() if d.created_at else None
        }
        for d in docs
    ]

@app.post("/candidates/generate")
async def generate_annotation_candidates(
    request: CandidateRequest,
    current_user: Dict = Depends(verify_token),
    db: SessionLocal = Depends(get_db)
):
    """Generate candidates for a sentence"""
    try:
        # Generate candidates
        candidates_data = await generate_candidates(
            request.text, 
            request.doc_id,
            request.sent_id
        )
        
        # Store in database
        candidate = Candidate(
            doc_id=request.doc_id,
            sent_id=request.sent_id,
            source="rule",  # or "llm" if Ollama worked
            entity_data=candidates_data["entities"],
            relation_data=candidates_data["relations"],
            topic_data=candidates_data["topics"],
            confidence=0.85
        )
        db.add(candidate)
        db.commit()
        
        # Add to triage queue
        triage_item = TriageItem(
            doc_id=request.doc_id,
            sent_id=request.sent_id,
            candidate_id=candidate.candidate_id,
            priority_score=0.7,
            priority_level="medium",
            status="pending"
        )
        db.add(triage_item)
        db.commit()
        
        return {
            "candidate_id": candidate.candidate_id,
            "entities": len(candidates_data["entities"]),
            "relations": len(candidates_data["relations"]),
            "status": "queued"
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Candidate generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/triage/next")
async def get_next_triage_item(
    current_user: Dict = Depends(verify_token),
    db: SessionLocal = Depends(get_db)
):
    """Get next item from triage queue"""
    # Get highest priority pending item
    item = db.query(TriageItem).filter(
        TriageItem.status == "pending"
    ).order_by(TriageItem.priority_score.desc()).first()
    
    if not item:
        return {"message": "No items in queue"}
    
    # Get candidate data
    candidate = db.query(Candidate).filter(
        Candidate.candidate_id == item.candidate_id
    ).first()
    
    # Get sentence
    sentence = db.query(Sentence).filter(
        Sentence.doc_id == item.doc_id,
        Sentence.sent_id == item.sent_id
    ).first()
    
    # Mark as in review
    item.status = "in_review"
    item.assigned_to = current_user["username"]
    db.commit()
    
    return {
        "item_id": item.item_id,
        "doc_id": item.doc_id,
        "sent_id": item.sent_id,
        "text": sentence.text if sentence else "",
        "candidates": {
            "entities": candidate.entity_data if candidate else [],
            "relations": candidate.relation_data if candidate else [],
            "topics": candidate.topic_data if candidate else []
        },
        "priority": item.priority_level
    }

@app.post("/annotations/decide")
async def make_annotation_decision(
    decision: AnnotationDecision,
    current_user: Dict = Depends(verify_token),
    db: SessionLocal = Depends(get_db)
):
    """Make annotation decision"""
    try:
        # Get candidate
        candidate = db.query(Candidate).filter(
            Candidate.candidate_id == decision.candidate_id
        ).first()
        
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")
        
        # Create gold annotation
        annotation_data = decision.modified_data if decision.modified_data else {
            "entities": candidate.entity_data,
            "relations": candidate.relation_data,
            "topics": candidate.topic_data
        }
        
        gold = GoldAnnotation(
            doc_id=candidate.doc_id,
            sent_id=candidate.sent_id,
            candidate_id=candidate.candidate_id,
            annotation_type="combined",
            annotation_data=annotation_data,
            annotator=current_user["username"],
            confidence=decision.confidence,
            decision=decision.decision,
            notes=decision.notes
        )
        db.add(gold)
        
        # Update triage item
        triage_item = db.query(TriageItem).filter(
            TriageItem.candidate_id == decision.candidate_id
        ).first()
        
        if triage_item:
            triage_item.status = "completed"
            triage_item.completed_at = datetime.utcnow()
        
        db.commit()
        
        return {
            "annotation_id": gold.annotation_id,
            "decision": decision.decision,
            "status": "saved"
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Annotation decision failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics")
async def get_statistics(
    current_user: Dict = Depends(verify_token),
    db: SessionLocal = Depends(get_db)
):
    """Get annotation statistics"""
    stats = {
        "documents": db.query(Document).count(),
        "sentences": db.query(Sentence).count(),
        "candidates": db.query(Candidate).count(),
        "gold_annotations": db.query(GoldAnnotation).count(),
        "queue_size": db.query(TriageItem).filter(
            TriageItem.status == "pending"
        ).count(),
        "completed_today": db.query(GoldAnnotation).filter(
            GoldAnnotation.created_at >= datetime.utcnow().date()
        ).count()
    }
    
    # User statistics
    if current_user["username"] != "local_user":
        user_stats = db.query(GoldAnnotation).filter(
            GoldAnnotation.annotator == current_user["username"]
        ).count()
        stats["your_annotations"] = user_stats
    
    return stats

@app.get("/export")
async def export_annotations(
    format: str = Query("json", enum=["json", "jsonl", "csv"]),
    current_user: Dict = Depends(verify_token),
    db: SessionLocal = Depends(get_db)
):
    """Export gold annotations"""
    if current_user["role"] not in ["admin", "reviewer"]:
        raise HTTPException(status_code=403, detail="Export requires admin/reviewer role")
    
    annotations = db.query(GoldAnnotation).all()
    
    export_data = []
    for ann in annotations:
        export_data.append({
            "doc_id": ann.doc_id,
            "sent_id": ann.sent_id,
            "annotations": ann.annotation_data,
            "annotator": ann.annotator,
            "confidence": ann.confidence,
            "decision": ann.decision,
            "created_at": ann.created_at.isoformat() if ann.created_at else None
        })
    
    if format == "jsonl":
        content = "\n".join(json.dumps(item) for item in export_data)
        return JSONResponse(content={"data": content}, media_type="application/x-ndjson")
    else:
        return export_data

# Run server
if __name__ == "__main__":
    # Use configuration
    host = config.get("api", {}).get("host", "127.0.0.1")
    port = config.get("api", {}).get("port", 8000)
    reload = config.get("api", {}).get("reload", True)
    
    logger.info(f"Starting Local API Server at http://{host}:{port}")
    logger.info("API Docs available at http://localhost:8000/docs")
    
    uvicorn.run(
        "local_annotation_api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )