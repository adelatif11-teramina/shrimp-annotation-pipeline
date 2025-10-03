"""
Mock API Server with All Frontend Endpoints
Provides working endpoints with realistic mock data
"""

import os
import json
import yaml
import logging
import hashlib
import random
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, Header, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock data storage
mock_data = {
    "documents": [],
    "sentences": [],
    "candidates": [],
    "annotations": [],
    "annotation_history": [],
    "triage_queue": [],
    "users": [
        {"id": 1, "username": "admin", "token": "local-admin-2024", "role": "admin"},
        {"id": 2, "username": "annotator1", "token": "anno-team-001", "role": "annotator"},
        {"id": 3, "username": "annotator2", "token": "anno-team-002", "role": "annotator"},
        {"id": 4, "username": "reviewer", "token": "review-lead-003", "role": "reviewer"},
    ],
    "stats": {
        "total_documents": 12,
        "total_sentences": 156,
        "processed_sentences": 89,
        "total_candidates": 234,
        "total_annotations": 178,
        "accepted_annotations": 142,
        "rejected_annotations": 36,
        "queue_size": 67,
        "annotations_today": 23,
        "average_confidence": 0.84,
        "accepted_rate": 0.76,
        "rejection_rate": 0.24,
        "annotations_per_hour": 15.2,
        "average_priority": 0.68
    }
}

# Pydantic models
class DocumentCreate(BaseModel):
    title: str
    text: str
    source: str = "manual"

class AnnotationDecision(BaseModel):
    candidate_id: int
    decision: str
    entities: Optional[List[Dict]] = []
    relations: Optional[List[Dict]] = []
    topics: Optional[List[Dict]] = []
    confidence: float = 0.8
    notes: Optional[str] = None

class CandidateRequest(BaseModel):
    doc_id: str
    sent_id: str
    text: str

# Authentication
async def get_current_user(request: Request, authorization: Optional[str] = Header(None, alias="Authorization")) -> Dict[str, Any]:
    """Get current user from token"""
    print(f"🔍 Auth header received: {authorization}")
    print(f"🔍 All headers: {dict(request.headers)}")
    
    if not authorization or not authorization.startswith("Bearer "):
        print(f"❌ Invalid auth header format")
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    token = authorization.replace("Bearer ", "")
    print(f"🔑 Extracted token: {token}")
    
    for user in mock_data["users"]:
        if user["token"] == token:
            return user
    
    raise HTTPException(status_code=401, detail="Invalid token")


# Load realistic mock data
try:
    with open("realistic_mock_data.json", "r", encoding="utf-8") as f:
        REALISTIC_MOCK_DATA = json.load(f)
    print(f"✅ Loaded {len(REALISTIC_MOCK_DATA)} realistic mock items")
except FileNotFoundError:
    REALISTIC_MOCK_DATA = []
    print("⚠️ No realistic mock data file found")

# Mock data generators
def generate_sample_entities(text: str) -> List[Dict]:
    """Generate realistic entity annotations"""
    entities = []
    
    # Sample entities based on text content
    if "Penaeus" in text or "vannamei" in text:
        entities.append({
            "id": len(entities) + 1,
            "text": "Penaeus vannamei",
            "label": "SPECIES",
            "start": text.find("Penaeus"),
            "end": text.find("vannamei") + 8,
            "confidence": 0.95,
            "canonical": "Litopenaeus vannamei"
        })
    
    if "Vibrio" in text:
        entities.append({
            "id": len(entities) + 1,
            "text": "Vibrio parahaemolyticus",
            "label": "PATHOGEN", 
            "start": text.find("Vibrio"),
            "end": text.find("Vibrio") + 23,
            "confidence": 0.92,
            "canonical": "Vibrio parahaemolyticus"
        })
    
    if "AHPND" in text:
        entities.append({
            "id": len(entities) + 1,
            "text": "AHPND",
            "label": "DISEASE",
            "start": text.find("AHPND"),
            "end": text.find("AHPND") + 5,
            "confidence": 0.98,
            "canonical": "Acute Hepatopancreatic Necrosis Disease"
        })
    
    # Add temperature measurements
    import re
    temp_matches = re.finditer(r'\d+\.?\d*\s*°C', text)
    for match in temp_matches:
        entities.append({
            "id": len(entities) + 1,
            "text": match.group(),
            "label": "MEASUREMENT",
            "start": match.start(),
            "end": match.end(),
            "confidence": 0.90,
            "canonical": "temperature"
        })
    
    return entities

def generate_sample_relations(entities: List[Dict]) -> List[Dict]:
    """Generate sample relations between entities"""
    relations = []
    
    species = [e for e in entities if e["label"] == "SPECIES"]
    pathogens = [e for e in entities if e["label"] == "PATHOGEN"]
    diseases = [e for e in entities if e["label"] == "DISEASE"]
    
    # Species-pathogen relations
    for species_ent in species:
        for pathogen_ent in pathogens:
            relations.append({
                "id": len(relations) + 1,
                "head_id": species_ent["id"],
                "tail_id": pathogen_ent["id"],
                "label": "infected_by",
                "confidence": 0.85,
                "evidence": f"{species_ent['text']} infected by {pathogen_ent['text']}"
            })
    
    # Pathogen-disease relations
    for pathogen_ent in pathogens:
        for disease_ent in diseases:
            relations.append({
                "id": len(relations) + 1,
                "head_id": pathogen_ent["id"],
                "tail_id": disease_ent["id"],
                "label": "causes",
                "confidence": 0.88,
                "evidence": f"{pathogen_ent['text']} causes {disease_ent['text']}"
            })
    
    return relations

def generate_sample_topics(text: str) -> List[Dict]:
    """Generate sample topic classifications"""
    topics = []
    
    if any(word in text.lower() for word in ["disease", "infection", "mortality", "ahpnd"]):
        topics.append({
            "topic": "T_DISEASE",
            "score": 0.92,
            "keywords": ["disease", "infection", "mortality"]
        })
    
    if any(word in text.lower() for word in ["treatment", "antibiotic", "therapy"]):
        topics.append({
            "topic": "T_TREATMENT", 
            "score": 0.78,
            "keywords": ["treatment"]
        })
    
    if any(word in text.lower() for word in ["temperature", "salinity", "water", "quality"]):
        topics.append({
            "topic": "T_WATER_QUALITY",
            "score": 0.65,
            "keywords": ["temperature", "water"]
        })
    
    return sorted(topics, key=lambda x: x["score"], reverse=True)

# Initialize sample data
def initialize_sample_data():
    """Create realistic sample data"""
    
    # Sample documents
    sample_docs = [
        {
            "id": 1,
            "doc_id": "doc_001",
            "title": "AHPND Outbreak Study",
            "source": "research_paper",
            "created_at": "2024-09-28T10:30:00Z",
            "sentence_count": 15,
            "processed": True
        },
        {
            "id": 2,
            "doc_id": "doc_002", 
            "title": "Vibrio Resistance Analysis",
            "source": "lab_report",
            "created_at": "2024-09-29T14:20:00Z",
            "sentence_count": 23,
            "processed": False
        },
        {
            "id": 3,
            "doc_id": "doc_003",
            "title": "Water Quality Management",
            "source": "field_notes",
            "created_at": "2024-09-30T09:15:00Z",
            "sentence_count": 8,
            "processed": True
        }
    ]
    mock_data["documents"] = sample_docs
    
    # Sample triage queue items
    sample_queue = []
    for i in range(20):
        priority_score = random.uniform(0.3, 0.98)
        if priority_score > 0.8:
            priority_level = "high"
        elif priority_score > 0.5:
            priority_level = "medium"
        else:
            priority_level = "low"
        
        entities = generate_sample_entities("Penaeus vannamei infected with Vibrio parahaemolyticus showing AHPND symptoms at 28°C")
        relations = generate_sample_relations(entities)
        topics = generate_sample_topics("Disease outbreak in shrimp culture")
        
        sample_queue.append({
            "id": i + 1,
            "candidate_id": i + 100,
            "doc_id": f"doc_{(i % 3) + 1:03d}",
            "sent_id": f"s{i % 10}",
            "text": f"Sample sentence {i+1} about shrimp disease and water quality management.",
            "priority_score": priority_score,
            "priority_level": priority_level,
            "status": random.choice(["pending", "pending", "pending", "in_review"]),
            "entities": entities,
            "relations": relations,
            "topics": topics,
            "created_at": (datetime.utcnow() - timedelta(hours=random.randint(1, 72))).isoformat()
        })
    
    mock_data["triage_queue"] = sample_queue

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting Mock API Server with Complete Endpoints...")
    initialize_sample_data()
    logger.info("✓ Mock data initialized")
    yield
    
    # Shutdown
    logger.info("Shutting down Mock API...")

# Create FastAPI app
app = FastAPI(
    title="Shrimp Annotation Pipeline - Mock API",
    description="Mock API with all frontend endpoints and realistic data",
    version="1.0.0-mock",
    lifespan=lifespan
)

# CORS middleware - properly configured for Authorization headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://localhost:3010"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Health endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "mode": "mock_api",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": "all_available"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "database": "mock",
        "documents": len(mock_data["documents"]),
        "queue_size": len([item for item in mock_data["triage_queue"] if item["status"] == "pending"]),
        "users": len(mock_data["users"]),
        "timestamp": datetime.utcnow().isoformat()
    }

# Statistics endpoints (what frontend expects)
@app.get("/statistics/overview")
async def get_statistics_overview(current_user: Dict = Depends(get_current_user)):
    """Get system overview statistics"""
    
    # Add some variance to make it look realistic
    stats = mock_data["stats"].copy()
    stats.update({
        "annotations_today": stats["annotations_today"] + random.randint(-2, 5),
        "queue_size": len([item for item in mock_data["triage_queue"] if item["status"] == "pending"]),
        "average_confidence": round(stats["average_confidence"] + random.uniform(-0.05, 0.05), 2),
        "timestamp": datetime.utcnow().isoformat()
    })
    
    return {
        "overview": {
            "total_documents": stats["total_documents"],
            "total_sentences": stats["total_sentences"], 
            "processed_sentences": stats["processed_sentences"],
            "total_candidates": stats["total_candidates"],
            "total_annotations": stats["total_annotations"],
            "queue_size": stats["queue_size"],
            "annotations_today": stats["annotations_today"]
        },
        "quality": {
            "average_confidence": stats["average_confidence"],
            "accepted_rate": stats["accepted_rate"],
            "rejection_rate": stats["rejection_rate"]
        },
        "performance": {
            "annotations_per_hour": stats["annotations_per_hour"],
            "average_priority": stats["average_priority"]
        },
        "timestamp": stats["timestamp"]
    }

@app.get("/statistics")
async def get_statistics(current_user: Dict = Depends(get_current_user)):
    """Legacy statistics endpoint"""
    return await get_statistics_overview(current_user)

# Triage endpoints
@app.get("/triage/statistics")
async def get_triage_statistics(current_user: Dict = Depends(get_current_user)):
    """Get triage queue statistics"""
    
    queue_items = mock_data["triage_queue"]
    total_items = len(queue_items)
    pending_items = len([item for item in queue_items if item["status"] == "pending"])
    in_review_items = len([item for item in queue_items if item["status"] == "in_review"])
    completed_items = len([item for item in queue_items if item["status"] == "completed"])
    
    # Priority breakdown
    critical_items = len([item for item in queue_items if item["priority_level"] == "critical" and item["status"] == "pending"])
    high_items = len([item for item in queue_items if item["priority_level"] == "high" and item["status"] == "pending"])
    medium_items = len([item for item in queue_items if item["priority_level"] == "medium" and item["status"] == "pending"])
    low_items = len([item for item in queue_items if item["priority_level"] == "low" and item["status"] == "pending"])
    
    return {
        "total_items": total_items,
        "pending_items": pending_items,
        "in_review_items": in_review_items,
        "completed_items": completed_items,
        "priority_breakdown": {
            "critical": critical_items,
            "high": high_items,
            "medium": medium_items,
            "low": low_items
        },
        "completion_rate": round((completed_items / max(1, total_items)) * 100, 1),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/triage/queue")
async def get_triage_queue(
    status: Optional[str] = Query(None),
    sort_by: str = Query("priority"),
    limit: int = Query(50),
    offset: int = Query(0),
    search: Optional[str] = Query(None),
    priority_level: Optional[str] = Query(None),
    current_user: Dict = Depends(get_current_user)
):
    """Get triage queue items - now with realistic data from your documents"""
    
    # Use realistic data if available, otherwise fall back to generated
    if REALISTIC_MOCK_DATA:
        items = REALISTIC_MOCK_DATA.copy()
    else:
        # Fallback to generated data
        items = []
        for i in range(min(limit, 10)):
            priority_score = random.uniform(0.3, 0.98)
            priority_level_calc = "high" if priority_score > 0.8 else "medium" if priority_score > 0.5 else "low"
            
            items.append({
                "id": i + 1,
                "candidate_id": i + 100,
                "doc_id": f"test{(i % 3) + 1}",
                "sent_id": f"s{i}",
                "text": f"Real sentence {i+1} about shrimp aquaculture from imported documents.",
                "priority_score": priority_score,
                "priority_level": priority_level_calc,
                "status": "pending",
                "entities": generate_sample_entities("Penaeus vannamei Vibrio"),
                "relations": [],
                "topics": generate_sample_topics("Disease in shrimp"),
                "created_at": datetime.utcnow().isoformat()
            })
    
    # Apply filters
    if status and status != "undefined":
        items = [item for item in items if item.get("status") == status]
    
    if priority_level:
        items = [item for item in items if item.get("priority_level") == priority_level]
    
    if search:
        items = [item for item in items if search.lower() in item.get("text", "").lower()]
    
    # Sort
    if sort_by == "priority":
        items.sort(key=lambda x: x.get("priority_score", 0), reverse=True)
    else:
        items.sort(key=lambda x: x.get("id", 0))
    
    # Pagination
    total = len(items)
    items = items[offset:offset + limit]
    
    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
        "filters": {
            "status": status,
            "search": search,
            "priority_level": priority_level,
            "sort_by": sort_by
        }
    }


@app.get("/triage/next")
async def get_next_triage_item():
    """Get next item from triage queue"""
    
    # Use realistic data
    if REALISTIC_MOCK_DATA:
        pending_items = [item for item in REALISTIC_MOCK_DATA if item["status"] == "pending"]
    else:
        pending_items = [item for item in mock_data["triage_queue"] if item["status"] == "pending"]
    
    if not pending_items:
        return {"message": "No items in queue", "item": None}
    
    # Get highest priority item
    next_item = max(pending_items, key=lambda x: x["priority_score"])
    
    # Mark as in review (but don't modify the original data)
    result_item = next_item.copy()
    result_item["status"] = "in_review"
    result_item["assigned_to"] = "anonymous"
    result_item["assigned_at"] = datetime.utcnow().isoformat()
    
    return {"item": result_item}

@app.get("/triage/queue/{item_id}")
async def get_triage_item(item_id: int):
    """Get a specific item from triage queue by ID"""
    
    # Use realistic data
    if REALISTIC_MOCK_DATA:
        items = REALISTIC_MOCK_DATA
    else:
        items = mock_data["triage_queue"]
    
    # Find item by ID
    item = next((item for item in items if item["id"] == item_id), None)
    
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    return {"item": item}

# Document management
@app.post("/documents")
async def create_document(
    doc: DocumentCreate,
    current_user: Dict = Depends(get_current_user)
):
    """Create a new document"""
    
    doc_id = hashlib.md5(f"{doc.title}:{doc.text[:50]}".encode()).hexdigest()[:12]
    
    new_doc = {
        "id": len(mock_data["documents"]) + 1,
        "doc_id": doc_id,
        "title": doc.title,
        "source": doc.source,
        "created_at": datetime.utcnow().isoformat(),
        "sentence_count": len([s for s in doc.text.split('.') if s.strip()]),
        "processed": False
    }
    
    mock_data["documents"].append(new_doc)
    mock_data["stats"]["total_documents"] += 1
    
    return new_doc

@app.get("/documents")
async def list_documents(
    limit: int = Query(50),
    offset: int = Query(0),
    search: Optional[str] = Query(None),
    source: Optional[str] = Query(None),
    current_user: Dict = Depends(get_current_user)
):
    """List documents with search and filter capabilities"""
    
    documents = mock_data["documents"].copy()
    
    # Apply search filter
    if search:
        search_lower = search.lower()
        documents = [
            doc for doc in documents 
            if search_lower in doc.get("title", "").lower() 
            or search_lower in doc.get("doc_id", "").lower()
        ]
    
    # Apply source filter
    if source:
        documents = [doc for doc in documents if doc.get("source") == source]
    
    total = len(documents)
    documents = documents[offset:offset + limit]
    
    return {
        "documents": documents,
        "total": total,
        "limit": limit,
        "offset": offset,
        "filters": {
            "search": search,
            "source": source
        }
    }

@app.get("/documents/{doc_id}")
async def get_document(
    doc_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Get a specific document"""
    
    doc = next((d for d in mock_data["documents"] if d["doc_id"] == doc_id), None)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Add sample sentences
    sentences = []
    for i in range(doc["sentence_count"]):
        sentences.append({
            "sent_id": f"s{i}",
            "text": f"Sample sentence {i+1} from document {doc_id}.",
            "start_offset": i * 100,
            "end_offset": (i + 1) * 100,
            "processed": i < doc["sentence_count"] // 2
        })
    
    return {
        **doc,
        "raw_text": f"Sample text content for document {doc['title']}...",
        "sentences": sentences
    }

# Candidate generation
@app.post("/candidates/generate")
async def generate_candidates(
    request: CandidateRequest,
    current_user: Dict = Depends(get_current_user)
):
    """Generate annotation candidates"""
    
    entities = generate_sample_entities(request.text)
    relations = generate_sample_relations(entities)
    topics = generate_sample_topics(request.text)
    
    candidate_id = len(mock_data["candidates"]) + 1
    confidence = 0.75 + (len(entities) * 0.05)
    priority_score = min(1.0, confidence * 0.8 + (len(relations) * 0.1))
    
    mock_data["candidates"].append({
        "id": candidate_id,
        "doc_id": request.doc_id,
        "sent_id": request.sent_id,
        "entities": entities,
        "relations": relations,
        "topics": topics,
        "confidence": confidence,
        "priority_score": priority_score
    })
    
    return {
        "candidate_id": candidate_id,
        "doc_id": request.doc_id,
        "sent_id": request.sent_id,
        "entities": len(entities),
        "relations": len(relations),
        "topics": len(topics),
        "confidence": confidence,
        "priority_score": priority_score,
        "status": "created"
    }

# Annotation decisions with history tracking
@app.post("/annotations/decide")
async def make_annotation_decision(
    decision: AnnotationDecision,
    request: Request,
    current_user: Dict = Depends(get_current_user)
):
    """Make annotation decision with history tracking"""
    
    annotation_id = len(mock_data["annotations"]) + 1
    timestamp = datetime.utcnow().isoformat()
    
    # Check if this is an update to existing annotation
    existing_annotation = None
    for ann in mock_data["annotations"]:
        if ann["candidate_id"] == decision.candidate_id:
            existing_annotation = ann
            break
    
    if existing_annotation:
        # Create history entry before updating
        history_entry = {
            "id": len(mock_data["annotation_history"]) + 1,
            "annotation_id": existing_annotation["id"],
            "candidate_id": decision.candidate_id,
            "version": len([h for h in mock_data["annotation_history"] if h["annotation_id"] == existing_annotation["id"]]) + 1,
            "previous_state": {
                "decision": existing_annotation["decision"],
                "entities": existing_annotation["entities"],
                "relations": existing_annotation["relations"], 
                "topics": existing_annotation["topics"],
                "confidence": existing_annotation["confidence"],
                "notes": existing_annotation["notes"]
            },
            "new_state": {
                "decision": decision.decision,
                "entities": decision.entities,
                "relations": decision.relations,
                "topics": decision.topics,
                "confidence": decision.confidence,
                "notes": decision.notes
            },
            "change_type": "modify",
            "changed_by": current_user["username"],
            "changed_at": timestamp,
            "change_reason": f"Updated from {existing_annotation['decision']} to {decision.decision}"
        }
        mock_data["annotation_history"].append(history_entry)
        
        # Update existing annotation
        existing_annotation.update({
            "decision": decision.decision,
            "entities": decision.entities,
            "relations": decision.relations,
            "topics": decision.topics,
            "confidence": decision.confidence,
            "notes": decision.notes,
            "updated_at": timestamp,
            "updated_by": current_user["username"],
            "version": history_entry["version"]
        })
        
        return {
            "annotation_id": existing_annotation["id"],
            "decision": decision.decision,
            "candidate_id": decision.candidate_id,
            "status": "updated",
            "version": history_entry["version"],
            "timestamp": timestamp
        }
    
    else:
        # Create new annotation
        new_annotation = {
            "id": annotation_id,
            "candidate_id": decision.candidate_id,
            "decision": decision.decision,
            "entities": decision.entities,
            "relations": decision.relations,
            "topics": decision.topics,
            "confidence": decision.confidence,
            "annotator": current_user["username"],
            "notes": decision.notes,
            "created_at": timestamp,
            "version": 1
        }
        mock_data["annotations"].append(new_annotation)
        
        # Create initial history entry
        history_entry = {
            "id": len(mock_data["annotation_history"]) + 1,
            "annotation_id": annotation_id,
            "candidate_id": decision.candidate_id,
            "version": 1,
            "previous_state": None,
            "new_state": {
                "decision": decision.decision,
                "entities": decision.entities,
                "relations": decision.relations,
                "topics": decision.topics,
                "confidence": decision.confidence,
                "notes": decision.notes
            },
            "change_type": "create",
            "changed_by": current_user["username"],
            "changed_at": timestamp,
            "change_reason": "Initial annotation"
        }
        mock_data["annotation_history"].append(history_entry)
    
    # Update triage item status
    for item in mock_data["triage_queue"]:
        if item["candidate_id"] == decision.candidate_id:
            item["status"] = "completed"
            item["completed_at"] = timestamp
            break
    
    # Update stats
    mock_data["stats"]["total_annotations"] += 1
    if decision.decision == "accept":
        mock_data["stats"]["accepted_annotations"] += 1
    elif decision.decision == "reject":
        mock_data["stats"]["rejected_annotations"] += 1
    
    return {
        "annotation_id": annotation_id,
        "decision": decision.decision,
        "candidate_id": decision.candidate_id,
        "status": "created",
        "version": 1,
        "timestamp": timestamp
    }

# Annotation history endpoints
@app.get("/annotations/{annotation_id}/history")
async def get_annotation_history(
    annotation_id: int,
    current_user: Dict = Depends(get_current_user)
):
    """Get history for a specific annotation"""
    
    history = [
        h for h in mock_data["annotation_history"] 
        if h["annotation_id"] == annotation_id
    ]
    
    if not history:
        raise HTTPException(status_code=404, detail="No history found for this annotation")
    
    # Sort by version
    history.sort(key=lambda x: x["version"])
    
    return {
        "annotation_id": annotation_id,
        "total_versions": len(history),
        "history": history
    }

@app.get("/annotations/{annotation_id}/versions/{version}")
async def get_annotation_version(
    annotation_id: int,
    version: int,
    current_user: Dict = Depends(get_current_user)
):
    """Get a specific version of an annotation"""
    
    history_entry = None
    for h in mock_data["annotation_history"]:
        if h["annotation_id"] == annotation_id and h["version"] == version:
            history_entry = h
            break
    
    if not history_entry:
        raise HTTPException(status_code=404, detail="Version not found")
    
    return {
        "annotation_id": annotation_id,
        "version": version,
        "state": history_entry["new_state"],
        "metadata": {
            "changed_by": history_entry["changed_by"],
            "changed_at": history_entry["changed_at"],
            "change_type": history_entry["change_type"],
            "change_reason": history_entry["change_reason"]
        }
    }

@app.post("/annotations/{annotation_id}/revert/{version}")
async def revert_annotation(
    annotation_id: int,
    version: int,
    current_user: Dict = Depends(get_current_user)
):
    """Revert annotation to a previous version"""
    
    if current_user["role"] not in ["admin", "reviewer"]:
        raise HTTPException(status_code=403, detail="Revert requires admin or reviewer role")
    
    # Find the target version
    target_history = None
    for h in mock_data["annotation_history"]:
        if h["annotation_id"] == annotation_id and h["version"] == version:
            target_history = h
            break
    
    if not target_history:
        raise HTTPException(status_code=404, detail="Version not found")
    
    # Find current annotation
    current_annotation = None
    for ann in mock_data["annotations"]:
        if ann["id"] == annotation_id:
            current_annotation = ann
            break
    
    if not current_annotation:
        raise HTTPException(status_code=404, detail="Annotation not found")
    
    # Create revert history entry
    timestamp = datetime.utcnow().isoformat()
    new_version = len([h for h in mock_data["annotation_history"] if h["annotation_id"] == annotation_id]) + 1
    
    revert_history = {
        "id": len(mock_data["annotation_history"]) + 1,
        "annotation_id": annotation_id,
        "candidate_id": current_annotation["candidate_id"],
        "version": new_version,
        "previous_state": {
            "decision": current_annotation["decision"],
            "entities": current_annotation["entities"],
            "relations": current_annotation["relations"],
            "topics": current_annotation["topics"],
            "confidence": current_annotation["confidence"],
            "notes": current_annotation["notes"]
        },
        "new_state": target_history["new_state"],
        "change_type": "revert",
        "changed_by": current_user["username"],
        "changed_at": timestamp,
        "change_reason": f"Reverted to version {version}"
    }
    mock_data["annotation_history"].append(revert_history)
    
    # Update current annotation
    target_state = target_history["new_state"]
    current_annotation.update({
        "decision": target_state["decision"],
        "entities": target_state["entities"],
        "relations": target_state["relations"],
        "topics": target_state["topics"],
        "confidence": target_state["confidence"],
        "notes": target_state["notes"],
        "updated_at": timestamp,
        "updated_by": current_user["username"],
        "version": new_version
    })
    
    return {
        "annotation_id": annotation_id,
        "reverted_to_version": version,
        "new_version": new_version,
        "timestamp": timestamp
    }

# Export endpoints
@app.get("/export")
async def export_annotations(
    format: str = Query("json", enum=["json", "jsonl", "csv", "conll"]),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    current_user: Dict = Depends(get_current_user)
):
    """Export annotations in various formats"""
    
    if current_user["role"] not in ["admin", "reviewer"]:
        raise HTTPException(status_code=403, detail="Export requires admin or reviewer role")
    
    annotations = mock_data["annotations"]
    
    # Filter by date if provided
    if date_from or date_to:
        filtered_annotations = []
        for ann in annotations:
            ann_date = ann.get("created_at", datetime.utcnow().isoformat())
            if date_from and ann_date < date_from:
                continue
            if date_to and ann_date > date_to:
                continue
            filtered_annotations.append(ann)
        annotations = filtered_annotations
    
    if format == "jsonl":
        content = "\n".join(json.dumps(item) for item in annotations)
        return {"data": content, "count": len(annotations), "format": "jsonl"}
    
    elif format == "csv":
        import io
        output = io.StringIO()
        if annotations:
            fieldnames = ["id", "candidate_id", "decision", "entities", "relations", "topics", 
                         "confidence", "annotator", "created_at", "notes"]
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            
            for ann in annotations:
                row = {
                    "id": ann.get("id", ""),
                    "candidate_id": ann.get("candidate_id", ""),
                    "decision": ann.get("decision", ""),
                    "entities": json.dumps(ann.get("entities", [])),
                    "relations": json.dumps(ann.get("relations", [])),
                    "topics": json.dumps(ann.get("topics", [])),
                    "confidence": ann.get("confidence", 0),
                    "annotator": ann.get("annotator", ""),
                    "created_at": ann.get("created_at", ""),
                    "notes": ann.get("notes", "")
                }
                writer.writerow(row)
        
        return {"data": output.getvalue(), "count": len(annotations), "format": "csv"}
    
    elif format == "conll":
        # CoNLL-style format for entity annotations
        output_lines = []
        for ann in annotations:
            if ann.get("decision") == "accept" and ann.get("entities"):
                doc_id = f"# doc_id = {ann.get('candidate_id', 'unknown')}"
                output_lines.append(doc_id)
                
                # Simple tokenization (would need proper tokenizer in production)
                text = "Sample sentence text for CoNLL export"  # Would get from annotation
                tokens = text.split()
                
                entities = ann.get("entities", [])
                labels = ["O"] * len(tokens)  # Start with all O labels
                
                # Map entities to BIO tags (simplified)
                for entity in entities:
                    start_token = 0  # Would calculate proper token positions
                    end_token = min(start_token + 1, len(tokens))
                    if start_token < len(labels):
                        labels[start_token] = f"B-{entity.get('label', 'ENTITY')}"
                    for i in range(start_token + 1, end_token):
                        if i < len(labels):
                            labels[i] = f"I-{entity.get('label', 'ENTITY')}"
                
                for token, label in zip(tokens, labels):
                    output_lines.append(f"{token}\t{label}")
                
                output_lines.append("")  # Empty line between sentences
        
        return {"data": "\n".join(output_lines), "count": len(annotations), "format": "conll"}
    
    else:
        return {
            "data": annotations,
            "count": len(annotations),
            "format": format,
            "exported_at": datetime.utcnow().isoformat()
        }

# Search endpoints
@app.get("/search")
async def global_search(
    q: str = Query(..., description="Search query"),
    type: Optional[str] = Query(None, enum=["documents", "annotations", "queue", "all"]),
    limit: int = Query(20),
    current_user: Dict = Depends(get_current_user)
):
    """Global search across all content"""
    
    results = {
        "query": q,
        "total_results": 0,
        "documents": [],
        "annotations": [],
        "queue_items": []
    }
    
    search_lower = q.lower()
    
    # Search documents
    if not type or type in ["documents", "all"]:
        for doc in mock_data["documents"]:
            if (search_lower in doc.get("title", "").lower() or 
                search_lower in doc.get("doc_id", "").lower()):
                results["documents"].append({
                    "type": "document",
                    "id": doc["doc_id"],
                    "title": doc["title"],
                    "snippet": doc["title"],
                    "score": 1.0
                })
    
    # Search triage queue
    if not type or type in ["queue", "all"]:
        for item in mock_data["triage_queue"]:
            if (search_lower in item.get("text", "").lower() or
                search_lower in item.get("doc_id", "").lower()):
                results["queue_items"].append({
                    "type": "queue_item", 
                    "id": item["id"],
                    "text": item["text"][:100] + "...",
                    "doc_id": item["doc_id"],
                    "priority": item["priority_level"],
                    "score": 0.8
                })
    
    # Search annotations
    if not type or type in ["annotations", "all"]:
        for ann in mock_data["annotations"]:
            # Search in entity text and notes
            searchable_text = ""
            for entity in ann.get("entities", []):
                searchable_text += entity.get("text", "") + " "
            searchable_text += ann.get("notes", "")
            
            if search_lower in searchable_text.lower():
                results["annotations"].append({
                    "type": "annotation",
                    "id": ann["id"],
                    "decision": ann["decision"],
                    "entities": len(ann.get("entities", [])),
                    "confidence": ann["confidence"],
                    "score": 0.9
                })
    
    # Limit results and calculate total
    if limit:
        results["documents"] = results["documents"][:limit//3]
        results["queue_items"] = results["queue_items"][:limit//3]
        results["annotations"] = results["annotations"][:limit//3]
    
    results["total_results"] = (len(results["documents"]) + 
                              len(results["queue_items"]) + 
                              len(results["annotations"]))
    
    return results

# Guidelines endpoint
@app.get("/guidelines")
async def get_annotation_guidelines(current_user: Dict = Depends(get_current_user)):
    """Get comprehensive annotation guidelines"""
    
    guidelines = {
        "entity_types": {
            "SPECIES": {
                "description": "Aquatic species, primarily shrimp varieties",
                "examples": ["Penaeus vannamei", "Litopenaeus vannamei", "Pacific white shrimp"],
                "rules": [
                    "Use scientific names when available",
                    "Include common names only if scientific name is not present",
                    "Annotate subspecies and varieties"
                ],
                "color": "#2196F3"
            },
            "PATHOGEN": {
                "description": "Disease-causing organisms including bacteria, viruses, parasites",
                "examples": ["Vibrio parahaemolyticus", "WSSV", "White Spot Syndrome Virus"],
                "rules": [
                    "Include both scientific and common names",
                    "Annotate specific strains when mentioned",
                    "Include emerging pathogens"
                ],
                "color": "#F44336"
            },
            "DISEASE": {
                "description": "Specific diseases and health conditions",
                "examples": ["AHPND", "White Spot Disease", "Early Mortality Syndrome"],
                "rules": [
                    "Use standard disease nomenclature",
                    "Include acronyms and full names",
                    "Annotate disease stages when specified"
                ],
                "color": "#FF9800"
            },
            "SYMPTOM": {
                "description": "Observable signs of disease or stress",
                "examples": ["mortality", "lethargy", "white spots", "red discoloration"],
                "rules": [
                    "Include physical symptoms",
                    "Annotate behavioral changes",
                    "Note severity when mentioned"
                ],
                "color": "#E91E63"
            },
            "CHEMICAL": {
                "description": "Chemicals, drugs, treatments, and compounds",
                "examples": ["oxytetracycline", "formalin", "probiotics", "vitamin C"],
                "rules": [
                    "Include active ingredients",
                    "Annotate commercial names when relevant",
                    "Include dosages when specified"
                ],
                "color": "#9C27B0"
            },
            "EQUIPMENT": {
                "description": "Tools, devices, and equipment used in aquaculture",
                "examples": ["aerators", "biofilters", "nets", "pumps"],
                "rules": [
                    "Include specific equipment types",
                    "Annotate brands when mentioned",
                    "Include technical specifications"
                ],
                "color": "#607D8B"
            },
            "LOCATION": {
                "description": "Geographic locations, facilities, and environmental settings",
                "examples": ["pond", "hatchery", "Southeast Asia", "farm"],
                "rules": [
                    "Include specific and general locations",
                    "Annotate water bodies and regions",
                    "Include facility types"
                ],
                "color": "#4CAF50"
            },
            "MEASUREMENT": {
                "description": "Quantitative measurements and values",
                "examples": ["28°C", "35 ppt", "7.5 pH", "80% survival rate"],
                "rules": [
                    "Always include units",
                    "Annotate ranges and thresholds",
                    "Include measurement contexts"
                ],
                "color": "#FF5722"
            },
            "PROCESS": {
                "description": "Procedures, methods, and operational processes",
                "examples": ["water exchange", "feeding", "vaccination", "harvesting"],
                "rules": [
                    "Include step-by-step procedures",
                    "Annotate timing and frequency",
                    "Include process parameters"
                ],
                "color": "#795548"
            },
            "FEED": {
                "description": "Feed types, nutrition, and feeding-related information",
                "examples": ["pellets", "artemia", "protein content", "feeding rate"],
                "rules": [
                    "Include feed composition",
                    "Annotate nutritional values",
                    "Include feeding schedules"
                ],
                "color": "#CDDC39"
            }
        },
        "relation_types": {
            "causes": {
                "description": "Direct causation relationship",
                "examples": ["Vibrio parahaemolyticus causes AHPND", "High temperature causes stress"],
                "rules": ["Only annotate direct, not indirect causation"]
            },
            "infected_by": {
                "description": "Species infected by pathogen",
                "examples": ["Penaeus vannamei infected by WSSV"],
                "rules": ["Link species to specific pathogens"]
            },
            "treated_with": {
                "description": "Treatment or medication relationship",
                "examples": ["AHPND treated with oxytetracycline"],
                "rules": ["Include both preventive and curative treatments"]
            },
            "prevents": {
                "description": "Prevention relationship",
                "examples": ["Probiotics prevent bacterial infections"],
                "rules": ["Include prophylactic measures"]
            },
            "located_in": {
                "description": "Spatial relationship",
                "examples": ["Shrimp farm located in Thailand"],
                "rules": ["Include geographic and facility relationships"]
            },
            "measured_by": {
                "description": "Measurement instrument or method",
                "examples": ["Temperature measured by thermometer"],
                "rules": ["Link measurements to their methods"]
            }
        },
        "topic_categories": {
            "T_DISEASE": {
                "description": "Disease-related content",
                "keywords": ["disease", "infection", "pathogen", "mortality", "outbreak"]
            },
            "T_TREATMENT": {
                "description": "Treatment and medication content",
                "keywords": ["treatment", "medicine", "therapy", "antibiotic", "cure"]
            },
            "T_PREVENTION": {
                "description": "Disease prevention and biosecurity",
                "keywords": ["prevention", "biosecurity", "vaccine", "quarantine"]
            },
            "T_WATER_QUALITY": {
                "description": "Water parameters and management",
                "keywords": ["temperature", "salinity", "pH", "oxygen", "water quality"]
            },
            "T_NUTRITION": {
                "description": "Feed and nutrition content",
                "keywords": ["feed", "nutrition", "protein", "growth", "diet"]
            },
            "T_PRODUCTION": {
                "description": "Production and farming practices",
                "keywords": ["production", "yield", "stocking", "density", "management"]
            },
            "T_DIAGNOSIS": {
                "description": "Diagnostic methods and tools",
                "keywords": ["diagnosis", "detection", "PCR", "histopathology", "test"]
            }
        },
        "annotation_rules": {
            "general": [
                "Select text precisely - include only the entity span",
                "Use canonical forms when possible",
                "Only annotate explicit relationships, not implied ones",
                "Mark uncertain annotations with low confidence",
                "Add notes for ambiguous cases",
                "Avoid overlapping entity annotations",
                "Prefer longer, more specific entities over shorter ones"
            ],
            "entity_guidelines": [
                "Include modifiers that are part of the entity name",
                "Exclude articles (a, an, the) unless part of proper names",
                "Include measurement units with values",
                "Use scientific nomenclature when available"
            ],
            "relation_guidelines": [
                "Only annotate relations explicitly stated in text",
                "Ensure both head and tail entities are correctly identified",
                "Use the most specific relation type available",
                "Avoid redundant relations (e.g., if A causes B and B causes C, don't infer A causes C)"
            ],
            "topic_guidelines": [
                "Select all relevant topics, not just the primary one",
                "Use confidence scores to indicate topic relevance",
                "Consider the document context, not just the sentence"
            ]
        },
        "quality_standards": {
            "confidence_levels": {
                "high": "Clear, unambiguous annotation with strong textual evidence",
                "medium": "Reasonably clear but may have minor ambiguity",
                "low": "Uncertain or ambiguous, requires review"
            },
            "annotation_speed": {
                "target": "15-20 annotations per hour",
                "quality_over_speed": "Accuracy is more important than speed"
            },
            "consistency": [
                "Use the same annotation approach throughout the session",
                "Follow established conventions for your team",
                "Consult guidelines when uncertain"
            ]
        },
        "examples": {
            "good_annotations": [
                {
                    "text": "Penaeus vannamei infected with Vibrio parahaemolyticus showed 80% mortality",
                    "entities": [
                        {"text": "Penaeus vannamei", "label": "SPECIES"},
                        {"text": "Vibrio parahaemolyticus", "label": "PATHOGEN"},
                        {"text": "80% mortality", "label": "MEASUREMENT"}
                    ],
                    "relations": [
                        {"head": "Penaeus vannamei", "tail": "Vibrio parahaemolyticus", "relation": "infected_by"}
                    ]
                }
            ],
            "common_mistakes": [
                "Annotating pronouns instead of the actual entity",
                "Including unnecessary words in entity boundaries",
                "Inferring relations not explicitly stated",
                "Inconsistent entity type assignments"
            ]
        }
    }
    
    return guidelines

# Batch operations
@app.post("/batch/annotations")
async def batch_annotate(
    batch_request: Dict,
    current_user: Dict = Depends(get_current_user)
):
    """Process multiple annotations in batch"""
    
    annotations = batch_request.get("annotations", [])
    results = {
        "processed": 0,
        "successful": 0,
        "failed": 0,
        "errors": [],
        "annotation_ids": []
    }
    
    for annotation_data in annotations:
        try:
            # Simulate annotation processing
            annotation_id = len(mock_data["annotations"]) + 1
            
            mock_data["annotations"].append({
                "id": annotation_id,
                "candidate_id": annotation_data.get("candidate_id"),
                "decision": annotation_data.get("decision", "accept"),
                "entities": annotation_data.get("entities", []),
                "relations": annotation_data.get("relations", []),
                "topics": annotation_data.get("topics", []),
                "confidence": annotation_data.get("confidence", 0.8),
                "annotator": current_user["username"],
                "notes": annotation_data.get("notes", ""),
                "created_at": datetime.utcnow().isoformat(),
                "batch_processed": True
            })
            
            results["annotation_ids"].append(annotation_id)
            results["successful"] += 1
            
        except Exception as e:
            results["errors"].append({
                "candidate_id": annotation_data.get("candidate_id"),
                "error": str(e)
            })
            results["failed"] += 1
        
        results["processed"] += 1
    
    return results

@app.post("/batch/assign")
async def batch_assign_users(
    assignment_request: Dict,
    current_user: Dict = Depends(get_current_user)
):
    """Assign multiple items to users in batch"""
    
    if current_user["role"] not in ["admin", "reviewer"]:
        raise HTTPException(status_code=403, detail="Batch assignment requires admin or reviewer role")
    
    item_ids = assignment_request.get("item_ids", [])
    assigned_user = assignment_request.get("assigned_user")
    
    results = {
        "processed": 0,
        "successful": 0,
        "failed": 0,
        "errors": []
    }
    
    for item_id in item_ids:
        try:
            # Find and update triage item
            for item in mock_data["triage_queue"]:
                if item["id"] == item_id:
                    item["assigned_to"] = assigned_user
                    item["assigned_at"] = datetime.utcnow().isoformat()
                    item["status"] = "assigned"
                    results["successful"] += 1
                    break
            else:
                results["errors"].append({
                    "item_id": item_id,
                    "error": "Item not found"
                })
                results["failed"] += 1
        except Exception as e:
            results["errors"].append({
                "item_id": item_id,
                "error": str(e)
            })
            results["failed"] += 1
        
        results["processed"] += 1
    
    return results

@app.post("/batch/priority")
async def batch_update_priority(
    priority_request: Dict,
    current_user: Dict = Depends(get_current_user)
):
    """Update priority for multiple items in batch"""
    
    if current_user["role"] not in ["admin", "reviewer"]:
        raise HTTPException(status_code=403, detail="Priority updates require admin or reviewer role")
    
    item_ids = priority_request.get("item_ids", [])
    new_priority = priority_request.get("priority_level")
    priority_score = priority_request.get("priority_score")
    
    results = {
        "processed": 0,
        "successful": 0,
        "failed": 0,
        "errors": []
    }
    
    for item_id in item_ids:
        try:
            for item in mock_data["triage_queue"]:
                if item["id"] == item_id:
                    if new_priority:
                        item["priority_level"] = new_priority
                    if priority_score is not None:
                        item["priority_score"] = priority_score
                    results["successful"] += 1
                    break
            else:
                results["errors"].append({
                    "item_id": item_id,
                    "error": "Item not found"
                })
                results["failed"] += 1
        except Exception as e:
            results["errors"].append({
                "item_id": item_id,
                "error": str(e)
            })
            results["failed"] += 1
        
        results["processed"] += 1
    
    return results

@app.delete("/batch/items")
async def batch_delete_items(
    delete_request: Dict,
    current_user: Dict = Depends(get_current_user)
):
    """Delete multiple items in batch"""
    
    if current_user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Batch deletion requires admin role")
    
    item_ids = delete_request.get("item_ids", [])
    item_type = delete_request.get("type", "triage")  # triage, documents, annotations
    
    results = {
        "processed": 0,
        "successful": 0,
        "failed": 0,
        "errors": []
    }
    
    for item_id in item_ids:
        try:
            if item_type == "triage":
                mock_data["triage_queue"] = [
                    item for item in mock_data["triage_queue"] 
                    if item["id"] != item_id
                ]
            elif item_type == "documents":
                mock_data["documents"] = [
                    doc for doc in mock_data["documents"] 
                    if doc["id"] != item_id
                ]
            elif item_type == "annotations":
                mock_data["annotations"] = [
                    ann for ann in mock_data["annotations"] 
                    if ann["id"] != item_id
                ]
            
            results["successful"] += 1
        except Exception as e:
            results["errors"].append({
                "item_id": item_id,
                "error": str(e)
            })
            results["failed"] += 1
        
        results["processed"] += 1
    
    return results

# Inter-annotator agreement metrics
@app.get("/metrics/agreement")
async def get_annotation_agreement(
    annotator1: Optional[str] = Query(None),
    annotator2: Optional[str] = Query(None),
    current_user: Dict = Depends(get_current_user)
):
    """Calculate inter-annotator agreement metrics"""
    
    if current_user["role"] not in ["admin", "reviewer"]:
        raise HTTPException(status_code=403, detail="Agreement metrics require admin or reviewer role")
    
    # Get annotations from both annotators
    annotations = mock_data["annotations"]
    
    if annotator1 and annotator2:
        ann1_data = [ann for ann in annotations if ann.get("annotator") == annotator1]
        ann2_data = [ann for ann in annotations if ann.get("annotator") == annotator2]
    else:
        # Get all annotators for comparison
        annotators = list(set(ann.get("annotator", "unknown") for ann in annotations))
        if len(annotators) < 2:
            return {"error": "Need at least 2 annotators for agreement calculation"}
        ann1_data = [ann for ann in annotations if ann.get("annotator") == annotators[0]]
        ann2_data = [ann for ann in annotations if ann.get("annotator") == annotators[1]]
        annotator1, annotator2 = annotators[0], annotators[1]
    
    # Find overlapping candidates (same text/document)
    overlap_candidates = []
    for ann1 in ann1_data:
        for ann2 in ann2_data:
            if ann1.get("candidate_id") == ann2.get("candidate_id"):
                overlap_candidates.append({
                    "candidate_id": ann1["candidate_id"],
                    "annotator1": ann1,
                    "annotator2": ann2
                })
    
    if not overlap_candidates:
        return {
            "error": "No overlapping annotations found between annotators",
            "annotator1": annotator1,
            "annotator2": annotator2
        }
    
    # Calculate agreement metrics
    metrics = {
        "annotator1": annotator1,
        "annotator2": annotator2,
        "total_overlapping_items": len(overlap_candidates),
        "decision_agreement": 0,
        "entity_agreement": 0,
        "relation_agreement": 0,
        "topic_agreement": 0,
        "confidence_correlation": 0,
        "detailed_agreement": {
            "decision": {"agree": 0, "disagree": 0, "percentage": 0},
            "entities": {"exact_match": 0, "partial_match": 0, "no_match": 0},
            "relations": {"exact_match": 0, "partial_match": 0, "no_match": 0},
            "topics": {"exact_match": 0, "partial_match": 0, "no_match": 0}
        }
    }
    
    decision_agreements = 0
    entity_agreements = 0
    relation_agreements = 0
    topic_agreements = 0
    confidence_pairs = []
    
    for overlap in overlap_candidates:
        ann1, ann2 = overlap["annotator1"], overlap["annotator2"]
        
        # Decision agreement
        if ann1.get("decision") == ann2.get("decision"):
            decision_agreements += 1
            metrics["detailed_agreement"]["decision"]["agree"] += 1
        else:
            metrics["detailed_agreement"]["decision"]["disagree"] += 1
        
        # Entity agreement (simplified - exact match)
        entities1 = set(tuple(sorted(e.items())) for e in ann1.get("entities", []))
        entities2 = set(tuple(sorted(e.items())) for e in ann2.get("entities", []))
        
        if entities1 == entities2:
            entity_agreements += 1
            metrics["detailed_agreement"]["entities"]["exact_match"] += 1
        elif entities1 & entities2:  # Some overlap
            metrics["detailed_agreement"]["entities"]["partial_match"] += 1
        else:
            metrics["detailed_agreement"]["entities"]["no_match"] += 1
        
        # Relation agreement
        relations1 = set(tuple(sorted(r.items())) for r in ann1.get("relations", []))
        relations2 = set(tuple(sorted(r.items())) for r in ann2.get("relations", []))
        
        if relations1 == relations2:
            relation_agreements += 1
            metrics["detailed_agreement"]["relations"]["exact_match"] += 1
        elif relations1 & relations2:
            metrics["detailed_agreement"]["relations"]["partial_match"] += 1
        else:
            metrics["detailed_agreement"]["relations"]["no_match"] += 1
        
        # Topic agreement
        topics1 = set(t.get("topic") for t in ann1.get("topics", []))
        topics2 = set(t.get("topic") for t in ann2.get("topics", []))
        
        if topics1 == topics2:
            topic_agreements += 1
            metrics["detailed_agreement"]["topics"]["exact_match"] += 1
        elif topics1 & topics2:
            metrics["detailed_agreement"]["topics"]["partial_match"] += 1
        else:
            metrics["detailed_agreement"]["topics"]["no_match"] += 1
        
        # Confidence correlation data
        confidence_pairs.append((
            ann1.get("confidence", 0.5),
            ann2.get("confidence", 0.5)
        ))
    
    # Calculate percentages
    total_items = len(overlap_candidates)
    metrics["decision_agreement"] = round(decision_agreements / total_items, 3)
    metrics["entity_agreement"] = round(entity_agreements / total_items, 3)
    metrics["relation_agreement"] = round(relation_agreements / total_items, 3)
    metrics["topic_agreement"] = round(topic_agreements / total_items, 3)
    
    metrics["detailed_agreement"]["decision"]["percentage"] = round(
        metrics["detailed_agreement"]["decision"]["agree"] / total_items * 100, 1
    )
    
    # Simple correlation coefficient for confidence
    if confidence_pairs:
        import statistics
        conf1_values = [pair[0] for pair in confidence_pairs]
        conf2_values = [pair[1] for pair in confidence_pairs]
        
        if len(set(conf1_values)) > 1 and len(set(conf2_values)) > 1:
            # Simplified correlation calculation
            mean1, mean2 = statistics.mean(conf1_values), statistics.mean(conf2_values)
            
            numerator = sum((x - mean1) * (y - mean2) for x, y in confidence_pairs)
            denom1 = sum((x - mean1) ** 2 for x in conf1_values)
            denom2 = sum((y - mean2) ** 2 for y in conf2_values)
            
            if denom1 > 0 and denom2 > 0:
                metrics["confidence_correlation"] = round(
                    numerator / (denom1 * denom2) ** 0.5, 3
                )
    
    # Cohen's Kappa for decision agreement (simplified)
    total = metrics["detailed_agreement"]["decision"]["agree"] + metrics["detailed_agreement"]["decision"]["disagree"]
    if total > 0:
        observed_agreement = metrics["detailed_agreement"]["decision"]["agree"] / total
        # Simplified expected agreement (assumes 50/50 distribution)
        expected_agreement = 0.5
        
        if expected_agreement < 1:
            kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
            metrics["cohens_kappa"] = round(kappa, 3)
    
    return metrics

@app.get("/metrics/quality")
async def get_annotation_quality_metrics(
    annotator: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    current_user: Dict = Depends(get_current_user)
):
    """Get annotation quality metrics for individual annotators"""
    
    annotations = mock_data["annotations"]
    
    # Filter by annotator
    if annotator:
        annotations = [ann for ann in annotations if ann.get("annotator") == annotator]
    
    # Filter by date
    if date_from or date_to:
        filtered_annotations = []
        for ann in annotations:
            ann_date = ann.get("created_at", datetime.utcnow().isoformat())
            if date_from and ann_date < date_from:
                continue
            if date_to and ann_date > date_to:
                continue
            filtered_annotations.append(ann)
        annotations = filtered_annotations
    
    if not annotations:
        return {"error": "No annotations found for the specified criteria"}
    
    # Calculate quality metrics
    total_annotations = len(annotations)
    decisions = [ann.get("decision", "unknown") for ann in annotations]
    confidences = [ann.get("confidence", 0.5) for ann in annotations]
    
    # Entity counts
    entity_counts = []
    relation_counts = []
    topic_counts = []
    
    for ann in annotations:
        entity_counts.append(len(ann.get("entities", [])))
        relation_counts.append(len(ann.get("relations", [])))
        topic_counts.append(len(ann.get("topics", [])))
    
    import statistics
    
    metrics = {
        "annotator": annotator or "all",
        "total_annotations": total_annotations,
        "decision_distribution": {
            "accept": decisions.count("accept"),
            "reject": decisions.count("reject"),
            "modify": decisions.count("modify")
        },
        "confidence_stats": {
            "mean": round(statistics.mean(confidences), 3),
            "median": round(statistics.median(confidences), 3),
            "std_dev": round(statistics.stdev(confidences) if len(confidences) > 1 else 0, 3)
        },
        "annotation_complexity": {
            "avg_entities_per_annotation": round(statistics.mean(entity_counts), 2),
            "avg_relations_per_annotation": round(statistics.mean(relation_counts), 2),
            "avg_topics_per_annotation": round(statistics.mean(topic_counts), 2)
        },
        "quality_indicators": {
            "high_confidence_annotations": len([c for c in confidences if c >= 0.8]),
            "low_confidence_annotations": len([c for c in confidences if c < 0.5]),
            "annotations_with_notes": len([ann for ann in annotations if ann.get("notes")])
        }
    }
    
    # Calculate percentages
    metrics["decision_distribution"]["accept_rate"] = round(
        metrics["decision_distribution"]["accept"] / total_annotations * 100, 1
    )
    metrics["decision_distribution"]["reject_rate"] = round(
        metrics["decision_distribution"]["reject"] / total_annotations * 100, 1
    )
    
    return metrics

# User info
@app.get("/users/me")
async def get_current_user_info(current_user: Dict = Depends(get_current_user)):
    """Get current user information"""
    return {
        "user": current_user,
        "permissions": {
            "can_annotate": True,
            "can_review": current_user["role"] in ["admin", "reviewer"],
            "can_export": current_user["role"] in ["admin", "reviewer"],
            "can_manage_users": current_user["role"] == "admin"
        }
    }

# Favicon (to stop 404 errors)
@app.get("/favicon.ico")
async def favicon():
    """Return empty favicon to stop 404 errors"""
    return JSONResponse(content={}, status_code=204)

if __name__ == "__main__":
    logger.info("🚀 Starting Mock API Server with ALL Frontend Endpoints")
    logger.info("📊 API Documentation: http://localhost:8000/docs")
    logger.info("🔍 Health Check: http://localhost:8000/health")
    
    uvicorn.run(
        "mock_api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )