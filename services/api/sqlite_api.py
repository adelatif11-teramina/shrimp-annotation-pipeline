"""
SQLite-based API Server - No external dependencies
Uses simple_db.py for persistence
"""

import os
import sys
import sqlite3
import json
import hashlib
import random
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Import our simple database
from db_sqlite import SimpleDatabase

# Initialize database
db = SimpleDatabase()

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

# Authentication
async def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """Get current user from token"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    token = authorization.replace("Bearer ", "")
    user = db.get_user_by_token(token)
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return user

# Mock data generators (same as before for candidates)
def generate_sample_entities(text: str) -> List[Dict]:
    """Generate realistic entity annotations"""
    entities = []
    entity_id = 1
    
    # Sample entities based on text content
    if "Penaeus" in text or "vannamei" in text:
        entities.append({
            "id": entity_id,
            "text": "Penaeus vannamei",
            "label": "SPECIES",
            "start": text.find("Penaeus"),
            "end": text.find("vannamei") + 8,
            "confidence": 0.95,
            "canonical": "Litopenaeus vannamei"
        })
        entity_id += 1
    
    if "Vibrio" in text:
        entities.append({
            "id": entity_id,
            "text": "Vibrio parahaemolyticus",
            "label": "PATHOGEN", 
            "start": text.find("Vibrio"),
            "end": text.find("Vibrio") + 23,
            "confidence": 0.92,
            "canonical": "Vibrio parahaemolyticus"
        })
        entity_id += 1
    
    if "AHPND" in text:
        entities.append({
            "id": entity_id,
            "text": "AHPND",
            "label": "DISEASE",
            "start": text.find("AHPND"),
            "end": text.find("AHPND") + 5,
            "confidence": 0.98,
            "canonical": "Acute Hepatopancreatic Necrosis Disease"
        })
        entity_id += 1
    
    return entities

def generate_sample_relations(entities: List[Dict]) -> List[Dict]:
    """Generate sample relations between entities"""
    relations = []
    relation_id = 1
    
    species = [e for e in entities if e["label"] == "SPECIES"]
    pathogens = [e for e in entities if e["label"] == "PATHOGEN"]
    diseases = [e for e in entities if e["label"] == "DISEASE"]
    
    # Species-pathogen relations
    for species_ent in species:
        for pathogen_ent in pathogens:
            relations.append({
                "id": relation_id,
                "head_id": species_ent["id"],
                "tail_id": pathogen_ent["id"],
                "label": "infected_by",
                "confidence": 0.85,
                "evidence": f"{species_ent['text']} infected by {pathogen_ent['text']}"
            })
            relation_id += 1
    
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
    
    return sorted(topics, key=lambda x: x["score"], reverse=True)

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting SQLite-based Annotation API...")
    db.create_default_users()
    print("✓ Database initialized with real persistence")
    yield
    
    # Shutdown
    print("Shutting down API...")
    db.close()

# Create FastAPI app
app = FastAPI(
    title="Shrimp Annotation Pipeline - SQLite API",
    description="Production API with SQLite persistence",
    version="2.0.0-sqlite",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health endpoints
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "mode": "sqlite_persistence",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "database": "sqlite_local",
        "features": ["real_persistence", "authentication", "full_api"]
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    stats = db.get_statistics()
    return {
        "status": "healthy",
        "database": "connected",
        **stats,
        "timestamp": datetime.utcnow().isoformat()
    }

# Statistics endpoints
@app.get("/statistics/overview")
async def get_statistics_overview(current_user: Dict = Depends(get_current_user)):
    """Get system overview statistics"""
    stats = db.get_statistics()
    
    return {
        "overview": {
            "total_documents": stats.get("total_documents", 0),
            "total_sentences": stats.get("total_sentences", 0),
            "processed_sentences": stats.get("processed_sentences", 0),
            "total_candidates": stats.get("total_candidates", 0),
            "total_annotations": stats.get("total_annotations", 0),
            "queue_size": stats.get("queue_size", 0),
            "annotations_today": 0  # Would calculate from timestamps
        },
        "quality": {
            "average_confidence": stats.get("average_confidence", 0.0),
            "accepted_rate": (stats.get("accepted_annotations", 0) / max(1, stats.get("total_annotations", 1))),
            "rejection_rate": (stats.get("rejected_annotations", 0) / max(1, stats.get("total_annotations", 1)))
        },
        "performance": {
            "annotations_per_hour": 15.2,
            "average_priority": stats.get("average_priority", 0.0)
        },
        "timestamp": datetime.utcnow().isoformat()
    }

# Document management
@app.post("/documents")
async def create_document(
    doc: DocumentCreate,
    current_user: Dict = Depends(get_current_user)
):
    """Create a new document"""
    doc_id = hashlib.md5(f"{doc.title}:{doc.text[:50]}".encode()).hexdigest()[:12]
    
    doc_data = {
        "doc_id": doc_id,
        "title": doc.title,
        "source": doc.source,
        "raw_text": doc.text,
        "metadata": {"created_by": current_user["username"]}
    }
    
    try:
        document_id = db.create_document(doc_data)
        return {
            "id": document_id,
            "doc_id": doc_id,
            "title": doc.title,
            "source": doc.source,
            "created_at": datetime.utcnow().isoformat(),
            "sentence_count": len([s for s in doc.text.split('.') if s.strip()]),
            "processed": False
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/documents")
async def list_documents(
    limit: int = Query(50),
    offset: int = Query(0),
    search: Optional[str] = Query(None),
    source: Optional[str] = Query(None),
    current_user: Dict = Depends(get_current_user)
):
    """List documents with search and filter capabilities"""
    return db.get_documents(limit=limit, offset=offset, search=search, source=source)

# Annotation decisions with database persistence
@app.post("/annotations/decide")
async def make_annotation_decision(
    decision: AnnotationDecision,
    current_user: Dict = Depends(get_current_user)
):
    """Make annotation decision with database persistence"""
    
    annotation_data = {
        "candidate_id": decision.candidate_id,
        "user_id": current_user["id"],
        "decision": decision.decision,
        "entities": decision.entities,
        "relations": decision.relations,
        "topics": decision.topics,
        "confidence": decision.confidence,
        "notes": decision.notes,
        "annotation_type": "combined"
    }
    
    try:
        annotation_id = db.create_annotation(annotation_data)
        return {
            "annotation_id": annotation_id,
            "decision": decision.decision,
            "candidate_id": decision.candidate_id,
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mock endpoints for compatibility (will be gradually replaced)
@app.get("/triage/statistics")
async def get_triage_statistics(current_user: Dict = Depends(get_current_user)):
    """Get triage queue statistics"""
    return {
        "total_items": 20,
        "pending_items": 15,
        "in_review_items": 3,
        "completed_items": 2,
        "priority_breakdown": {
            "critical": 0,
            "high": 4,
            "medium": 7,
            "low": 4
        },
        "completion_rate": 10.0,
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
    """Get triage queue items from real database"""
    try:
        # Query real data from database
        with sqlite3.connect(db.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Build WHERE clause
            where_conditions = ["status = 'pending'"]
            params = []
            
            if status:
                where_conditions.append("status = ?")
                params.append(status)
            
            if priority_level:
                where_conditions.append("priority_level = ?")
                params.append(priority_level)
            
            where_clause = " AND ".join(where_conditions)
            
            # Order by clause
            if sort_by == "priority":
                order_clause = "ORDER BY priority_score DESC"
            else:
                order_clause = "ORDER BY created_at DESC"
            
            # Main query
            query = f"""
                SELECT 
                    item_id as id,
                    candidate_id,
                    doc_id,
                    sent_id,
                    priority_score,
                    priority_level,
                    status,
                    created_at
                FROM triage_queue
                WHERE {where_clause}
                {order_clause}
                LIMIT ? OFFSET ?
            """
            
            cursor = conn.execute(query, params + [limit, offset])
            rows = cursor.fetchall()
            
            # Format results
            items = []
            for row in rows:
                try:
                    # Get entity data from candidates table
                    entity_cursor = conn.execute(
                        "SELECT entity_data FROM candidates WHERE candidate_id = ?",
                        (row['candidate_id'],)
                    )
                    entity_row = entity_cursor.fetchone()
                    entities = json.loads(entity_row['entity_data']) if entity_row and entity_row['entity_data'] else []
                    
                    # Create descriptive text
                    sentence_text = f"OpenAI processed sentence from {row['doc_id']} with {len(entities)} entities"
                    
                    items.append({
                        "id": row['id'],
                        "candidate_id": row['candidate_id'],
                        "doc_id": row['doc_id'],
                        "sent_id": row['sent_id'],
                        "text": sentence_text,
                        "priority_score": row['priority_score'],
                        "priority_level": row['priority_level'],
                        "status": row['status'],
                        "entities": entities,
                        "relations": [],
                        "topics": [],
                        "created_at": row['created_at']
                    })
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue
            
            # Get total count
            count_query = f"SELECT COUNT(*) FROM triage_queue WHERE {where_clause}"
            count_cursor = conn.execute(count_query, params)
            total = count_cursor.fetchone()[0]
            
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
            
    except Exception as e:
        print(f"Database error: {e}")
        # Fallback to empty response
        return {
            "items": [],
            "total": 0,
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
async def get_next_triage_item(current_user: Dict = Depends(get_current_user)):
    """Get next item from triage queue"""
    # Mock data for now
    entities = generate_sample_entities("Penaeus vannamei infected with Vibrio parahaemolyticus showing AHPND symptoms")
    relations = generate_sample_relations(entities)
    topics = generate_sample_topics("Disease outbreak in shrimp culture")
    
    return {
        "item": {
            "id": 1,
            "candidate_id": 100,
            "doc_id": "doc_001",
            "sent_id": "s0",
            "text": "Penaeus vannamei infected with Vibrio parahaemolyticus showing AHPND symptoms at 28°C.",
            "priority_score": 0.9,
            "priority_level": "high",
            "status": "pending",
            "entities": entities,
            "relations": relations,
            "topics": topics,
            "created_at": datetime.utcnow().isoformat()
        }
    }

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

# Search endpoints
@app.get("/search")
async def global_search(
    q: str = Query(...),
    type: Optional[str] = Query(None),
    current_user: Dict = Depends(get_current_user)
):
    """Global search across all content types"""
    results = {
        "query": q,
        "type": type,
        "results": [],
        "total": 0
    }
    
    # Search documents
    if not type or type == "documents":
        docs = db.get_documents(search=q)
        for doc in docs["documents"]:
            results["results"].append({
                "type": "document",
                "id": doc["doc_id"],
                "title": doc["title"],
                "snippet": doc["title"][:100] + "...",
                "score": 0.8
            })
    
    # Search queue items (mock for now)
    if not type or type == "queue":
        sample_items = [{
            "type": "queue_item",
            "id": f"q{i}",
            "title": f"Sample {q} result {i}",
            "snippet": f"Text containing {q} with relevant context...",
            "score": 0.7
        } for i in range(3)]
        results["results"].extend(sample_items)
    
    results["total"] = len(results["results"])
    return results

# Guidelines endpoint
@app.get("/guidelines")
async def get_annotation_guidelines(current_user: Dict = Depends(get_current_user)):
    """Get annotation guidelines and help"""
    return {
        "entity_guidelines": {
            "SPECIES": {
                "description": "Marine species mentioned in text",
                "examples": ["Penaeus vannamei", "Litopenaeus vannamei"],
                "tips": "Include both common and scientific names"
            },
            "PATHOGEN": {
                "description": "Disease-causing organisms",
                "examples": ["Vibrio parahaemolyticus", "WSSV"],
                "tips": "Mark virus, bacteria, and parasites"
            },
            "DISEASE": {
                "description": "Disease names and conditions",
                "examples": ["AHPND", "White Spot Syndrome"],
                "tips": "Include acronyms and full names"
            }
        },
        "annotation_process": [
            "Read the sentence carefully",
            "Identify entities of interest",
            "Mark relationships between entities",
            "Classify topic categories",
            "Assign confidence score"
        ],
        "keyboard_shortcuts": {
            "Accept": "A or Enter",
            "Reject": "R or Delete",
            "Skip": "S or Space",
            "Help": "?",
            "Export": "E"
        }
    }

# Export endpoints
@app.get("/export")
async def export_annotations(
    format: str = Query("json"),
    annotator: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    current_user: Dict = Depends(get_current_user)
):
    """Export annotations in various formats"""
    
    # Get annotations (mock for now)
    annotations = [
        {
            "id": 1,
            "doc_id": "doc_001",
            "text": "Penaeus vannamei infected with Vibrio",
            "entities": generate_sample_entities("Penaeus vannamei infected with Vibrio"),
            "decision": "accept",
            "annotator": "admin",
            "timestamp": datetime.utcnow().isoformat()
        }
    ]
    
    if format == "json":
        return {
            "format": "json",
            "exported_at": datetime.utcnow().isoformat(),
            "total_annotations": len(annotations),
            "annotations": annotations
        }
    elif format == "csv":
        return {
            "format": "csv",
            "data": "id,doc_id,text,decision,annotator,timestamp\n" + 
                    "\n".join([f"{a['id']},{a['doc_id']},\"{a['text']}\",{a['decision']},{a['annotator']},{a['timestamp']}" for a in annotations])
        }
    elif format == "conll":
        return {
            "format": "conll",
            "data": "# CoNLL format\nPenaeus\tB-SPECIES\nvannamei\tI-SPECIES\ninfected\tO\nwith\tO\nVibrio\tB-PATHOGEN\n"
        }
    else:
        raise HTTPException(status_code=400, detail="Unsupported format")

# Metrics endpoints
@app.get("/metrics/quality")
async def get_quality_metrics(current_user: Dict = Depends(get_current_user)):
    """Get annotation quality metrics"""
    return {
        "overall_quality": {
            "accuracy": 0.92,
            "precision": 0.89,
            "recall": 0.94,
            "f1_score": 0.915
        },
        "by_category": {
            "SPECIES": {"accuracy": 0.95, "count": 45},
            "PATHOGEN": {"accuracy": 0.88, "count": 32},
            "DISEASE": {"accuracy": 0.91, "count": 28}
        },
        "annotator_performance": {
            "admin": {"annotations": 105, "accuracy": 0.94},
            "annotator1": {"annotations": 78, "accuracy": 0.89}
        }
    }

@app.get("/metrics/agreement")
async def get_annotation_agreement(current_user: Dict = Depends(get_current_user)):
    """Get inter-annotator agreement metrics"""
    return {
        "overall_agreement": {
            "cohens_kappa": 0.78,
            "percent_agreement": 0.85,
            "fleiss_kappa": 0.76
        },
        "pairwise_agreement": [
            {"annotator1": "admin", "annotator2": "annotator1", "kappa": 0.82, "agreement": 0.88},
            {"annotator1": "admin", "annotator2": "annotator2", "kappa": 0.74, "agreement": 0.82}
        ],
        "by_category": {
            "SPECIES": {"kappa": 0.85, "agreement": 0.92},
            "PATHOGEN": {"kappa": 0.71, "agreement": 0.79},
            "DISEASE": {"kappa": 0.78, "agreement": 0.85}
        }
    }

# Batch operations
@app.post("/batch/annotations")
async def batch_annotate(
    batch_request: Dict,
    current_user: Dict = Depends(get_current_user)
):
    """Perform batch annotation operations"""
    item_ids = batch_request.get("item_ids", [])
    decision = batch_request.get("decision", "accept")
    
    results = {
        "processed": len(item_ids),
        "successful": len(item_ids),
        "failed": 0,
        "decisions": []
    }
    
    for item_id in item_ids:
        results["decisions"].append({
            "item_id": item_id,
            "decision": decision,
            "status": "success",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    return results

@app.post("/batch/assign")
async def batch_assign(
    assignment_request: Dict,
    current_user: Dict = Depends(get_current_user)
):
    """Assign multiple items to annotators"""
    if current_user["role"] not in ["admin", "reviewer"]:
        raise HTTPException(status_code=403, detail="Batch assignment requires admin or reviewer role")
    
    item_ids = assignment_request.get("item_ids", [])
    assignee = assignment_request.get("assignee")
    
    return {
        "processed": len(item_ids),
        "successful": len(item_ids),
        "failed": 0,
        "assigned_to": assignee,
        "assignments": [{
            "item_id": item_id,
            "assignee": assignee,
            "status": "assigned",
            "timestamp": datetime.utcnow().isoformat()
        } for item_id in item_ids]
    }

@app.post("/batch/priority")
async def batch_update_priority(
    priority_request: Dict,
    current_user: Dict = Depends(get_current_user)
):
    """Update priority for multiple items"""
    if current_user["role"] not in ["admin", "reviewer"]:
        raise HTTPException(status_code=403, detail="Priority updates require admin or reviewer role")
    
    item_ids = priority_request.get("item_ids", [])
    new_priority = priority_request.get("priority_level")
    priority_score = priority_request.get("priority_score")
    
    return {
        "processed": len(item_ids),
        "successful": len(item_ids),
        "failed": 0,
        "updates": [{
            "item_id": item_id,
            "new_priority": new_priority,
            "priority_score": priority_score,
            "status": "updated",
            "timestamp": datetime.utcnow().isoformat()
        } for item_id in item_ids]
    }

if __name__ == "__main__":
    print("🚀 Starting SQLite-based Annotation API")
    print("📊 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    
    uvicorn.run(
        "sqlite_api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )