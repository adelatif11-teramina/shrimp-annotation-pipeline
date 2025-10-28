"""
REST API Service for Annotation Pipeline

Provides REST endpoints for ML pipeline integration, candidate generation,
annotation management, and data export.
"""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn
import time

# Setup logging before other imports
try:
    from utils.logging_config import setup_logging, get_logger, LogOperation
    from config.settings import get_settings

    # Initialize logging
    settings = get_settings()
    setup_logging(
        level=settings.log_level,
        log_file=f"logs/api_{datetime.now().strftime('%Y%m%d')}.log",
        json_format=settings.is_production,
        enable_sensitive_filter=True
    )
    logger = get_logger(__name__)
except ImportError as e:
    # Fallback for Railway deployment
    import os
    from pydantic_settings import BaseSettings
    
    class SimpleSettings(BaseSettings):
        environment: str = os.getenv("ENVIRONMENT", "production")
        log_level: str = os.getenv("LOG_LEVEL", "INFO")
        api_host: str = os.getenv("API_HOST", "0.0.0.0")
        api_port: int = int(os.getenv("PORT", 8000))
        jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "temp-key-for-railway")
        cors_origins: List[str] = ["*"]
        rate_limit_enabled: bool = False
        
        @property
        def is_production(self):
            return self.environment == "production"
    
    settings = SimpleSettings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper()))
    logger = logging.getLogger(__name__)

# Import our services
import sys
pipeline_root = Path(__file__).parent.parent.parent
sys.path.append(str(pipeline_root))

# Optional imports for Railway deployment
try:
    from services.candidates.llm_candidate_generator import LLMCandidateGenerator
    from services.candidates.triplet_workflow import TripletWorkflow
except ImportError:
    LLMCandidateGenerator = None
    TripletWorkflow = None
    logger.warning("LLM Candidate Generator or triplet workflow not available")

try:
    from services.ingestion.document_ingestion import DocumentIngestionService, Document
except ImportError:
    DocumentIngestionService = None
    Document = None
    logger.warning("Document Ingestion Service not available")

try:
    from services.triage.triage_prioritization import TriagePrioritizationEngine, TriageItem
except ImportError:
    TriagePrioritizationEngine = None
    TriageItem = None
    logger.warning("Triage Prioritization Engine not available")

try:
    from services.rules.rule_based_annotator import ShimpAquacultureRuleEngine
except ImportError:
    ShimpAquacultureRuleEngine = None
    logger.warning("Rule-based Annotator not available")

# Pydantic models for API
class DocumentRequest(BaseModel):
    doc_id: str
    text: str
    title: Optional[str] = None
    source: str = "manual"
    metadata: Dict[str, Any] = {}

class SentenceRequest(BaseModel):
    doc_id: str
    sent_id: str
    text: str
    title: Optional[str] = None

class BatchSentenceRequest(BaseModel):
    sentences: List[SentenceRequest]
    batch_size: int = 10

class CandidateResponse(BaseModel):
    doc_id: str
    sent_id: str
    candidates: Dict[str, Any]
    rule_results: Optional[Dict[str, Any]] = None
    triage_score: Optional[float] = None
    processing_time: float

class AnnotationDecision(BaseModel):
    item_id: str
    decision: str  # accepted, rejected, modified
    final_annotation: Optional[Dict[str, Any]] = None
    annotator: str
    notes: Optional[str] = None

class ExportRequest(BaseModel):
    format: str = "jsonl"  # jsonl, conll, bio, scibert
    doc_ids: Optional[List[str]] = None
    date_range: Optional[Dict[str, str]] = None
    annotator: Optional[str] = None

# Global service instances
app = FastAPI(
    title="Shrimp Annotation Pipeline API",
    description="REST API for human-in-the-loop annotation pipeline",
    version="1.0.0"
)

# Logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all API requests with timing and status"""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Extract user info if available
    user_id = None
    try:
        # Try to get user from Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            from utils.auth import jwt_manager
            token = auth_header.split(" ")[1]
            payload = jwt_manager.decode_token(token)
            user_id = payload.get("user_id")
    except Exception:
        # Ignore auth errors in logging
        pass
    
    # Log the request
    from utils.logging_config import log_api_request
    log_api_request(
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration=duration,
        user_id=user_id
    )
    
    return response

# CORS middleware
# settings already imported and initialized above

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting middleware
if settings.rate_limit_enabled:
    from utils.rate_limiting import RateLimitMiddleware, create_rate_limiter, RATE_LIMIT_CONFIGS
    
    rate_limiter = create_rate_limiter(settings.redis_url if hasattr(settings, 'redis_url') else None)
    rate_limit_rules = RATE_LIMIT_CONFIGS.get(settings.environment, RATE_LIMIT_CONFIGS["development"])
    
    rate_limit_middleware = RateLimitMiddleware(rate_limiter, rate_limit_rules)
    app.middleware("http")(rate_limit_middleware)

# Serve React frontend static files
ui_build_path = pipeline_root / "ui" / "build"
if ui_build_path.exists():
    # Serve static files
    app.mount("/static", StaticFiles(directory=str(ui_build_path / "static")), name="static")
    
    @app.get("/")
    async def serve_frontend():
        """Serve the React frontend"""
        index_file = ui_build_path / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"message": "🦐 Shrimp Annotation Pipeline API", "frontend": "not_built"}
else:
    @app.get("/")
    async def api_root():
        """API root when frontend not available"""
        return {
            "message": "🦐 Shrimp Annotation Pipeline API",
            "version": "1.0.0",
            "status": "running",
            "frontend": "not_built",
            "endpoints": {
                "health": "/health",
                "docs": "/docs",
                "openapi": "/openapi.json"
            }
        }

# Include authentication endpoints (optional)
try:
    from services.api.auth_endpoints import router as auth_router
    app.include_router(auth_router)
except ImportError:
    logger.warning("Authentication endpoints not available")

# Service instances (will be initialized on startup)
llm_generator: Optional[LLMCandidateGenerator] = None
triplet_workflow: Optional[TripletWorkflow] = None
ingestion_service: Optional[DocumentIngestionService] = None
triage_engine: Optional[TriagePrioritizationEngine] = None
rule_engine: Optional[ShimpAquacultureRuleEngine] = None


def reset_service_state() -> None:
    """Reset singleton service instances (testing helper)."""
    global llm_generator, triplet_workflow, ingestion_service, triage_engine, rule_engine
    llm_generator = None
    triplet_workflow = None
    ingestion_service = None
    triage_engine = None
    rule_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global llm_generator, triplet_workflow, ingestion_service, triage_engine, rule_engine
    
    # Start monitoring
    from utils.monitoring import monitor
    await monitor.start_monitoring(interval=60)
    
    logger.info("Initializing annotation pipeline services...")
    
    # Initialize LLM generator (will need API key from environment)
    try:
        llm_provider = "openai" if settings.openai_api_key else "ollama"
        llm_kwargs: Dict[str, Any] = {
            "provider": llm_provider,
            "model": settings.openai_model if llm_provider == "openai" else settings.ollama_model,
            "cache_dir": pipeline_root / "data/candidates/.cache"
        }

        if llm_provider == "openai":
            llm_kwargs["api_key"] = settings.openai_api_key

        llm_generator = LLMCandidateGenerator(**llm_kwargs)
        logger.info("✓ LLM candidate generator initialized (%s)", llm_provider)
    except Exception as e:
        logger.warning(f"LLM generator initialization failed: {e}")
        llm_generator = None
    
    # Initialize ingestion service with smart chunking for production
    from services.ingestion.chunking_integration import ImprovedDocumentIngestionService
    ingestion_service = ImprovedDocumentIngestionService(
        chunking_mode="smart_paragraph",
        smart_chunk_length=(150, 400)
    )
    logger.info("✓ Document ingestion service initialized with smart chunking")
    
    # Initialize rule engine
    rule_engine = ShimpAquacultureRuleEngine()
    logger.info("✓ Rule-based annotation engine initialized")
    
    # Initialize triage engine
    triage_engine = TriagePrioritizationEngine(
        gold_store_path=pipeline_root / "data/gold"
    )
    logger.info("✓ Triage prioritization engine initialized")
    
    # Auto-populate queue on startup if empty
    try:
        if len(triage_engine.queue) == 0:
            logger.info("🔄 Queue is empty on startup, populating from documents...")
            raw_dir = pipeline_root / "data/raw"
            if raw_dir.exists() and list(raw_dir.glob("*.txt")):
                # Use a simple approach: populate directly without API call
                doc_files = list(raw_dir.glob("*.txt"))
                total_candidates = 0
                
                for doc_file in doc_files[:3]:  # Limit to first 3 files for startup
                    logger.info(f"🔄 Startup processing: {doc_file.name}")
                    try:
                        document = ingestion_service.ingest_text_file(doc_file, source="startup", title=doc_file.stem)
                        
                        for sentence in document.sentences[:5]:  # Limit to first 5 sentences per doc
                            rule_result = rule_engine.process_sentence(document.doc_id, sentence.sent_id, sentence.text)
                            
                            candidates = {"text": sentence.text, "entities": rule_result.get("entities", [])}
                            doc_metadata = {"doc_id": document.doc_id, "sent_id": sentence.sent_id, "title": document.title, "source": "startup"}
                            
                            triage_engine.add_candidates(candidates, doc_metadata, rule_result)
                            total_candidates += 1
                            
                    except Exception as e:
                        logger.warning(f"Failed to process {doc_file.name} on startup: {e}")
                        continue
                
                logger.info(f"✅ Startup queue population complete: {total_candidates} candidates")
            else:
                logger.info("📭 No documents found for startup queue population")
    except Exception as e:
        logger.warning(f"Startup queue population failed: {e}")

    if llm_generator and TripletWorkflow:
        triplet_workflow = TripletWorkflow(llm_generator, rule_engine)
        logger.info("✓ Triplet workflow orchestrator initialized")
    else:
        triplet_workflow = None
        logger.warning("Triplet workflow not initialized (dependencies missing)")
    
    logger.info("All services initialized successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    from utils.monitoring import monitor
    await monitor.stop_monitoring()
    logger.info("Application shutdown complete")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint with circuit breaker status"""
    # Check circuit breaker status
    from utils.error_handling import error_handler
    circuit_status = {}
    
    for name, breaker in error_handler.circuit_breakers.items():
        circuit_status[name] = {
            "state": breaker.state.value,
            "failure_count": breaker.failure_count,
            "last_failure_time": breaker.last_failure_time
        }
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "environment": settings.environment,
        "services": {
            "llm_generator": llm_generator is not None,
            "triplet_workflow": triplet_workflow is not None,
            "ingestion_service": ingestion_service is not None,
            "triage_engine": triage_engine is not None,
            "rule_engine": rule_engine is not None
        },
        "circuit_breakers": circuit_status,
        "rate_limiting": {
            "enabled": settings.rate_limit_enabled,
            "per_minute": getattr(settings, 'rate_limit_per_minute', 'N/A'),
            "per_hour": getattr(settings, 'rate_limit_per_hour', 'N/A')
        }
    }

@app.get("/ready")
async def readiness_check():
    """Readiness check for kubernetes/docker deployments"""
    services_ready = {
        "llm_generator": llm_generator is not None,
        "triplet_workflow": triplet_workflow is not None,
        "ingestion_service": ingestion_service is not None,
        "triage_engine": triage_engine is not None,
        "rule_engine": rule_engine is not None
    }
    
    # Check if any circuit breakers are open
    from utils.error_handling import error_handler
    critical_breakers_open = []
    
    for name, breaker in error_handler.circuit_breakers.items():
        if breaker.state.value == "open" and name in ["openai_api", "ollama_api"]:
            critical_breakers_open.append(name)
    
    all_ready = all(services_ready.values()) and not critical_breakers_open
    
    response = {
        "ready": all_ready,
        "timestamp": datetime.now().isoformat(),
        "services": services_ready,
        "critical_breakers_open": critical_breakers_open
    }
    
    if not all_ready:
        raise HTTPException(status_code=503, detail=response)
    
    return response

@app.get("/metrics")
async def get_metrics():
    """Get application metrics in Prometheus format"""
    from utils.monitoring import monitor
    from fastapi import Request
    from fastapi.responses import Response
    
    # Return JSON format by default
    return monitor.metrics.get_metrics_summary()

@app.get("/metrics/prometheus")
async def get_prometheus_metrics():
    """Get metrics in Prometheus format"""
    from utils.monitoring import monitor
    from fastapi.responses import Response
    
    return Response(
        content=monitor.metrics.get_prometheus_metrics(),
        media_type="text/plain"
    )

@app.get("/metrics/summary")
async def get_metrics_summary():
    """Get metrics summary in JSON format"""
    from utils.monitoring import monitor
    return monitor.metrics.get_metrics_summary()

@app.get("/alerts")
async def get_alerts():
    """Get active alerts"""
    from utils.monitoring import monitor
    return {
        "active_alerts": monitor.alert_manager.active_alerts,
        "alert_count": len(monitor.alert_manager.active_alerts)
    }

@app.get("/health/detailed") 
async def detailed_health_check():
    """Detailed health check with component status"""
    from utils.monitoring import monitor
    health_status = await monitor.health_checker.check_system_health()
    
    return {
        "status": health_status.status,
        "timestamp": health_status.timestamp.isoformat(),
        "components": health_status.components,
        "system_metrics": health_status.metrics,
        "message": health_status.message
    }

# Document ingestion endpoints
@app.post("/documents/ingest")
async def ingest_document(doc_request: DocumentRequest) -> Dict[str, Any]:
    """
    Ingest a document for annotation.
    
    Creates document object with sentence segmentation.
    """
    if not ingestion_service:
        raise HTTPException(status_code=503, detail="Ingestion service not available")
    
    try:
        # Create temporary file for ingestion
        temp_path = pipeline_root / "data/raw" / f"{doc_request.doc_id}.txt"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_text(doc_request.text)
        
        # Ingest document
        document = ingestion_service.ingest_text_file(
            temp_path,
            source=doc_request.source,
            title=doc_request.title,
            metadata=doc_request.metadata
        )
        
        # Keep the file for listing in /documents endpoint
        # Don't delete the temp file so it appears in document list
        
        return {
            "doc_id": document.doc_id,
            "sentence_count": len(document.sentences),
            "message": "Document ingested successfully"
        }
        
    except Exception as e:
        logger.error(f"Document ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{doc_id}")
async def get_document(doc_id: str) -> Dict[str, Any]:
    """Get document by ID"""
    # This would typically query a database
    # For now, return a placeholder
    return {"doc_id": doc_id, "status": "not_implemented"}

# Candidate generation endpoints
@app.post("/candidates/generate")
async def generate_candidates(
    sentence_request: SentenceRequest,
    current_user: Optional[Any] = Depends(lambda: None)  # Optional auth for now
) -> CandidateResponse:
    """
    Generate annotation candidates for a sentence.
    
    Uses both LLM and rule-based approaches.
    """
    if not llm_generator or not rule_engine or not triplet_workflow:
        raise HTTPException(status_code=503, detail="Candidate generation services not available")
    
    # Log the API request
    from utils.logging_config import log_api_request, log_llm_request
    
    async with LogOperation("candidate_generation", "api.candidates", {
        "doc_id": sentence_request.doc_id,
        "sent_id": sentence_request.sent_id,
        "user": getattr(current_user, 'username', 'anonymous') if current_user else 'anonymous'
    }):
        try:
            from utils.monitoring import monitor
            
            # Monitor the operation
            async with monitor.monitor_operation("candidate_generation"):
                start_time = datetime.now()
                
                # Track request
                monitor.metrics.increment_counter("candidate_requests_total")
                
                # Generate combined LLM triplet workflow output
                llm_start = datetime.now()
                workflow_result = await triplet_workflow.process_sentence(
                    sentence_request.doc_id,
                    sentence_request.sent_id,
                    sentence_request.text,
                    sentence_request.title
                )
                topics = await llm_generator.suggest_topics(
                    sentence_request.text,
                    sentence_request.title
                )
                llm_duration = (datetime.now() - llm_start).total_seconds()

                log_llm_request(
                    provider="openai",
                    model="gpt-4o",
                    duration=llm_duration,
                    tokens_used=None
                )

            # Use rule results from workflow (fallback to direct call if missing)
            rule_result = workflow_result.rule_result or rule_engine.process_sentence(
                sentence_request.doc_id,
                sentence_request.sent_id,
                sentence_request.text
            )

            entities_payload = workflow_result.entities
            topics_payload = [asdict(topic) for topic in topics]

            triplets_payload = [item.to_dict() for item in workflow_result.triplets]
            relations_payload: List[Dict[str, Any]] = []
            for item in workflow_result.triplets:
                audit_data = item.audit.to_dict() if item.audit else {}
                relations_payload.append({
                    "triplet_id": item.triplet_id,
                    "head_cid": item.head.get("cid"),
                    "tail_cid": item.tail.get("cid"),
                    "head_text": item.head.get("text"),
                    "tail_text": item.tail.get("text"),
                    "label": item.relation,
                    "confidence": item.confidence,
                    "evidence": item.evidence,
                    "audit": audit_data,
                    "rule_support": item.rule_support,
                    "rule_sources": item.rule_sources,
                })

            candidates_payload = {
                "entities": entities_payload,
                "relations": relations_payload,
                "topics": topics_payload,
                "triplets": triplets_payload,
                "metadata": {
                    "audit_overall_verdict": workflow_result.audit_overall_verdict,
                    "audit_notes": workflow_result.audit_notes,
                }
            }
            
            # Calculate triage score if triage engine available
            triage_score = None
            if triage_engine:
                doc_metadata = {
                    "doc_id": sentence_request.doc_id,
                    "sent_id": sentence_request.sent_id,
                    "source": "manual"
                }
                
                # Add candidates to triage (this will calculate scores)
                triage_engine.add_candidates(
                    candidates_payload,
                    doc_metadata,
                    rule_result
                )
                
                # Get the latest triage score
                if triage_engine.queue:
                    triage_score = triage_engine.queue[-1].priority_score
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return CandidateResponse(
                doc_id=sentence_request.doc_id,
                sent_id=sentence_request.sent_id,
                candidates=candidates_payload,
                rule_results=rule_result,
                triage_score=triage_score,
                processing_time=processing_time
            )
        
        except Exception as e:
            from utils.error_handling import error_handler
            error_response = error_handler.handle_api_error(e, {
                "endpoint": "/candidates/generate",
                "doc_id": sentence_request.doc_id,
                "sent_id": sentence_request.sent_id
            })
            raise HTTPException(
                status_code=error_response["status_code"],
                detail=error_response["response"]
            )

@app.post("/candidates/batch")
async def generate_batch_candidates(batch_request: BatchSentenceRequest) -> List[CandidateResponse]:
    """Generate candidates for multiple sentences"""
    if not llm_generator:
        raise HTTPException(status_code=503, detail="LLM generator not available")
    
    try:
        # Convert to LLM generator format
        sentences = [
            {
                "doc_id": s.doc_id,
                "sent_id": s.sent_id,
                "text": s.text,
                "title": s.title
            }
            for s in batch_request.sentences
        ]
        
        # Process batch
        results = await llm_generator.process_batch(sentences, batch_request.batch_size)
        
        # Convert to response format
        responses = []
        for result in results:
            responses.append(CandidateResponse(
                doc_id=result["doc_id"],
                sent_id=result["sent_id"],
                candidates=result["candidates"],
                processing_time=result["processing_time"]
            ))
        
        return responses
        
    except Exception as e:
        from utils.error_handling import error_handler
        error_response = error_handler.handle_api_error(e, {
            "endpoint": "/candidates/batch",
            "batch_size": batch_request.batch_size,
            "sentence_count": len(batch_request.sentences)
        })
        raise HTTPException(
            status_code=error_response["status_code"],
            detail=error_response["response"]
        )

# Triage and queue management endpoints
@app.get("/triage/queue")
async def get_triage_queue(
    limit: int = Query(10, ge=1, le=100),
    priority_filter: Optional[str] = None,
    annotator: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Get items from triage queue"""
    if not triage_engine:
        raise HTTPException(status_code=503, detail="Triage engine not available")
    
    try:
        # Use peek_queue for display purposes to avoid disrupting the queue
        if annotator:
            # If annotator specified, use get_next_batch for actual assignment
            batch = triage_engine.get_next_batch(limit, annotator)
        else:
            # For display purposes, use peek_queue to avoid queue disruption
            batch = triage_engine.peek_queue(limit)
        
        # Enrich items with sentence text and document titles
        enriched_items = []
        for item in batch:
            enriched_item = {
                "item_id": item.item_id,
                "doc_id": item.doc_id,
                "sent_id": item.sent_id,
                "item_type": item.item_type,
                "priority_score": item.priority_score,
                "priority_level": item.priority_level.name,
                "candidate_data": item.candidate_data,
                "assigned_to": item.assigned_to,
                "sentence_text": "No text available...",  # Default fallback
                "document_title": "Unknown..."  # Default fallback
            }
            
            # Try to get full sentence text and document title
            try:
                # Look for document files to get title and sentence text
                raw_dir = pipeline_root / "data/raw"
                for doc_file in raw_dir.glob("*.txt"):
                    if ingestion_service:
                        # Process the document to get sentences
                        doc = ingestion_service.ingest_text_file(
                            doc_file,
                            source="uploaded",
                            title=doc_file.stem
                        )
                        
                        if doc.doc_id == item.doc_id:
                            # Found matching document
                            enriched_item["document_title"] = doc.title or doc_file.stem
                            
                            # Find matching sentence
                            for sentence in doc.sentences:
                                if sentence.sent_id == item.sent_id:
                                    enriched_item["sentence_text"] = sentence.text
                                    break
                            break
            except Exception as e:
                logger.warning(f"Could not enrich item {item.item_id}: {e}")
            
            enriched_items.append(enriched_item)
        
        return enriched_items
        
    except Exception as e:
        logger.error(f"Triage queue access failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/triage/statistics")
async def get_triage_statistics() -> Dict[str, Any]:
    """Get triage queue statistics"""
    if not triage_engine:
        raise HTTPException(status_code=503, detail="Triage engine not available")
    
    try:
        return triage_engine.get_queue_statistics()
    except Exception as e:
        logger.error(f"Triage statistics failed: {e}")
        # Return default statistics instead of failing
        return {
            "total_items": 0,
            "by_priority": {},
            "by_type": {},
            "completed_items": 0,
            "pending_items": 0,
            "error": "Failed to load statistics"
        }

# Helper function for queue repopulation
async def repopulate_queue_from_documents():
    """Repopulate triage queue from available documents"""
    if not ingestion_service or not rule_engine or not triage_engine:
        raise Exception("Required services not available")
    
    logger.info("🔄 Starting queue repopulation from documents...")
    
    # Find all raw documents
    raw_dir = pipeline_root / "data/raw"
    if not raw_dir.exists():
        logger.warning("No data/raw directory found")
        return
    
    doc_files = list(raw_dir.glob("*.txt"))
    if not doc_files:
        logger.warning("No documents found to repopulate queue")
        return
    
    total_candidates = 0
    processed_docs = 0
    
    # Process each document
    for doc_file in doc_files:
        logger.info(f"🔄 Processing {doc_file.name} for queue repopulation")
        
        try:
            # Ingest document to get sentences
            document = ingestion_service.ingest_text_file(
                doc_file,
                source="uploaded", 
                title=doc_file.stem
            )
            
            # Process all sentences
            for sentence in document.sentences:
                try:
                    # Generate rule-based candidates
                    rule_result = rule_engine.process_sentence(
                        document.doc_id,
                        sentence.sent_id,
                        sentence.text
                    )
                    
                    # Create candidate data
                    candidates = {
                        "text": sentence.text,
                        "entities": rule_result.get("entities", [])
                    }
                    
                    doc_metadata = {
                        "doc_id": document.doc_id,
                        "sent_id": sentence.sent_id,
                        "title": document.title,
                        "source": "uploaded"
                    }
                    
                    # Add to triage queue
                    triage_engine.add_candidates(
                        candidates,
                        doc_metadata,
                        {
                            "entities": rule_result.get("entities", []),
                            "relations": rule_result.get("relations", []),
                            "topics": rule_result.get("topics", [])
                        }
                    )
                    total_candidates += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to process sentence in {doc_file.name}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Failed to process document {doc_file.name}: {e}")
            continue
    
    logger.info(f"🔄 Queue repopulation complete: {total_candidates} candidates from {processed_docs} documents")

# Simpler approach - let's just avoid duplicates by checking existing queue items
async def repopulate_queue_simple():
    """Simple queue repopulation that avoids duplicates"""
    try:
        # Call the existing populate endpoint logic
        result = await populate_triage_queue()
        logger.info("🔄 Queue repopulated using existing populate endpoint")
        return result
    except Exception as e:
        logger.warning(f"Failed to repopulate queue: {e}")
        return None

# Annotation decision endpoints
@app.post("/annotations/decisions")
async def submit_annotation_decision(decision: AnnotationDecision) -> Dict[str, str]:
    """Submit an annotation decision (structured format)"""
    if not triage_engine:
        raise HTTPException(status_code=503, detail="Triage engine not available")
    
    try:
        # Mark item as completed in triage
        removed = triage_engine.mark_completed(decision.item_id, decision.decision)
        if not removed:
            logger.warning(
                "Triage item %s could not be removed for decision %s",
                decision.item_id,
                decision.decision,
            )
        
        # Store gold annotation if accepted/modified
        if decision.decision in ["accepted", "modified"] and decision.final_annotation:
            gold_path = pipeline_root / "data/gold" / f"{decision.item_id}.json"
            gold_path.parent.mkdir(parents=True, exist_ok=True)
            
            gold_data = {
                **decision.final_annotation,
                "annotator": decision.annotator,
                "decision": decision.decision,
                "timestamp": datetime.now().isoformat(),
                "notes": decision.notes
            }
            
            with open(gold_path, 'w') as f:
                json.dump(gold_data, f, indent=2)
        
        return {"status": "success", "message": "Decision recorded"}
        
    except Exception as e:
        logger.error(f"Decision submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/annotations/decide")  # Frontend compatibility endpoint
async def submit_annotation_decision_frontend(annotation_data: Dict[str, Any]) -> Dict[str, str]:
    """Submit an annotation decision (frontend format)"""
    if not triage_engine:
        raise HTTPException(status_code=503, detail="Triage engine not available")
    
    try:
        # Extract fields from frontend format
        item_id = annotation_data.get("item_id") or annotation_data.get("candidate_id")
        decision = annotation_data.get("decision", "accept")
        
        # DEBUG: Log what we're trying to remove vs what's in queue
        logger.info(f"🔍 ANNOTATION SUBMISSION DEBUG:")
        logger.info(f"   Submitted item_id: {item_id}")
        logger.info(f"   Decision: {decision}")
        logger.info(f"   Queue has {len(triage_engine.queue)} items:")
        for i, queue_item in enumerate(triage_engine.queue[:5]):  # Show first 5
            logger.info(f"     Queue item {i}: {queue_item.item_id}")
        if len(triage_engine.queue) > 5:
            logger.info(f"     ... and {len(triage_engine.queue) - 5} more items")
        
        # Normalize decision values
        if decision == "accept":
            decision = "accepted"
        elif decision == "reject":
            decision = "rejected"
        elif decision == "modify":
            decision = "modified"
        elif decision == "skip":
            decision = "skipped"
        
        # Mark item as completed in triage
        removal_ids = [item_id]
        candidate_identifier = annotation_data.get("candidate_id")
        if candidate_identifier is not None and candidate_identifier not in removal_ids:
            removal_ids.append(candidate_identifier)
        # Legacy payloads sometimes send camelCase identifiers
        legacy_identifier = annotation_data.get("itemId")
        if legacy_identifier is not None and legacy_identifier not in removal_ids:
            removal_ids.append(legacy_identifier)

        removed = False
        for candidate in removal_ids:
            if candidate is None:
                continue
            if triage_engine.mark_completed(candidate, decision):
                removed = True
                if candidate != item_id:
                    logger.info(
                        "✅ Queue removal recovered using fallback ID %s (submitted %s)",
                        candidate,
                        item_id,
                    )
                break

        if not removed:
            logger.warning(
                "Item %s not found in triage queue for decision %s (fallbacks tried: %s)",
                item_id,
                decision,
                [str(x) for x in removal_ids],
            )
        
        # Store gold annotation if accepted/modified
        if decision in ["accepted", "modified"]:
            gold_path = pipeline_root / "data/gold" / f"{item_id}.json"
            gold_path.parent.mkdir(parents=True, exist_ok=True)
            
            gold_data = {
                "item_id": item_id,
                "decision": decision,
                "entities": annotation_data.get("entities", []),
                "relations": annotation_data.get("relations", []),
                "topics": annotation_data.get("topics", []),
                "triplets": annotation_data.get("triplets", []),
                "confidence": annotation_data.get("confidence", 0.9),
                "notes": annotation_data.get("notes", ""),
                "annotator": f"user_{annotation_data.get('user_id', 'unknown')}",
                "timestamp": datetime.now().isoformat()
            }
            
            with open(gold_path, 'w') as f:
                json.dump(gold_data, f, indent=2)
        
        # Get next item from queue
        next_item = None
        try:
            # Debug queue state before getting next batch
            queue_stats = triage_engine.get_queue_statistics()
            logger.info(f"🔍 Queue state after marking {item_id} as completed:")
            logger.info(f"   Total items in queue: {queue_stats.get('total_items', 0)}")
            logger.info(f"   Completed items: {queue_stats.get('completed_items', 0)}")
            logger.info(f"   Queue length: {len(triage_engine.queue)}")
            
            batch = triage_engine.get_next_batch(1)
            logger.info(f"🔍 get_next_batch(1) returned: {len(batch) if batch else 0} items")
            
            # If queue is empty, try to repopulate from documents
            if not batch and len(triage_engine.queue) == 0:
                logger.info("🔄 Queue is empty, attempting to repopulate from documents...")
                try:
                    await repopulate_queue_simple()
                    batch = triage_engine.get_next_batch(1)
                    logger.info(f"🔍 After repopulation, get_next_batch(1) returned: {len(batch) if batch else 0} items")
                except Exception as e:
                    logger.warning(f"Failed to repopulate queue: {e}")
            
            if batch:
                next_item_data = batch[0]
                # Convert TriageItem to dict format expected by frontend
                next_item = {
                    "item_id": next_item_data.item_id,
                    "id": next_item_data.item_id,
                    "doc_id": next_item_data.doc_id,
                    "sent_id": next_item_data.sent_id,
                    "item_type": next_item_data.item_type,
                    "priority_score": next_item_data.priority_score,
                    "candidate_data": next_item_data.candidate_data
                }
                logger.info(f"✅ Next item for annotation: {next_item['item_id']}")
            else:
                logger.warning(f"❌ No items returned from get_next_batch despite queue having {len(triage_engine.queue)} items")
                # Debug: log first few items in queue
                if triage_engine.queue:
                    for i, item in enumerate(triage_engine.queue[:3]):
                        logger.info(f"   Queue item {i}: {item.item_id} (status: {item.status})")
        except Exception as e:
            logger.error(f"❌ Failed to get next item: {e}", exc_info=True)
        
        logger.info(f"Annotation decision recorded: {decision} for item {item_id}")
        return {
            "status": "success", 
            "message": "Decision recorded", 
            "next_item": next_item
        }
        
    except Exception as e:
        logger.error(f"Decision submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Export endpoints
@app.post("/export/gold")
async def export_gold_data(export_request: ExportRequest) -> Dict[str, Any]:
    """Export gold annotations in specified format"""
    try:
        gold_dir = pipeline_root / "data/gold"
        if not gold_dir.exists():
            raise HTTPException(status_code=404, detail="No gold annotations found")
        
        # Collect gold files
        gold_files = list(gold_dir.glob("*.json"))
        if not gold_files:
            raise HTTPException(status_code=404, detail="No gold annotations found")
        
        # Filter by doc_ids if specified
        if export_request.doc_ids:
            gold_files = [
                f for f in gold_files 
                if any(doc_id in f.name for doc_id in export_request.doc_ids)
            ]
        
        # Load and format data
        exported_data = []
        for gold_file in gold_files:
            with open(gold_file, 'r') as f:
                gold_annotation = json.load(f)
                
                if export_request.format == "jsonl":
                    exported_data.append(gold_annotation)
                elif export_request.format == "conll":
                    # Convert to CoNLL format
                    # This would need implementation based on specific requirements
                    pass
                elif export_request.format == "scibert":
                    # Convert to SciBERT training format
                    # This would integrate with existing training data generation
                    pass
        
        # Save export
        export_path = pipeline_root / "data/exports" / f"gold_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_request.format}"
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        if export_request.format == "jsonl":
            with open(export_path, 'w') as f:
                for item in exported_data:
                    f.write(json.dumps(item) + "\n")
        else:
            with open(export_path, 'w') as f:
                json.dump(exported_data, f, indent=2)
        
        return {
            "export_path": str(export_path),
            "item_count": len(exported_data),
            "format": export_request.format
        }
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Integration endpoints for ML pipeline
@app.get("/integration/training-data")
async def get_training_data(
    format: str = "scibert",
    split: Optional[str] = None,
    min_annotations: int = 1
) -> Dict[str, Any]:
    """
    Get training data for ML pipeline.
    
    This endpoint provides data in the format expected by the 
    existing SciBERT training pipeline.
    """
    try:
        gold_dir = pipeline_root / "data/gold"
        
        # Collect and format training data
        training_data = []
        
        for gold_file in gold_dir.glob("*.json"):
            with open(gold_file, 'r') as f:
                annotation = json.load(f)
                
                # Convert to training format
                if format == "scibert":
                    training_item = {
                        "text": annotation.get("text", ""),
                        "entities": annotation.get("entities", []),
                        "relations": annotation.get("relations", []),
                        "doc_id": annotation.get("doc_id"),
                        "annotator": annotation.get("annotator"),
                        "timestamp": annotation.get("timestamp")
                    }
                    training_data.append(training_item)
        
        return {
            "training_data": training_data,
            "count": len(training_data),
            "format": format,
            "generation_time": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Training data generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/integration/model-feedback")
async def submit_model_feedback(feedback_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Receive feedback from ML training pipeline.
    
    This can be used to update candidate generation or triage priorities.
    """
    try:
        # Store feedback for analysis
        feedback_path = pipeline_root / "data/feedback" / f"model_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        feedback_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(feedback_path, 'w') as f:
            json.dump(feedback_data, f, indent=2)
        
        return {"status": "success", "message": "Feedback recorded"}
        
    except Exception as e:
        logger.error(f"Model feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Document management endpoints
@app.get("/documents")
async def get_documents() -> Dict[str, Any]:
    """Get list of processed documents"""
    try:
        documents = []
        
        # Check for processed documents in data directory
        data_dir = pipeline_root / "data"
        
        # Look for raw documents
        raw_dir = data_dir / "raw"
        if raw_dir.exists():
            for doc_file in raw_dir.glob("*.txt"):
                # Get sentence count by actually processing the document
                sentence_count = 0
                created_at = None
                try:
                    if ingestion_service:
                        # Process the document to get sentence count
                        doc = ingestion_service.ingest_text_file(
                            doc_file,
                            source="raw",
                            title=doc_file.stem
                        )
                        sentence_count = len(doc.sentences)
                        created_at = doc.metadata.get('ingestion_time')
                    else:
                        # Fallback: simple sentence count estimation
                        with open(doc_file, 'r', encoding='utf-8') as f:
                            text = f.read()
                            import re
                            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
                            sentence_count = len([s for s in sentences if s.strip()])
                except Exception as e:
                    logger.warning(f"Could not get sentence count for {doc_file}: {e}")
                    sentence_count = 0
                
                documents.append({
                    "doc_id": doc_file.stem,
                    "title": doc_file.stem,
                    "source": "raw",
                    "status": "processed",
                    "sentence_count": sentence_count,
                    "created_at": created_at or datetime.fromtimestamp(doc_file.stat().st_mtime).isoformat(),
                    "file_path": str(doc_file)
                })
        
        # Look for gold annotations (these indicate completed documents)
        gold_dir = data_dir / "gold"
        if gold_dir.exists():
            for gold_file in gold_dir.glob("*.json"):
                doc_id = gold_file.stem
                if not any(d["doc_id"] == doc_id for d in documents):
                    documents.append({
                        "doc_id": doc_id,
                        "title": doc_id,
                        "source": "annotated",
                        "status": "completed",
                        "file_path": str(gold_file)
                    })
        
        return {
            "documents": documents,
            "count": len(documents),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Document listing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str) -> Dict[str, Any]:
    """Delete a document and associated files"""
    try:
        data_dir = pipeline_root / "data"
        deleted_files = []
        
        # Delete raw document file
        raw_dir = data_dir / "raw"
        raw_file = raw_dir / f"{doc_id}.txt"
        if raw_file.exists():
            raw_file.unlink()
            deleted_files.append(str(raw_file))
            logger.info(f"Deleted raw document: {raw_file}")
        
        # Delete gold annotation file
        gold_dir = data_dir / "gold" 
        gold_file = gold_dir / f"{doc_id}.json"
        if gold_file.exists():
            gold_file.unlink()
            deleted_files.append(str(gold_file))
            logger.info(f"Deleted gold annotations: {gold_file}")
        
        # Delete candidates file
        candidates_dir = data_dir / "candidates"
        candidates_file = candidates_dir / f"{doc_id}.json"
        if candidates_file.exists():
            candidates_file.unlink()
            deleted_files.append(str(candidates_file))
            logger.info(f"Deleted candidates: {candidates_file}")
        
        if not deleted_files:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        
        return {
            "doc_id": doc_id,
            "deleted_files": deleted_files,
            "status": "deleted",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Queue population endpoint
@app.post("/triage/populate")
async def populate_triage_queue() -> Dict[str, Any]:
    """Populate triage queue from uploaded documents"""
    if not ingestion_service or not rule_engine or not triage_engine:
        raise HTTPException(status_code=503, detail="Required services not available")
    
    try:
        logger.info("Starting triage queue population...")
        
        # Find all raw documents
        raw_dir = pipeline_root / "data/raw"
        if not raw_dir.exists():
            raise HTTPException(status_code=404, detail="No documents found")
        
        doc_files = list(raw_dir.glob("*.txt"))
        if not doc_files:
            raise HTTPException(status_code=404, detail="No documents to process")
        
        total_candidates = 0
        processed_docs = 0
        
        # Process each document
        for doc_file in doc_files:
            logger.info(f"Processing {doc_file.name}")
            
            # Ingest document to get sentences
            document = ingestion_service.ingest_text_file(
                doc_file,
                source="uploaded", 
                title=doc_file.stem
            )
            
            # Process all sentences
            for sentence in document.sentences:
                try:
                    # Generate rule-based candidates
                    rule_result = rule_engine.process_sentence(
                        document.doc_id,
                        sentence.sent_id,
                        sentence.text
                    )
                    
                    # Create candidate data
                    candidates = {
                        "text": sentence.text,
                        "entities": rule_result.get("entities", [])
                    }
                    
                    doc_metadata = {
                        "doc_id": document.doc_id,
                        "sent_id": sentence.sent_id,
                        "title": document.title,
                        "source": "uploaded"
                    }
                    
                    # Add to triage queue
                    triage_engine.add_candidates(
                        candidates,
                        doc_metadata,
                        {
                            "entities": rule_result.get("entities", []),
                            "relations": rule_result.get("relations", []),
                            "topics": rule_result.get("topics", [])
                        }
                    )
                    total_candidates += 1
                    
                except Exception as e:
                    logger.error(f"Failed to process sentence: {e}")
                    continue
            
            processed_docs += 1
        
        queue_stats = triage_engine.get_queue_statistics()
        
        return {
            "message": "Triage queue populated successfully",
            "processed_documents": processed_docs,
            "total_candidates": total_candidates,
            "queue_statistics": dict(queue_stats)
        }
        
    except Exception as e:
        logger.error(f"Queue population failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Statistics and monitoring
@app.get("/statistics/overview")
async def get_system_statistics() -> Dict[str, Any]:
    """Get overall system statistics"""
    stats = {
        "timestamp": datetime.now().isoformat(),
        "services": {}
    }
    
    try:
        if triage_engine:
            stats["services"]["triage"] = triage_engine.get_queue_statistics()
        else:
            stats["services"]["triage"] = {"error": "Triage engine not available"}
    except Exception as e:
        logger.error(f"Triage statistics failed: {e}")
        stats["services"]["triage"] = {"error": "Failed to load triage statistics"}
    
    try:
        if rule_engine:
            stats["services"]["rules"] = rule_engine.get_statistics()
        else:
            stats["services"]["rules"] = {"error": "Rule engine not available"}
    except Exception as e:
        logger.error(f"Rule statistics failed: {e}")
        stats["services"]["rules"] = {"error": "Failed to load rule statistics"}
    
    # Count gold annotations
    try:
        gold_dir = pipeline_root / "data/gold"
        if gold_dir.exists():
            stats["gold_annotations"] = len(list(gold_dir.glob("*.json")))
        else:
            stats["gold_annotations"] = 0
    except Exception as e:
        logger.error(f"Gold annotations count failed: {e}")
        stats["gold_annotations"] = 0
    
    return stats

if ui_build_path.exists():
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_frontend_routes(full_path: str):
        """Serve React frontend for unmatched non-API routes"""
        # Serve index.html for frontend routes
        index_file = ui_build_path / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        raise HTTPException(status_code=404, detail="Frontend not available")

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "annotation_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
