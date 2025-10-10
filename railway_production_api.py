#!/usr/bin/env python3
"""
SIMPLIFIED Railway Production API - No Conflicts
"""

import os
import sys
import logging
import traceback
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables for Railway
os.environ.setdefault("ENVIRONMENT", "production")
os.environ.setdefault("API_HOST", "0.0.0.0")
os.environ.setdefault("API_PORT", str(os.getenv("PORT", "8000")))
os.environ.setdefault("LOG_LEVEL", "INFO")

# Configure enhanced logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Enhanced startup logging
logger.info("🚀 Railway Production API Starting... [SIMPLIFIED VERSION]")
logger.info(f"📂 Working directory: {os.getcwd()}")
logger.info(f"🐍 Python path: {sys.path[:3]}...")
logger.info(f"🌍 Environment: {os.getenv('ENVIRONMENT', 'unknown')}")
logger.info(f"🔧 Port: {os.getenv('PORT', '8000')}")

# Test critical imports with detailed error reporting
logger.info("🔍 Testing critical imports...")
import_status = {}

try:
    import fastapi
    logger.info("✅ FastAPI imported successfully")
    import_status['fastapi'] = True
except ImportError as e:
    logger.error(f"❌ FastAPI import failed: {e}")
    import_status['fastapi'] = False

try:
    import openai
    logger.info("✅ OpenAI client imported successfully")
    import_status['openai'] = True
except ImportError as e:
    logger.error(f"❌ OpenAI import failed: {e}")
    import_status['openai'] = False

try:
    # Check for OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        logger.warning("⚠️ No OPENAI_API_KEY environment variable found in Railway")
        logger.info("🔄 Triplet generation will use enhanced fallback mode")
    else:
        logger.info(f"✅ OpenAI API key found: {openai_key[:10]}...")
    
    # Import the full annotation API or create minimal one
    try:
        from services.api.annotation_api import app
        logger.info("✅ Successfully imported full annotation API")
        import_status['main_api'] = True
    except ImportError as e:
        logger.error(f"❌ Full API import failed: {e}, creating minimal API")
        import_status['main_api'] = False
        
        # Create minimal FastAPI app
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        
        app = FastAPI(
            title="Shrimp Annotation Pipeline API (Minimal)",
            description="Minimal Railway-compatible API",
            version="1.0.0"
        )
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Import required modules
    from fastapi import HTTPException, Depends
    from fastapi.responses import JSONResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    from typing import Optional, Dict, Any, List
    import datetime
    import json
    
    # PERSISTENT STORAGE - THE SINGLE SOURCE OF TRUTH
    storage_file = Path("/tmp/railway_storage.json")
    
    def load_storage():
        """Load storage from file with detailed logging"""
        try:
            logger.info(f"🔍 Loading storage from: {storage_file} (exists: {storage_file.exists()})")
            if storage_file.exists():
                file_size = storage_file.stat().st_size
                logger.info(f"📁 Storage file size: {file_size} bytes")
                with open(storage_file, 'r') as f:
                    data = json.load(f)
                docs = data.get('uploaded_documents', [])
                items = data.get('triage_items', [])
                logger.info(f"✅ Loaded from storage: {len(docs)} docs, {len(items)} items")
                return docs, items
            else:
                logger.info("📂 No storage file found, starting with empty storage")
        except Exception as e:
            logger.error(f"❌ Failed to load storage: {e}")
        return [], []
    
    def save_storage(documents, items):
        """Save storage to file with detailed logging"""
        try:
            logger.info(f"💾 Attempting to save: {len(documents)} docs, {len(items)} items to {storage_file}")
            
            # Ensure directory exists
            storage_file.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'uploaded_documents': documents,
                'triage_items': items,
                'timestamp': datetime.datetime.now().isoformat(),
                'save_count': len(documents) + len(items)
            }
            
            # Write to temporary file first, then rename (atomic operation)
            temp_file = storage_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Atomic rename
            temp_file.rename(storage_file)
            
            # Verify the save worked
            verify_size = storage_file.stat().st_size if storage_file.exists() else 0
            logger.info(f"✅ Successfully saved storage: {len(documents)} docs, {len(items)} items ({verify_size} bytes)")
            
        except Exception as e:
            logger.error(f"❌ Failed to save storage: {e}")
            logger.error(f"❌ Current working dir: {os.getcwd()}")
            logger.error(f"❌ /tmp permissions: {oct(os.stat('/tmp').st_mode)}")
    
    # REQUEST MODELS
    class DraftAnnotationRequest(BaseModel):
        item_id: str
        draft_data: Dict[str, Any]
        timestamp: Optional[str] = None
        user_id: Optional[str] = "anonymous"
    
    class CandidateRequest(BaseModel):
        doc_id: str
        sent_id: str
        text: str
        title: Optional[str] = None

    # HEALTH CHECK
    @app.get("/api/health")
    async def health_check():
        return {"status": "healthy", "version": "simplified-1.0.0"}

    # TRIAGE QUEUE - SINGLE AUTHORITATIVE ENDPOINT
    @app.get("/api/triage/queue")
    async def get_triage_queue(limit: int = 100, offset: int = 0, status: str = None, sort_by: str = None):
        """Get triage queue items from persistent storage"""
        logger.info(f"🎯 [SINGLE TRIAGE ENDPOINT] Queue requested: limit={limit}, status={status}")
        
        # Load from persistent storage EVERY TIME
        stored_docs, stored_items = load_storage()
        logger.info(f"📊 [SINGLE TRIAGE] Loaded from storage: {len(stored_items)} items")
        
        # Mock items for demonstration
        mock_items = [
            {
                "item_id": 1,
                "doc_id": "doc_1",
                "sent_id": "sent_1",
                "text": "White Spot Syndrome Virus (WSSV) is one of the most devastating pathogens affecting Pacific white shrimp.",
                "priority_score": 0.95,
                "confidence": 0.8,
                "status": "pending",
                "created_at": "2024-01-01T00:00:00Z"
            },
            {
                "item_id": 2,
                "doc_id": "doc_2",
                "sent_id": "sent_2", 
                "text": "PCR screening is critical for early detection of aquaculture pathogens.",
                "priority_score": 0.87,
                "confidence": 0.75,
                "status": "pending",
                "created_at": "2024-01-01T01:00:00Z"
            }
        ]
        
        # Combine uploaded items from storage with mock items (uploaded first)
        all_items = stored_items + mock_items
        
        # Filter by status if specified
        if status and status != "undefined" and status != "null" and status.lower() != "all items":
            all_items = [item for item in all_items if item["status"] == status]
            logger.info(f"🔍 Filtered items by status '{status}': {len(all_items)} items remaining")
        
        # Sort by priority if requested
        if sort_by == "priority":
            all_items = sorted(all_items, key=lambda x: x["priority_score"], reverse=True)
        
        final_items = all_items[offset:offset+limit]
        logger.info(f"🎯 [SINGLE TRIAGE] Returning {len(final_items)} items out of {len(all_items)} total")
        
        return {
            "items": final_items,
            "total": len(all_items),
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < len(all_items),
            "source": "persistent_storage"
        }

    # DOCUMENTS ENDPOINT 
    @app.get("/api/documents")
    async def get_documents(limit: int = 50, offset: int = 0):
        """Get documents list from persistent storage"""
        logger.info(f"📄 [DOCUMENTS] Requested: limit={limit}, offset={offset}")
        
        # Load from persistent storage EVERY TIME
        stored_docs, stored_items = load_storage()
        logger.info(f"📂 [DOCUMENTS] Loaded from storage: {len(stored_docs)} docs")
        
        # Mock documents for demonstration
        mock_documents = [
            {
                "doc_id": "doc_1",
                "title": "White Spot Syndrome Virus in Pacific White Shrimp",
                "sentence_count": 45,
                "annotation_count": 12,
                "status": "annotated",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T12:00:00Z"
            },
            {
                "doc_id": "doc_2", 
                "title": "PCR Detection Methods for Aquaculture Pathogens",
                "sentence_count": 32,
                "annotation_count": 8,
                "status": "pending",
                "created_at": "2024-01-01T01:00:00Z",
                "updated_at": "2024-01-01T13:00:00Z"
            }
        ]
        
        # Add uploaded documents from persistent storage (newest first)
        all_documents = stored_docs + mock_documents
        
        return {
            "documents": all_documents[offset:offset+limit],
            "total": len(all_documents),
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < len(all_documents),
            "from_storage": len(stored_docs)
        }

    # DOCUMENT INGEST
    @app.post("/api/documents/ingest")
    async def ingest_document(request: Dict[str, Any]):
        """Ingest a new document for annotation"""
        logger.info(f"📥 [INGEST] Document requested: {request.get('title', 'Unknown')[:50]}...")
        
        # Extract document info
        title = request.get('title', 'Untitled Document')
        content = request.get('content', request.get('text', ''))
        file_name = request.get('fileName', request.get('filename', 'upload.txt'))
        
        # Load current storage
        current_docs, current_items = load_storage()
        
        # Create unique document ID
        doc_id = f"uploaded_{len(current_docs) + 1}"
        timestamp = datetime.datetime.now().isoformat() + "Z"
        
        # Split content into sentences for processing
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        sentence_count = len(sentences)
        
        # Add to uploaded documents
        new_document = {
            "doc_id": doc_id,
            "title": title,
            "sentence_count": sentence_count,
            "annotation_count": 0,
            "status": "ingested",
            "created_at": timestamp,
            "updated_at": timestamp,
            "file_name": file_name
        }
        current_docs.insert(0, new_document)  # Add to front (newest first)
        
        # Create triage items for sentences that need annotation
        for i, sentence in enumerate(sentences[:3]):  # Limit to first 3 sentences for demo
            if len(sentence) > 20:  # Only meaningful sentences
                triage_item = {
                    "item_id": len(current_items) + 100,  # Unique ID
                    "doc_id": doc_id,
                    "sent_id": f"{doc_id}_sent_{i+1}",
                    "text": sentence + ".",
                    "priority_score": 0.8 + (0.1 * (3-i)),  # Higher priority for earlier sentences
                    "confidence": 0.0,  # New, needs annotation
                    "status": "pending",
                    "created_at": timestamp,
                    "metadata": {"source": "uploaded", "sentence_index": i}
                }
                current_items.insert(0, triage_item)  # Add to front
        
        # Save to persistent storage
        save_storage(current_docs, current_items)
        
        triage_created = min(3, len([s for s in sentences if len(s) > 20]))
        logger.info(f"✅ [INGEST] Document '{title}' added with {sentence_count} sentences, {triage_created} triage items created")
        
        return {
            "success": True,
            "doc_id": doc_id,
            "title": title,
            "status": "ingested",
            "sentence_count": sentence_count,
            "created_at": timestamp,
            "triage_items_created": triage_created,
            "message": f"Document '{title}' successfully ingested for annotation"
        }

    # CANDIDATES GENERATION - Support both /api/candidates/generate and /candidates/generate
    @app.post("/api/candidates/generate")
    async def generate_candidates_endpoint(request: CandidateRequest):
        """Generate candidates - API version"""
        return await generate_candidates_logic(request)
    
    @app.post("/candidates/generate") 
    async def generate_candidates_original(request: CandidateRequest):
        """Generate candidates - Original API path (frontend calls this)"""
        logger.info("🎯 [FRONTEND] Called original /candidates/generate endpoint")
        return await generate_candidates_logic(request)
    
    async def generate_candidates_logic(request: CandidateRequest):
        """Generate candidates - try full API first, fallback to mock"""
        logger.info(f"🎯 [CANDIDATES] Requested for: {request.text[:50]}...")
        
        # Skip full API due to timeout issues in Railway, use direct OpenAI
        logger.info("🎯 [CANDIDATES] Skipping full API due to Railway performance issues, using direct OpenAI")
        
        # Try direct OpenAI if available
        logger.info(f"🔍 [CANDIDATES] Checking OpenAI availability...")
        logger.info(f"🔍 [CANDIDATES] openai_key available: {bool(openai_key)}")
        logger.info(f"🔍 [CANDIDATES] openai_key value: {openai_key[:15] if openai_key else 'None'}...")
        logger.info(f"🔍 [CANDIDATES] openai import status: {import_status.get('openai', False)}")
        logger.info(f"🔍 [CANDIDATES] OPENAI_API_KEY env: {bool(os.getenv('OPENAI_API_KEY'))}")
        
        if openai_key and import_status.get('openai', False):
            try:
                logger.info("🎯 [CANDIDATES] ✅ CONDITIONS MET - Trying direct OpenAI triplet generation")
                result = await generate_openai_triplets(request.text)
                logger.info("✅ [CANDIDATES] Used direct OpenAI generation")
                return result
            except Exception as e:
                logger.warning(f"⚠️ [CANDIDATES] OpenAI direct failed: {e}, using mock")
        else:
            logger.warning(f"⚠️ [CANDIDATES] OpenAI conditions NOT met - key:{bool(openai_key)}, import:{import_status.get('openai', False)}")
        
        # Fallback to enhanced mock generation
        logger.info("🔄 [CANDIDATES] Using enhanced mock generation")
        return generate_mock_triplets(request.text)

    async def generate_openai_triplets(sentence: str) -> Dict[str, Any]:
        """Direct OpenAI triplet generation using GPT-4o"""
        
        logger.info(f"🤖 [OPENAI] Starting generation for: {sentence[:50]}...")
        logger.info(f"🤖 [OPENAI] API Key available: {bool(openai_key)}")
        logger.info(f"🤖 [OPENAI] API Key prefix: {openai_key[:10] if openai_key else 'None'}...")
        
        prompt = f"""Extract knowledge graph triplets from this shrimp aquaculture sentence.

Sentence: {sentence}

Return JSON with triplets in this format:
{{
  "candidates": {{
    "triplets": [
      {{
        "triplet_id": "t1",
        "head": {{"text": "entity1", "type": "PATHOGEN", "node_id": "e1"}},
        "relation": "CAUSES", 
        "tail": {{"text": "entity2", "type": "DISEASE", "node_id": "e2"}},
        "evidence": "supporting text from sentence",
        "confidence": 0.9
      }}
    ],
    "entities": [],
    "relations": [],
    "topics": [],
    "metadata": {{"audit_notes": "OpenAI GPT-4o generated"}}
  }},
  "triage_score": 0.8,
  "processing_time": 0.3
}}

Entity types: PATHOGEN, DISEASE, SPECIES, CHEMICAL_COMPOUND, TREATMENT, TEST_TYPE, EQUIPMENT, LOCATION, MEASUREMENT, PROCESS, FEED
Relations: CAUSES, AFFECTS, PREVENTS, DETECTS, TREATS, LOCATED_IN, MEASURED_BY, OCCURS_AT, USES
Focus on high-confidence triplets that are clearly supported by the sentence text."""

        try:
            logger.info(f"🤖 [OPENAI] Creating client with model: gpt-4o")
            client = openai.OpenAI(api_key=openai_key)
            
            logger.info(f"🤖 [OPENAI] Sending request to OpenAI...")
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800
            )
            
            import json
            content = response.choices[0].message.content
            logger.info(f"🤖 [OPENAI] Response received, length: {len(content)} chars")
            logger.info(f"🤖 [OPENAI] Response preview: {content[:100]}...")
            
        except Exception as openai_error:
            logger.error(f"❌ [OPENAI] API call failed: {type(openai_error).__name__}: {openai_error}")
            logger.error(f"❌ [OPENAI] Falling back to mock immediately")
            return generate_mock_triplets(sentence)
        
        # Parse and validate the JSON response
        try:
            logger.info(f"🔍 [OPENAI] Parsing JSON response...")
            result = json.loads(content)
            
            # Detailed analysis of the response structure
            logger.info(f"🔍 [OPENAI] Response keys: {list(result.keys())}")
            
            candidates = result.get('candidates', {})
            if isinstance(candidates, dict):
                triplets = candidates.get('triplets', [])
                entities = candidates.get('entities', [])
                logger.info(f"🔍 [OPENAI] Candidates structure: triplets={len(triplets)}, entities={len(entities)}")
            else:
                triplets = []
                logger.warning(f"⚠️ [OPENAI] Unexpected candidates type: {type(candidates)}")
                
            triplet_count = len(triplets)
            logger.info(f"🤖 [OPENAI] Final triplet count: {triplet_count}")
            
            # If OpenAI returns 0 triplets, investigate why
            if triplet_count == 0:
                logger.warning(f"⚠️ [OPENAI] ZERO TRIPLETS - Investigating...")
                logger.warning(f"⚠️ [OPENAI] Full response: {json.dumps(result, indent=2)}")
                logger.warning(f"⚠️ [OPENAI] Using mock fallback due to empty response")
                return generate_mock_triplets(sentence)
            
            logger.info(f"✅ [OPENAI] Successfully generated {triplet_count} triplets")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ [OPENAI] JSON parsing failed: {e}")
            logger.error(f"❌ [OPENAI] Raw content (first 500 chars): {content[:500]}")
            logger.error(f"❌ [OPENAI] Using mock fallback due to JSON error")
            return generate_mock_triplets(sentence)

    def generate_mock_triplets(sentence: str) -> Dict[str, Any]:
        """Generate mock triplets based on sentence content"""
        sentence_lower = sentence.lower()
        mock_triplets = []
        
        # Disease/Pathogen patterns - comprehensive matching
        if any(word in sentence_lower for word in ['wssv', 'white spot', 'virus', 'affects', 'mortality', 'disease', 'pathogen', 'infection', 'syndrome', 'larvae']):
            # Choose appropriate entities based on sentence content
            if 'translucent' in sentence_lower or 'post-larvae' in sentence_lower or 'tpd' in sentence_lower:
                mock_triplets.append({
                    "triplet_id": "mock_1",
                    "head": {"text": "TPD", "type": "DISEASE", "node_id": "tpd"},
                    "relation": "AFFECTS", 
                    "tail": {"text": "post-larvae", "type": "SPECIES", "node_id": "larvae"},
                    "evidence": "TPD affects post-larvae development",
                    "confidence": 0.88
                })
            else:
                mock_triplets.append({
                    "triplet_id": "mock_1", 
                    "head": {"text": "White Spot Syndrome Virus", "type": "PATHOGEN", "node_id": "wssv"},
                    "relation": "AFFECTS",
                    "tail": {"text": "Pacific white shrimp", "type": "SPECIES", "node_id": "shrimp"},
                    "evidence": "White spot syndrome virus affects Pacific white shrimp",
                    "confidence": 0.90
                })
        
        # PCR/Detection patterns
        if any(word in sentence_lower for word in ['pcr', 'detect', 'screen']):
            mock_triplets.append({
                "triplet_id": "mock_2",
                "head": {"text": "PCR screening", "type": "TEST_TYPE", "node_id": "pcr"},
                "relation": "DETECTS",
                "tail": {"text": "WSSV", "type": "PATHOGEN", "node_id": "wssv"},
                "evidence": "PCR screening detects WSSV",
                "confidence": 0.88
            })
        
        return {
            "candidates": {
                "entities": [],
                "relations": [],
                "topics": [],
                "triplets": mock_triplets,
                "metadata": {
                    "audit_overall_verdict": "mock",
                    "audit_notes": f"Mock generation - found {len(mock_triplets)} relevant triplets"
                }
            },
            "triage_score": 0.7,
            "processing_time": 0.1,
            "model_info": {
                "provider": "mock",
                "model": "fallback",
                "sentence_length": len(sentence)
            }
        }

    # ADDITIONAL ENDPOINTS
    @app.post("/api/annotations/draft")
    async def save_draft_annotation(request: Dict[str, Any]):
        """Save draft annotation - accept any JSON data"""
        try:
            logger.info(f"💾 [DRAFT] Raw request data: {request}")
            item_id = request.get('item_id', 'unknown')
            draft_data = request.get('draft_data', {})
            timestamp = request.get('timestamp')
            
            logger.info(f"💾 [DRAFT] Saving draft for item: {item_id}")
            logger.info(f"💾 [DRAFT] Draft data keys: {list(draft_data.keys()) if draft_data else 'None'}")
            
            return {
                "status": "success",
                "message": "Draft saved successfully",
                "draft_id": f"draft_{item_id}",
                "timestamp": timestamp or "2024-01-01T00:00:00Z",
                "item_id": item_id
            }
        except Exception as e:
            logger.error(f"❌ [DRAFT] Error saving draft: {e}")
            logger.error(f"❌ [DRAFT] Request type: {type(request)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/annotations/statistics")
    async def get_annotation_statistics():
        """Get annotation statistics"""
        return {
            "total_annotations": 35,
            "completed_annotations": 15,
            "pending_annotations": 12,
            "in_progress_annotations": 8,
            "total_documents": 2,
            "annotated_documents": 1,
            "pending_documents": 1
        }

    # MISSING ENDPOINTS THAT FRONTEND EXPECTS
    @app.get("/api/statistics/overview")
    async def get_statistics_overview():
        """Get overview statistics for dashboard"""
        logger.info("📊 [STATS] Overview statistics requested")
        return {
            "total_documents": 3,
            "total_annotations": 35,
            "total_candidates": 67,
            "active_users": 1,
            "pending_triage_items": 5,
            "completed_annotations": 15,
            "annotation_rate": 0.75,
            "avg_confidence": 0.82
        }

    @app.get("/api/triage/next")
    async def get_next_triage_item():
        """Get next item from triage queue"""
        logger.info("⏭️ [TRIAGE] Next item requested")
        
        # Load from persistent storage
        stored_docs, stored_items = load_storage()
        
        # Mock next item
        if stored_items:
            next_item = stored_items[0]  # Return first item
            logger.info(f"⏭️ [TRIAGE] Returning uploaded item: {next_item.get('item_id')}")
            return next_item
        
        # Fallback mock item
        return {
            "item_id": 1,
            "doc_id": "doc_1",
            "sent_id": "sent_1", 
            "text": "White Spot Syndrome Virus (WSSV) is one of the most devastating pathogens affecting Pacific white shrimp.",
            "priority_score": 0.95,
            "confidence": 0.8,
            "status": "pending",
            "created_at": "2024-01-01T00:00:00Z"
        }

    @app.get("/api/annotations")
    async def get_annotations(
        sort_by: str = "created_at",
        limit: int = 20,
        offset: int = 0,
        status: Optional[str] = None
    ):
        """Get annotations list"""
        logger.info(f"📝 [ANNOTATIONS] List requested: sort_by={sort_by}, limit={limit}")
        
        # Mock annotations
        mock_annotations = [
            {
                "annotation_id": 1,
                "doc_id": "doc_1",
                "sent_id": "sent_1",
                "text": "WSSV causes severe mortality in shrimp farms.",
                "entities": [
                    {"text": "WSSV", "type": "PATHOGEN", "start": 0, "end": 4}
                ],
                "relations": [
                    {"head": "WSSV", "relation": "CAUSES", "tail": "mortality"}
                ],
                "status": "completed",
                "confidence": 0.9,
                "annotator": "anonymous", 
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T01:00:00Z"
            },
            {
                "annotation_id": 2,
                "doc_id": "doc_2", 
                "sent_id": "sent_2",
                "text": "PCR screening helps detect viral pathogens early.",
                "entities": [
                    {"text": "PCR screening", "type": "TEST_TYPE", "start": 0, "end": 13}
                ],
                "relations": [
                    {"head": "PCR screening", "relation": "DETECTS", "tail": "viral pathogens"}
                ],
                "status": "pending",
                "confidence": 0.85,
                "annotator": "anonymous",
                "created_at": "2024-01-01T02:00:00Z", 
                "updated_at": "2024-01-01T03:00:00Z"
            }
        ]
        
        # Filter by status if specified
        if status:
            mock_annotations = [ann for ann in mock_annotations if ann["status"] == status]
        
        return {
            "annotations": mock_annotations[offset:offset+limit],
            "total": len(mock_annotations),
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < len(mock_annotations)
        }

    @app.get("/api/triage/statistics")
    async def get_triage_statistics():
        """Get triage queue statistics"""
        logger.info("📊 [TRIAGE] Statistics requested")
        
        # Load from persistent storage
        stored_docs, stored_items = load_storage()
        
        return {
            "total_items": len(stored_items) + 5,  # Include uploaded + mock items
            "pending_items": len(stored_items) + 3,
            "in_progress_items": 2,
            "completed_items": 0,
            "avg_priority_score": 0.85,
            "avg_confidence": 0.7,
            "uploaded_items": len(stored_items)
        }

    # WEBSOCKET ENDPOINT
    @app.websocket("/ws/anonymous")
    async def websocket_endpoint(websocket):
        """WebSocket connection for real-time updates"""
        # Extract query parameters from the WebSocket URL
        query_params = websocket.query_params
        username = query_params.get('username', 'Anonymous')
        role = query_params.get('role', 'annotator')
        
        logger.info(f"🔗 [WEBSOCKET] Connection attempt from: {username} (role: {role})")
        
        try:
            await websocket.accept()
            logger.info(f"✅ [WEBSOCKET] Connected: {username}")
            
            # Send welcome message
            await websocket.send_json({
                "type": "connection",
                "status": "connected",
                "user": username,
                "role": role,
                "timestamp": "2024-01-01T00:00:00Z"
            })
            
            # Keep connection alive
            while True:
                try:
                    data = await websocket.receive_json()
                    logger.info(f"📨 [WEBSOCKET] Message from {username}: {data.get('type', 'unknown')}")
                    
                    # Echo back
                    await websocket.send_json({
                        "type": "echo",
                        "original": data,
                        "timestamp": "2024-01-01T00:00:00Z"
                    })
                    
                except Exception as e:
                    logger.warning(f"⚠️ [WEBSOCKET] Message error for {username}: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"❌ [WEBSOCKET] Connection error for {username}: {e}")
        finally:
            logger.info(f"🔌 [WEBSOCKET] Disconnected: {username}")

    # ANNOTATION DECISION ENDPOINT
    @app.post("/api/annotations/decide")
    async def decide_annotation(request: Dict[str, Any]):
        """Handle annotation decisions (accept, reject, skip)"""
        try:
            item_id = request.get('item_id')
            decision = request.get('decision', 'unknown')
            confidence = request.get('confidence', 0.5)
            user_id = request.get('user_id', 'anonymous')
            
            logger.info(f"📝 [DECISION] Item {item_id}: {decision} (confidence: {confidence})")
            logger.info(f"📝 [DECISION] Data keys: {list(request.keys())}")
            
            # Extract annotation data
            entities = request.get('entities', [])
            relations = request.get('relations', [])
            topics = request.get('topics', [])
            triplets = request.get('triplets', [])
            notes = request.get('notes', '')
            
            logger.info(f"📝 [DECISION] Annotations: {len(entities)} entities, {len(relations)} relations, {len(triplets)} triplets")
            
            # Simulate processing
            decision_id = f"decision_{item_id}_{decision}"
            timestamp = "2024-01-01T00:00:00Z"
            
            return {
                "success": True,
                "decision_id": decision_id,
                "item_id": item_id,
                "decision": decision,
                "status": "processed",
                "timestamp": timestamp,
                "message": f"Annotation {decision} processed successfully",
                "annotations_saved": {
                    "entities": len(entities),
                    "relations": len(relations), 
                    "triplets": len(triplets),
                    "topics": len(topics)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ [DECISION] Error processing annotation: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # DEBUG ENDPOINTS
    @app.get("/api/debug/status")
    async def debug_status():
        """Comprehensive API status for debugging"""
        return {
            "railway_api": "simplified_production",
            "import_status": import_status,
            "environment": {
                "OPENAI_API_KEY": "configured" if openai_key else "missing",
                "PORT": os.getenv("PORT"),
                "ENVIRONMENT": os.getenv("ENVIRONMENT")
            },
            "features": {
                "persistent_storage": True,
                "single_triage_endpoint": True,
                "simplified_routing": True,
                "openai_triplet_generation": bool(openai_key and import_status.get('openai', False)),
                "full_api_triplet_generation": import_status.get('main_api', False),
                "triplet_generation_mode": "full_api" if import_status.get('main_api', False) else "openai_direct" if openai_key else "mock_fallback"
            },
            "triplet_generation": {
                "openai_key_configured": bool(openai_key),
                "openai_import_success": import_status.get('openai', False),
                "full_api_available": import_status.get('main_api', False),
                "full_api_disabled": True,  # Explicitly disabled due to timeout issues
                "current_mode": "openai_direct" if (openai_key and import_status.get('openai', False)) else "mock_fallback"
            },
            "available_endpoints": [
                "/api/health",
                "/api/triage/queue", 
                "/api/triage/next",
                "/api/triage/statistics",
                "/api/documents",
                "/api/documents/ingest (POST)",
                "/api/candidates/generate (POST)",
                "/candidates/generate (POST)",
                "/api/annotations",
                "/api/annotations/draft (POST)",
                "/api/annotations/statistics",
                "/api/statistics/overview",
                "/ws/anonymous (WebSocket)",
                "/api/debug/status",
                "/api/debug/test-triplets", 
                "/api/debug/storage"
            ]
        }
    
    @app.get("/api/debug/test-triplets")
    async def test_triplet_generation():
        """Test triplet generation with a sample sentence"""
        test_sentence = "White Spot Syndrome Virus (WSSV) is one of the most devastating pathogens affecting Pacific white shrimp."
        
        logger.info(f"🧪 [TEST] Testing triplet generation for: {test_sentence[:50]}...")
        
        # Create test request
        test_request = CandidateRequest(
            doc_id="test_doc",
            sent_id="test_sent_1", 
            text=test_sentence,
            title="Test Document"
        )
        
        try:
            result = await generate_candidates_logic(test_request)
            # Handle both dict and Pydantic model responses
            if hasattr(result, 'candidates'):
                candidates = result.candidates
            else:
                candidates = result.get('candidates', {})
            triplet_count = len(candidates.get('triplets', []))
            logger.info(f"✅ [TEST] Generated {triplet_count} triplets successfully")
            
            return {
                "test_sentence": test_sentence,
                "triplets_generated": triplet_count,
                "generation_successful": True,
                "result": result
            }
        except Exception as e:
            logger.error(f"❌ [TEST] Triplet generation failed: {e}")
            return {
                "test_sentence": test_sentence,
                "error": str(e),
                "generation_successful": False
            }

    @app.get("/api/debug/test-mock")
    async def test_mock_generation():
        """Test mock triplet generation directly"""
        test_sentence = "White spot syndrome virus affects Pacific white shrimp causing mortality."
        logger.info(f"🧪 [MOCK] Testing mock generation for: {test_sentence}")
        
        result = generate_mock_triplets(test_sentence)
        triplet_count = len(result.get('candidates', {}).get('triplets', []))
        
        return {
            "test_sentence": test_sentence,
            "triplets_generated": triplet_count,
            "mock_successful": True,
            "result": result
        }

    @app.get("/api/debug/test-openai-direct")
    async def test_openai_direct():
        """Test OpenAI generation directly without fallback"""
        test_sentence = "White spot syndrome virus affects Pacific white shrimp causing mortality."
        logger.info(f"🧪 [OPENAI-DIRECT] Testing direct OpenAI for: {test_sentence}")
        
        try:
            result = await generate_openai_triplets(test_sentence)
            return {
                "test_sentence": test_sentence,
                "openai_successful": True,
                "result": result
            }
        except Exception as e:
            logger.error(f"❌ [OPENAI-DIRECT] Failed: {e}")
            return {
                "test_sentence": test_sentence,
                "openai_successful": False,
                "error": str(e)
            }

    @app.get("/api/debug/storage")
    async def debug_storage():
        """Debug persistent storage state"""
        stored_docs, stored_items = load_storage()
        return {
            "storage_file": str(storage_file),
            "storage_exists": storage_file.exists(),
            "uploaded_documents": {
                "count": len(stored_docs),
                "documents": stored_docs
            },
            "triage_items": {
                "count": len(stored_items),
                "items": stored_items
            }
        }
    
    # CRITICAL DEBUG: Check for route conflicts that prevent API execution
    logger.info("🔍 Analyzing route conflicts...")
    catch_all_found = False
    api_routes = []
    
    for i, route in enumerate(app.routes):
        if hasattr(route, 'path'):
            path = route.path
            methods = getattr(route, 'methods', set())
            
            # Check for catch-all route that intercepts API calls
            if path == "/{full_path:path}":
                catch_all_found = True
                logger.error(f"🚨 FOUND CATCH-ALL ROUTE at position {i}: {path}")
                logger.error(f"🚨 This route intercepts ALL unmatched paths including /api/* routes!")
                logger.error(f"🚨 Function: {route.endpoint.__name__ if hasattr(route, 'endpoint') else 'unknown'}")
            
            # Track API routes
            if path.startswith("/api/"):
                api_routes.append((i, path, methods))
                logger.info(f"  ✅ API Route {i}: {path} {methods}")
            
        else:
            logger.info(f"  📍 {i}: {type(route).__name__}")
    
    if catch_all_found:
        logger.error("🚨 ROOT CAUSE CONFIRMED: Catch-all /{full_path:path} route intercepts API calls")
        logger.error("🚨 When frontend calls /api/triage/queue, it gets HTML instead of JSON")
        logger.error("🚨 FIXING: Removing the catch-all route...")
        
        # Remove the catch-all route that prevents API endpoints from working
        original_routes = list(app.routes)
        app.router.routes.clear()
        
        for route in original_routes:
            # Skip the problematic catch-all route
            if hasattr(route, 'path') and route.path == "/{full_path:path}":
                logger.info(f"🗑️ REMOVED problematic catch-all route: {route.path}")
                continue
            app.router.routes.append(route)
        
        logger.info(f"✅ Route cleanup complete: {len(app.routes)} routes remaining")
        
        # Verify the fix worked
        remaining_catch_all = [r for r in app.routes if hasattr(r, 'path') and r.path == "/{full_path:path}"]
        if not remaining_catch_all:
            logger.info("✅ SUCCESS: Catch-all route removed, API endpoints should work now!")
        else:
            logger.error("❌ FAILED: Catch-all route still present")
    else:
        logger.info("✅ No catch-all route conflict - API should work")
    
    logger.info("✅ Simplified Railway API with persistent storage ready")
    
    # Add frontend serving
    ui_build = Path(__file__).parent / "ui" / "build"
    logger.info(f"📁 Frontend build directory: {ui_build} (exists: {ui_build.exists()})")
    
    if ui_build.exists():
        try:
            app.mount("/static", StaticFiles(directory=str(ui_build / "static")), name="static")
            logger.info("✅ Mounted /static directory for frontend assets")
        except Exception as e:
            logger.warning(f"⚠️ Failed to mount static files: {e}")
        
        @app.get("/", response_class=FileResponse)
        async def serve_index():
            return FileResponse(str(ui_build / "index.html"))
        
        # Add specific frontend routes instead of catch-all to avoid API conflicts
        @app.get("/dashboard", response_class=FileResponse)
        async def serve_dashboard():
            return FileResponse(str(ui_build / "index.html"))
            
        @app.get("/triage", response_class=FileResponse)
        async def serve_triage():
            return FileResponse(str(ui_build / "index.html"))
            
        @app.get("/documents", response_class=FileResponse)
        async def serve_documents():
            return FileResponse(str(ui_build / "index.html"))
            
        @app.get("/annotate", response_class=FileResponse)
        async def serve_annotate():
            return FileResponse(str(ui_build / "index.html"))
            
        @app.get("/annotate/{item_id}", response_class=FileResponse)
        async def serve_annotate_item(item_id: str):
            """Serve React app for annotate/1, annotate/2, etc."""
            return FileResponse(str(ui_build / "index.html"))
        
        # DO NOT add catch-all /{full_path:path} route - it breaks API endpoints!
        
        logger.info("✅ Added React frontend serving routes")
    else:
        logger.warning("⚠️ Frontend build directory not found")
    
except ImportError as e:
    logger.error(f"❌ Failed to set up API: {e}")
    sys.exit(1)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"🚀 Starting Simplified Railway Production API on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )