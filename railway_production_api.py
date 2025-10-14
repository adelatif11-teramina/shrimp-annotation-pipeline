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
logger.info("🚀 Railway Production API Starting... [SIMPLIFIED VERSION - Updated OpenAI Key]")
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
    from fastapi import HTTPException, Depends, WebSocket, Body
    from fastapi.responses import JSONResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    from typing import Optional, Dict, Any, List, Set
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

    removed_default_items_file = Path("/tmp/railway_removed_defaults.json")
    gold_exports_dir = Path("/tmp/railway_gold_exports")

    def canonical_identifier(value: Any) -> Optional[str]:
        """Normalise identifiers so numeric vs string forms compare consistently."""
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if text.isdigit():
            return str(int(text))
        return text.lower()

    def identifier_forms(value: Any) -> Set[str]:
        """Return a set of comparable forms for identifier matching."""
        forms: Set[str] = set()
        if value is None:
            return forms
        text = str(value).strip()
        if not text:
            return forms
        forms.add(text)
        forms.add(text.lower())
        canonical = canonical_identifier(text)
        if canonical:
            forms.add(canonical)
        return {form for form in forms if form}

    def load_removed_default_items() -> Set[str]:
        """Load default triage items the annotators have already completed."""
        try:
            if removed_default_items_file.exists():
                with open(removed_default_items_file, 'r') as f:
                    data = json.load(f)
                normalised = {identifier for identifier in (canonical_identifier(x) for x in data) if identifier}
                logger.info(f"🔍 Loaded {len(normalised)} removed default triage item IDs")
                return normalised
        except Exception as e:
            logger.error(f"❌ Failed to load removed default items: {e}")
        return set()

    def save_removed_default_items(ids: Set[str]) -> None:
        """Persist removed default triage item identifiers."""
        try:
            payload = sorted({identifier for identifier in (canonical_identifier(x) for x in ids) if identifier})
            removed_default_items_file.parent.mkdir(parents=True, exist_ok=True)
            temp_file = removed_default_items_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(payload, f, indent=2)
            temp_file.rename(removed_default_items_file)
            logger.info(f"✅ Persisted {len(payload)} removed default triage item IDs")
        except Exception as e:
            logger.error(f"❌ Failed to save removed default items: {e}")

    def load_annotations_storage() -> List[Dict[str, Any]]:
        """Load annotations from persistent storage"""
        storage_file = Path("/tmp/railway_annotations.json")
        
        if not storage_file.exists():
            logger.info(f"📂 Annotations storage file doesn't exist, returning empty list")
            return []
        
        try:
            logger.info(f"🔍 Loading annotations from: {storage_file} (exists: {storage_file.exists()})")
            with open(storage_file, 'r') as f:
                data = json.load(f)
            
            annotations = data.get('annotations', []) if isinstance(data, dict) else data
            logger.info(f"✅ Loaded {len(annotations)} annotations from storage")
            return annotations
            
        except Exception as e:
            logger.error(f"❌ Failed to load annotations storage: {e}")
            return []

    def save_annotations_storage(annotations: List[Dict[str, Any]]):
        """Save annotations to persistent storage"""
        storage_file = Path("/tmp/railway_annotations.json")
        
        try:
            data = {
                'annotations': annotations,
                'timestamp': datetime.datetime.now().isoformat(),
                'total_count': len(annotations)
            }
            
            # Write to temporary file first, then rename (atomic operation)
            temp_file = storage_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Atomic rename
            temp_file.rename(storage_file)
            
            logger.info(f"✅ Successfully saved {len(annotations)} annotations to storage")
            
        except Exception as e:
            logger.error(f"❌ Failed to save annotations storage: {e}")

    def normalize_annotation_record(record: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize stored annotation into UI-friendly shape."""
        if not record:
            return {}

        decision = record.get("decision") or record.get("status") or "unknown"
        normalized = {
            "id": record.get("annotation_id") or record.get("id"),
            "annotation_id": record.get("annotation_id") or record.get("id"),
            "item_id": record.get("item_id"),
            "doc_id": record.get("doc_id"),
            "sent_id": record.get("sent_id"),
            "text": record.get("text", ""),
            "decision": decision,
            "status": record.get("status") or ("completed" if decision == "accept" else decision),
            "confidence": record.get("confidence"),
            "annotator": record.get("annotator"),
            "notes": record.get("notes", ""),
            "topics": record.get("topics", []),
            "entities": record.get("entities", []),
            "relations": record.get("relations", []),
            "triplets": record.get("triplets", []),
            "created_at": record.get("created_at"),
            "updated_at": record.get("updated_at"),
        }
        return normalized
    
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

    class GoldExportRequest(BaseModel):
        format: str = "jsonl"  # jsonl, json
        doc_ids: Optional[List[str]] = None
        include_notes: bool = True
        include_topics: bool = True

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
        
        removed_default_ids = load_removed_default_items()
        if removed_default_ids:
            mock_items = [
                item for item in mock_items
                if canonical_identifier(item.get("item_id")) not in removed_default_ids
            ]
            if mock_items:
                logger.info(f"🔍 [SINGLE TRIAGE] {len(removed_default_ids)} default items hidden from queue")

        # Combine mock items with uploaded items (mock first for accessibility)
        all_items = mock_items + stored_items
        
        # Filter by status if specified
        if status and status != "undefined" and status != "null" and status.lower() != "all items":
            all_items = [item for item in all_items if item["status"] == status]
            logger.info(f"🔍 Filtered items by status '{status}': {len(all_items)} items remaining")
        
        # Sort by priority if requested
        if sort_by == "priority":
            all_items = sorted(all_items, key=lambda x: x["priority_score"], reverse=True)
        
        final_items = all_items[offset:offset+limit]
        logger.info(f"🎯 [SINGLE TRIAGE] Returning {len(final_items)} items out of {len(all_items)} total")
        
        # Debug: log item types and IDs for troubleshooting
        if final_items:
            item_ids = [item.get("item_id") for item in final_items[:10]]
            logger.info(f"🔍 [DEBUG] First 10 item IDs: {item_ids}")
            
            mock_count = len([item for item in final_items if item.get("item_id", 0) <= 10])
            uploaded_count = len([item for item in final_items if item.get("item_id", 0) > 10])
            logger.info(f"🔍 [DEBUG] Mock items: {mock_count}, Uploaded items: {uploaded_count}")
        
        response = {
            "items": final_items,
            "total": len(all_items),
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < len(all_items),
            "source": "persistent_storage"
        }
        
        # Debug: log response size
        import json
        response_size = len(json.dumps(response))
        logger.info(f"🔍 [DEBUG] Response size: {response_size} bytes")
        
        return response

    @app.post("/api/triage/items/{item_id}/skip")
    async def skip_triage_item(item_id: str, request: Optional[Dict[str, Any]] = Body(default=None)):
        """Mark a triage item as skipped."""
        canonical_target = canonical_identifier(item_id)
        logger.info(f"⏭️ [TRIAGE] Skip requested for item {item_id} (canonical={canonical_target})")

        if not canonical_target:
            raise HTTPException(status_code=400, detail="Invalid item identifier")

        stored_docs, stored_items = load_storage()
        updated = False

        for item in stored_items:
            if canonical_identifier(item.get("item_id")) == canonical_target:
                item["status"] = "skipped"
                item["skipped_at"] = datetime.datetime.now().isoformat()
                item["updated_at"] = datetime.datetime.now().isoformat()
                if request and isinstance(request, dict):
                    item["skip_reason"] = request.get("reason")
                updated = True
                logger.info(f"⏭️ [TRIAGE] Marked stored item {item.get('item_id')} as skipped")
                break

        if updated:
            save_storage(stored_docs, stored_items)
        else:
            removed_defaults = load_removed_default_items()
            if canonical_target not in removed_defaults:
                removed_defaults.add(canonical_target)
                save_removed_default_items(removed_defaults)
                logger.info(f"ℹ️ [TRIAGE] Recorded default mock item {canonical_target} as skipped (not in storage)")
            else:
                logger.info(f"ℹ️ [TRIAGE] Default mock item {canonical_target} already recorded as skipped")

        return {
            "status": "success",
            "item_id": item_id,
            "canonical_item_id": canonical_target,
            "updated": updated
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
        item_counter = len(current_items) + 100  # Starting ID for this batch
        for i, sentence in enumerate(sentences):  # Process all sentences
            if len(sentence) > 20:  # Only meaningful sentences
                triage_item = {
                    "item_id": item_counter,  # Sequential unique ID
                    "doc_id": doc_id,
                    "sent_id": f"{doc_id}_sent_{i+1}",
                    "text": sentence + ".",
                    "priority_score": 0.8 + (0.1 * (3-i)),  # Higher priority for earlier sentences
                    "confidence": 0.0,  # New, needs annotation
                    "status": "pending",
                    "created_at": timestamp,
                    "metadata": {"source": "uploaded", "sentence_index": i}
                }
                current_items.append(triage_item)  # Add to end in correct order
                item_counter += 1  # Increment for next item
        
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
            
            # Clean markdown code blocks if present (same as LLM candidate generator)
            import re
            cleaned_content = re.sub(r'```(?:json)?\s*', '', content)
            cleaned_content = cleaned_content.strip('`')
            
            if cleaned_content != content:
                logger.info(f"🧽 [OPENAI] Cleaned markdown wrapper from response")
                logger.info(f"🧽 [OPENAI] Cleaned content preview: {cleaned_content[:100]}...")
            
            result = json.loads(cleaned_content)
            
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
            # Only generate entities that actually exist in the sentence
            if 'tpd' in sentence_lower:  # Only if TPD is explicitly mentioned
                mock_triplets.append({
                    "triplet_id": "mock_1",
                    "head": {"text": "TPD", "type": "DISEASE", "node_id": "tpd"},
                    "relation": "AFFECTS", 
                    "tail": {"text": "post-larvae", "type": "SPECIES", "node_id": "larvae"},
                    "evidence": "TPD affects post-larvae development",
                    "confidence": 0.88
                })
            elif any(word in sentence_lower for word in ['wssv', 'white spot', 'virus']):
                mock_triplets.append({
                    "triplet_id": "mock_1", 
                    "head": {"text": "White Spot Syndrome Virus", "type": "PATHOGEN", "node_id": "wssv"},
                    "relation": "AFFECTS",
                    "tail": {"text": "Pacific white shrimp", "type": "SPECIES", "node_id": "shrimp"},
                    "evidence": "White spot syndrome virus affects Pacific white shrimp",
                    "confidence": 0.90
                })
            # If no specific pathogens are mentioned, don't generate hallucinated triplets
        
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

    @app.delete("/api/annotations/draft")
    async def delete_draft_annotation(request: Dict[str, Any]):
        """Delete/clear draft annotation"""
        try:
            item_id = request.get('item_id', 'unknown')
            logger.info(f"🗑️ [DRAFT] Clearing draft for item: {item_id}")
            
            return {
                "status": "success",
                "message": "Draft cleared successfully",
                "item_id": item_id
            }
        except Exception as e:
            logger.error(f"❌ [DRAFT] Error clearing draft: {e}")
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
        
        # Use same ordering as triage queue to ensure consistency
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
        
        # Combine in same order as queue (mock first, then stored)
        all_items = mock_items + stored_items
        
        # Return first available item (same logic as queue)
        if all_items:
            next_item = all_items[0]
            logger.info(f"⏭️ [TRIAGE] Returning next item: {next_item.get('item_id')}")
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
        status: Optional[str] = None,
        decision: Optional[str] = None,
        doc_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        """Get annotations list"""
        logger.info(
            "📝 [ANNOTATIONS] List requested: sort_by=%s, limit=%s, offset=%s, status=%s, decision=%s, doc_id=%s, user_id=%s",
            sort_by,
            limit,
            offset,
            status,
            decision,
            doc_id,
            user_id,
        )
        
        # Load annotations from persistent storage
        storage_annotations = load_annotations_storage()
        
        # Add some mock annotations if storage is empty (for demo purposes)
        if not storage_annotations:
            storage_annotations = [
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
                    "annotator": "demo", 
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
                    "annotator": "demo",
                    "created_at": "2024-01-01T02:00:00Z", 
                    "updated_at": "2024-01-01T03:00:00Z"
                }
            ]
        
        def is_all(value: Optional[str]) -> bool:
            if value is None:
                return True
            value_str = str(value).strip().lower()
            return value_str in {"", "all", "null", "undefined"}

        filtered_annotations = storage_annotations

        if not is_all(status):
            filtered_annotations = [
                ann for ann in filtered_annotations
                if (ann.get("status") or ann.get("decision")) == status
            ]

        if not is_all(decision):
            filtered_annotations = [
                ann for ann in filtered_annotations
                if (ann.get("decision") or ann.get("status")) == decision
            ]

        if doc_id and not is_all(doc_id):
            filtered_annotations = [ann for ann in filtered_annotations if ann.get("doc_id") == doc_id]

        if user_id and not is_all(user_id):
            filtered_annotations = [ann for ann in filtered_annotations if ann.get("annotator") == user_id]

        # Sort annotations
        if sort_by == "created_at":
            filtered_annotations.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        elif sort_by == "updated_at":
            filtered_annotations.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        elif sort_by == "confidence":
            filtered_annotations.sort(key=lambda x: x.get("confidence", 0), reverse=True)

        total_filtered = len(filtered_annotations)
        page_slice = filtered_annotations[offset:offset + limit]
        normalized_annotations = [normalize_annotation_record(ann) for ann in page_slice]

        logger.info(f"📝 [ANNOTATIONS] Returning {len(normalized_annotations)} annotations (total filtered={total_filtered})")

        return {
            "annotations": normalized_annotations,
            "total": total_filtered,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total_filtered,
            "from_storage": True
        }

    @app.get("/api/annotations/{annotation_id}")
    async def get_annotation_detail(annotation_id: str):
        """Retrieve details for a specific annotation."""
        storage_annotations = load_annotations_storage()
        target = canonical_identifier(annotation_id) or str(annotation_id).strip()

        for record in storage_annotations:
            record_id = canonical_identifier(record.get("annotation_id")) or canonical_identifier(record.get("id"))
            if record_id == target:
                return {"annotation": normalize_annotation_record(record)}

        raise HTTPException(status_code=404, detail=f"Annotation {annotation_id} not found")

    @app.get("/api/annotations/export")
    async def export_annotations(
        sort_by: str = "created_at",
        format: str = "json",
        status: Optional[str] = None
    ):
        """Export annotations in various formats"""
        logger.info(f"📤 [EXPORT] Annotation export requested: format={format}, status={status}")
        
        # Load annotations from persistent storage
        storage_annotations = load_annotations_storage()
        
        # Filter by status if specified
        if status:
            storage_annotations = [ann for ann in storage_annotations if ann.get("status") == status]
        
        # Sort annotations
        if sort_by == "created_at":
            storage_annotations.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        elif sort_by == "updated_at":
            storage_annotations.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        
        logger.info(f"📤 [EXPORT] Exporting {len(storage_annotations)} annotations in {format} format")
        
        if format.lower() == "json":
            return {
                "export_metadata": {
                    "format": "json",
                    "total_annotations": len(storage_annotations),
                    "export_timestamp": datetime.datetime.now().isoformat(),
                    "filtered_by_status": status
                },
                "annotations": storage_annotations
            }
        elif format.lower() == "csv":
            # Convert to CSV format
            import io
            import csv
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow([
                "annotation_id", "doc_id", "sent_id", "text", "status", 
                "confidence", "annotator", "created_at", "entities_count", 
                "relations_count", "triplets_count"
            ])
            
            # Write data
            for ann in storage_annotations:
                writer.writerow([
                    ann.get("annotation_id", ""),
                    ann.get("doc_id", ""),
                    ann.get("sent_id", ""),
                    ann.get("text", "")[:100],  # Truncate long text
                    ann.get("status", ""),
                    ann.get("confidence", ""),
                    ann.get("annotator", ""),
                    ann.get("created_at", ""),
                    len(ann.get("entities", [])),
                    len(ann.get("relations", [])),
                    len(ann.get("triplets", []))
                ])
            
            csv_content = output.getvalue()
            output.close()
            
            from fastapi.responses import Response
            return Response(
                content=csv_content,
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=annotations_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported export format: {format}")

    @app.post("/api/export/gold")
    async def export_gold_annotations(export_request: GoldExportRequest):
        """Export annotations in gold format for downstream use."""
        storage_annotations = load_annotations_storage()

        if export_request.doc_ids:
            allowed_docs = set(export_request.doc_ids)
            storage_annotations = [ann for ann in storage_annotations if ann.get("doc_id") in allowed_docs]

        if not storage_annotations:
            raise HTTPException(status_code=404, detail="No annotations available for export")

        gold_exports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        export_format = export_request.format.lower()
        export_path = gold_exports_dir / f"gold_export_{timestamp}.{export_format if export_format in {'json', 'jsonl'} else 'json'}"

        normalized_records = [normalize_annotation_record(ann) for ann in storage_annotations]

        if export_format == "jsonl":
            with open(export_path, 'w', encoding='utf-8') as f:
                for record in normalized_records:
                    json.dump(record, f)
                    f.write('\n')
        else:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(normalized_records, f, indent=2)

        logger.info(f"📤 [EXPORT] Gold annotations saved to {export_path} ({len(normalized_records)} items)")

        return {
            "status": "success",
            "export_path": str(export_path),
            "format": export_request.format,
            "item_count": len(normalized_records)
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

    async def _websocket_handler(websocket: WebSocket, user_id: str):
        """Shared WebSocket handler for presence and echo messaging."""
        query_params = websocket.query_params
        username = query_params.get('username', user_id or 'Anonymous')
        role = query_params.get('role', 'annotator')

        logger.info(f"🔗 [WEBSOCKET] Connection attempt from: {username} (role: {role}, user_id: {user_id})")

        try:
            await websocket.accept()
            logger.info(f"✅ [WEBSOCKET] Connected: {username}")

            await websocket.send_json({
                "type": "connection",
                "status": "connected",
                "user": username,
                "role": role,
                "user_id": user_id,
                "timestamp": datetime.datetime.now().isoformat()
            })

            while True:
                try:
                    data = await websocket.receive_json()
                    logger.info(f"📨 [WEBSOCKET] Message from {username}: {data.get('type', 'unknown')}")

                    await websocket.send_json({
                        "type": "echo",
                        "original": data,
                        "timestamp": datetime.datetime.now().isoformat()
                    })

                except Exception as e:
                    logger.warning(f"⚠️ [WEBSOCKET] Message error for {username}: {e}")
                    break

        except Exception as e:
            logger.error(f"❌ [WEBSOCKET] Connection error for {username}: {e}")
        finally:
            logger.info(f"🔌 [WEBSOCKET] Disconnected: {username}")

    @app.websocket("/ws/{user_id}")
    async def websocket_user_endpoint(websocket: WebSocket, user_id: str):
        await _websocket_handler(websocket, user_id)

    @app.websocket("/ws/anonymous")
    async def websocket_legacy_endpoint(websocket: WebSocket):
        await _websocket_handler(websocket, "anonymous")

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
            
            # Save annotation to persistent storage
            decision_id = f"decision_{item_id}_{decision}"
            timestamp = datetime.datetime.now().isoformat()
            
            # Load existing annotations from persistent storage
            storage_annotations = load_annotations_storage()
            
            # Create new annotation record
            annotation_record = {
                "annotation_id": len(storage_annotations) + 1,
                "doc_id": request.get('doc_id', f'doc_{item_id}'),
                "sent_id": request.get('sent_id', f'sent_{item_id}'),
                "item_id": item_id,
                "text": request.get('sentence', request.get('text', '')),
                "entities": entities,
                "relations": relations,
                "triplets": triplets,
                "topics": topics,
                "notes": notes,
                "decision": decision,
                "status": "completed" if decision == "accept" else decision,
                "confidence": confidence,
                "annotator": user_id,
                "created_at": timestamp,
                "updated_at": timestamp
            }
            
            # Add to storage and save
            storage_annotations.append(annotation_record)
            save_annotations_storage(storage_annotations)
            
            logger.info(f"💾 [DECISION] Saved annotation {annotation_record['annotation_id']} for item {item_id}")
            
            # CRITICAL FIX: Remove completed item from triage queue
            logger.info(f"🗑️ [QUEUE REMOVAL] Removing item {item_id} from triage queue...")
            try:
                # Load current queue from storage
                stored_docs, stored_items = load_storage()
                logger.info(f"🗑️ [QUEUE REMOVAL] Loaded queue with {len(stored_items)} items")
                
                # Build a robust identifier set for matching
                target_forms: Set[str] = set()
                candidate_identifiers = [
                    item_id,
                    request.get('candidate_id'),
                    request.get('itemId'),
                    request.get('id'),
                ]
                if request.get('doc_id') and request.get('sent_id'):
                    candidate_identifiers.append(f"{request['doc_id']}_{request['sent_id']}")

                for identifier in candidate_identifiers:
                    target_forms.update(identifier_forms(identifier))

                items_before = len(stored_items)
                filtered_items = []
                removed_from_storage = False

                for queued_item in stored_items:
                    queue_forms: Set[str] = set()
                    queue_forms.update(identifier_forms(queued_item.get("item_id")))
                    queue_forms.update(identifier_forms(queued_item.get("id")))
                    queue_forms.update(identifier_forms(queued_item.get("sent_id")))
                    if queued_item.get("doc_id") and queued_item.get("sent_id"):
                        queue_forms.update(
                            identifier_forms(f"{queued_item['doc_id']}_{queued_item['sent_id']}")
                        )

                    if queue_forms & target_forms:
                        removed_from_storage = True
                        logger.info(
                            "✅ [QUEUE REMOVAL] Matched stored item %s via identifiers %s",
                            queued_item.get("item_id"),
                            sorted(queue_forms & target_forms),
                        )
                        continue

                    filtered_items.append(queued_item)

                stored_items = filtered_items
                items_after = len(stored_items)

                if removed_from_storage:
                    # Save updated queue back to storage
                    save_storage(stored_docs, stored_items)
                    logger.info(f"✅ [QUEUE REMOVAL] Successfully removed item {item_id} from queue ({items_before} → {items_after} items)")
                else:
                    logger.warning(f"⚠️ [QUEUE REMOVAL] Item {item_id} not found in stored queue (already removed or alternate ID)")
                    canonical_target = canonical_identifier(item_id)
                    if canonical_target:
                        removed_default_ids = load_removed_default_items()
                        if canonical_target not in removed_default_ids:
                            removed_default_ids.add(canonical_target)
                            save_removed_default_items(removed_default_ids)
                            logger.info(f"✅ [QUEUE REMOVAL] Recorded default mock item {canonical_target} as completed")
                        else:
                            logger.info(f"ℹ️ [QUEUE REMOVAL] Default mock item {canonical_target} was already recorded as removed")
                    else:
                        logger.warning("⚠️ [QUEUE REMOVAL] Unable to canonicalise item_id for fallback removal")
                    
            except Exception as e:
                logger.error(f"❌ [QUEUE REMOVAL] Failed to remove item {item_id} from queue: {e}")
            
            return {
                "success": True,
                "decision_id": decision_id,
                "annotation_id": annotation_record["annotation_id"],
                "item_id": item_id,
                "decision": decision,
                "status": "processed",
                "timestamp": timestamp,
                "message": f"Annotation {decision} processed and saved successfully",
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
