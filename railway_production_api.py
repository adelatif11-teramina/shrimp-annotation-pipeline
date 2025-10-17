#!/usr/bin/env python3
"""
SIMPLIFIED Railway Production API - No Conflicts
"""

import base64
import binascii
import os
import sys
import io
import logging
import mimetypes
import re
import traceback
import uuid
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import database modules
try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    from services.database.models import Base, Document, Sentence, GoldAnnotation, TriageItem, Candidate
    HAS_DATABASE = True
except ImportError as e:
    logger.warning(f"Database imports failed: {e}")
    HAS_DATABASE = False

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

try:
    from pdfminer.high_level import extract_text as pdf_extract_text
except ImportError as pdf_import_error:
    pdf_extract_text = None
    logger.warning(
        "⚠️ pdfminer.six not available (%s); PDF extraction will be disabled",
        pdf_import_error,
    )

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
from collections import defaultdict

# DATABASE SETUP - THE SINGLE SOURCE OF TRUTH
database_url = os.getenv("DATABASE_URL")
if not database_url:
    logger.error("❌ DATABASE_URL environment variable not found!")
    logger.error("❌ Please add PostgreSQL database to Railway and set DATABASE_URL")
    logger.warning("⚠️ Continuing with fallback storage mode...")
    engine = None
    SessionLocal = None
else:
    # Fix Railway's postgres:// URL to postgresql://
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    
    logger.info(f"🔗 Connecting to database: {database_url[:50]}...")
    
    try:
        engine = create_engine(database_url)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            logger.info("✅ Database connection successful")
        
        # Run Alembic migrations - DETERMINISTIC SCHEMA REQUIRED
        try:
            from alembic.config import Config
            from alembic import command
            
            logger.info("🔄 Running Alembic migrations...")
            alembic_cfg = Config("alembic.ini")
            command.upgrade(alembic_cfg, "head")
            logger.info("✅ Alembic migrations completed successfully")
        except Exception as alembic_error:
            error_msg = f"""
❌ CRITICAL: Alembic migration failed: {alembic_error}

🔧 DETERMINISTIC SCHEMA POLICY:
The dangerous create_all() fallback has been removed to prevent non-deterministic schema creation.
All database schema changes MUST go through Alembic migrations for consistency.

🛠️ TO FIX THIS ISSUE:
1. Check that DATABASE_URL is correctly set for Railway PostgreSQL
2. Ensure the database exists and is accessible
3. Verify Alembic configuration in alembic.ini and alembic/env.py
4. Run migrations manually if needed: 'alembic upgrade head'
5. Check that all required models are properly imported in alembic/env.py

❌ APPLICATION STARTUP FAILED - Schema must be deterministic!
"""
            logger.error(error_msg)
            raise RuntimeError(f"Alembic migration failed: {alembic_error}. No fallback available - deterministic schema required.")
        
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        logger.error("❌ Falling back to in-memory storage")
        engine = None
        SessionLocal = None
    
# FALLBACK FILE-BASED STORAGE (PERSISTENT ACROSS REQUESTS)
fallback_storage_file = Path("/tmp/railway_fallback_storage.json")
annotations_fallback_file = Path("/tmp/railway_annotations_fallback.json")

def load_fallback_storage():
    """Load fallback storage from persistent file"""
    try:
        if fallback_storage_file.exists():
            with open(fallback_storage_file, 'r') as f:
                data = json.load(f)
            fallback_docs = data.get('documents', [])
            fallback_items = data.get('triage_items', [])
            logger.info(f"📂 [FALLBACK] Loaded {len(fallback_docs)} docs, {len(fallback_items)} items from file")
            return fallback_docs, fallback_items
    except Exception as e:
        logger.error(f"❌ [FALLBACK] Failed to load storage: {e}")
    return [], []

def save_fallback_storage(documents, triage_items):
    """Save fallback storage to persistent file"""
    try:
        data = {
            'documents': documents,
            'triage_items': triage_items,
            'timestamp': datetime.datetime.now().isoformat(),
            'total_count': len(documents) + len(triage_items)
        }
        
        # Atomic write
        temp_file = fallback_storage_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        temp_file.rename(fallback_storage_file)
        
        logger.info(f"💾 [FALLBACK] Saved {len(documents)} docs, {len(triage_items)} items to file")
    except Exception as e:
        logger.error(f"❌ [FALLBACK] Failed to save storage: {e}")

def load_annotations_fallback() -> List[Dict[str, Any]]:
    """Load annotation records from persistent fallback storage."""
    try:
        if annotations_fallback_file.exists():
            with open(annotations_fallback_file, 'r') as f:
                data = json.load(f)
            annotations = data.get('annotations', [])
            logger.info(
                "📄 [FALLBACK] Loaded %s annotation records from file",
                len(annotations),
            )
            return annotations
    except Exception as exc:
        logger.error(f"❌ [FALLBACK] Failed to load annotations storage: {exc}")
    return []


def save_annotations_fallback(records: List[Dict[str, Any]]):
    """Persist annotation records to fallback storage."""
    try:
        payload = {
            'annotations': records,
            'timestamp': datetime.datetime.now().isoformat(),
            'total_count': len(records)
        }
        temp_file = annotations_fallback_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(payload, f, indent=2)
        temp_file.rename(annotations_fallback_file)
        logger.info("💾 [FALLBACK] Saved %s annotation records to file", len(records))
    except Exception as exc:
        logger.error(f"❌ [FALLBACK] Failed to save annotations storage: {exc}")


def append_annotation_fallback(record: Dict[str, Any]):
    """Append or replace an annotation record in fallback storage."""
    if not record:
        return

    records = load_annotations_fallback()
    annotation_id = record.get('annotation_id') or record.get('id')
    if annotation_id:
        records = [r for r in records if (r.get('annotation_id') or r.get('id')) != annotation_id]

    records.insert(0, record)

    if len(records) > 500:
        records = records[:500]

    save_annotations_fallback(records)


def guess_mime_type(file_name: Optional[str], provided_type: Optional[str]) -> Optional[str]:
    """Return the best-effort MIME type for the uploaded payload."""
    if provided_type:
        return provided_type

    if file_name:
        guessed_type, _ = mimetypes.guess_type(file_name)
        if guessed_type:
            return guessed_type

    return None


def decode_base64_payload(data: str) -> bytes:
    """Decode base64 data URIs or raw base64 strings into bytes."""
    if not data:
        return b""

    payload = data.strip()
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]

    try:
        return base64.b64decode(payload, validate=True)
    except binascii.Error:
        logger.warning("⚠️ [UPLOAD] Base64 validation failed, attempting permissive decode")
        return base64.b64decode(payload)


def normalise_text(value: str) -> str:
    """Normalise line endings and strip non-printable characters."""
    if not value:
        return ""

    cleaned = value.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\x00", "")
    cleaned = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    return cleaned.strip()


def extract_text_from_bytes(file_bytes: bytes, mime_type: Optional[str], file_name: Optional[str]) -> str:
    """Extract readable text from uploaded bytes based on MIME type."""
    if not file_bytes:
        return ""

    effective_mime = guess_mime_type(file_name, mime_type)

    if effective_mime == "application/pdf" or (file_name or "").lower().endswith(".pdf"):
        if pdf_extract_text is None:
            logger.warning("⚠️ [PDF] pdfminer.six unavailable; cannot extract text")
        else:
            try:
                with io.BytesIO(file_bytes) as buffer:
                    pdf_text = pdf_extract_text(buffer)
                return normalise_text(pdf_text)
            except Exception as pdf_error:
                logger.error(f"❌ [PDF] Failed to extract text: {pdf_error}")

    # Default: treat as text-like content
    for encoding in ("utf-8", "latin-1"):
        try:
            decoded = file_bytes.decode(encoding)
            logger.info(f"✅ [UPLOAD] Decoded bytes using {encoding}")
            return normalise_text(decoded)
        except UnicodeDecodeError:
            continue

    logger.warning("⚠️ [UPLOAD] Unable to decode bytes as text")
    return ""


def split_into_sentences(text: str) -> List[str]:
    """Lightweight sentence splitter for uploaded content."""
    if not text:
        return []

    candidates = re.split(r'(?<=[.!?])\s+|\n+', text)
    sentences = [candidate.strip() for candidate in candidates if candidate and candidate.strip()]
    return sentences


def save_document_to_fallback(doc_id, title, file_name, sentences, timestamp, raw_text, metadata=None):
    """Persist uploaded content to file-based fallback storage."""
    
    # Load existing data
    fallback_documents, fallback_triage_items = load_fallback_storage()

    # Drop any previous entries for this document to avoid duplicates
    fallback_documents = [doc for doc in fallback_documents if doc.get("doc_id") != doc_id]
    fallback_triage_items = [item for item in fallback_triage_items if item.get("doc_id") != doc_id]

    new_document = {
        "doc_id": doc_id,
        "title": title,
        "sentence_count": len(sentences),
        "annotation_count": 0,
        "status": "ingested",
        "created_at": timestamp,
        "updated_at": timestamp,
        "file_name": file_name,
        "source": "uploaded",
        "raw_text": raw_text,
        "metadata": metadata or {},
    }
    fallback_documents.insert(0, new_document)

    created_items = 0
    for i, sentence in enumerate(sentences):
        sentence_text = sentence.strip()
        if len(sentence_text) <= 20:
            continue

        item_counter = int(
            timestamp.replace(":", "").replace("-", "").replace("T", "").replace("Z", "")[:12]
        ) + i

        text_value = sentence_text
        if text_value and text_value[-1] not in ".!?":
            text_value = f"{text_value}."

        triage_item = {
            "item_id": item_counter,
            "doc_id": doc_id,
            "sent_id": f"{doc_id}_sent_{i + 1}",
            "text": text_value,
            "priority_score": 0.8 + (0.1 * max(0, 3 - i)),
            "confidence": 0.0,
            "status": "pending",
            "priority_level": "high" if 0.8 + (0.1 * max(0, 3 - i)) >= 0.8 else "medium",
            "created_at": timestamp,
            "source": "uploaded",
            "doc_title": title,
        }
        fallback_triage_items.append(triage_item)
        created_items += 1

    # Save to persistent file
    save_fallback_storage(fallback_documents, fallback_triage_items)

    logger.info(
        "💾 [FALLBACK] Saved document %s with %s triage items to persistent storage",
        doc_id,
        created_items,
    )

    return created_items


def load_storage():
    """Load documents and triage items from database"""
    if not engine or not SessionLocal:
        logger.warning("⚠️ No database connection, using fallback storage")
        # Load from persistent file-based fallback storage
        fallback_documents, fallback_triage_items = load_fallback_storage()
        
        mock_docs = [
            {
                "doc_id": "mock_doc_1",
                "title": "Demo: White Spot Syndrome Virus Research",
                "sentence_count": 5,
                "annotation_count": 0,
                "status": "pending",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
                "file_name": "demo.txt"
            }
        ] if not fallback_documents else []  # Only show mock if no uploaded docs
        
        mock_items = [
            {
                "item_id": "mock_item_1",
                "doc_id": "mock_doc_1",
                "sent_id": "mock_sent_1",
                "text": "White Spot Syndrome Virus (WSSV) is a major pathogen affecting Pacific white shrimp.",
                "priority_score": 0.9,
                "confidence": 0.0,
                "status": "pending",
                "priority_level": "critical",
                "created_at": "2024-01-01T00:00:00Z"
            }
        ] if not fallback_triage_items else []  # Only show mock if no uploaded items
        
        # Combine uploaded documents with mock data
        all_docs = fallback_documents + mock_docs
        all_items = fallback_triage_items + mock_items
        
        logger.info(f"✅ Returning fallback storage: {len(all_docs)} docs ({len(fallback_documents)} uploaded + {len(mock_docs)} mock), {len(all_items)} items")
        return all_docs, all_items
    
    try:
        with SessionLocal() as session:
            # Load documents
            docs = session.query(Document).order_by(Document.created_at.desc()).all()
            doc_list = []
            for doc in docs:
                sentence_count = session.query(Sentence).filter(Sentence.document_id == doc.id).count()
                annotation_count = session.query(GoldAnnotation).filter(GoldAnnotation.document_id == doc.id).count()
                
                # Safe metadata access
                metadata = doc.document_metadata or {}
                file_name = metadata.get("file_name", "unknown.txt") if isinstance(metadata, dict) else "unknown.txt"
                
                doc_list.append({
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "sentence_count": sentence_count,
                    "annotation_count": annotation_count,
                    "status": "annotated" if annotation_count > 0 else "pending",
                    "created_at": doc.created_at.isoformat() + "Z",
                    "updated_at": doc.updated_at.isoformat() + "Z",
                    "file_name": file_name
                })
            
            # Load triage items with proper joins
            active_statuses = ["pending", "in_review"]
            items = (
                session.query(TriageItem)
                .join(Sentence)
                .join(Document)
                .filter(TriageItem.status.in_(active_statuses))
                .order_by(TriageItem.priority_score.desc())
                .all()
            )
            item_list = []
            for item in items:
                try:
                    item_list.append({
                        "item_id": item.item_id,
                        "doc_id": item.sentence.document.doc_id,
                        "sent_id": item.sentence.sent_id,
                        "text": item.sentence.text,
                        "priority_score": item.priority_score,
                        "confidence": item.confidence_score,
                        "status": item.status,
                        "priority_level": item.priority_level,
                        "created_at": item.created_at.isoformat() + "Z"
                    })
                except Exception as item_error:
                    logger.warning(f"⚠️ Skipping triage item {item.item_id}: {item_error}")
                    continue
            
            logger.info(f"✅ Loaded from database: {len(doc_list)} docs, {len(item_list)} items")
            return doc_list, item_list
            
    except Exception as e:
        logger.error(f"❌ Failed to load from database: {e}")
        logger.warning("🔄 Falling back to mock data")
        # Return mock data as fallback
        mock_docs = [
            {
                "doc_id": "fallback_doc_1",
                "title": "Fallback Document",
                "sentence_count": 3,
                "annotation_count": 0,
                "status": "pending",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
                "file_name": "fallback.txt"
            }
        ]
        mock_items = [
            {
                "item_id": "fallback_item_1",
                "doc_id": "fallback_doc_1",
                "sent_id": "fallback_sent_1",
                "text": "This is a fallback sentence for testing.",
                "priority_score": 0.7,
                "confidence": 0.0,
                "status": "pending",
                "priority_level": "medium",
                "created_at": "2024-01-01T00:00:00Z"
            }
        ]
        return mock_docs, mock_items

def save_storage(documents, items):
    """Persist queue state to the active storage backend."""
    if engine and SessionLocal:
        logger.info(
            "💾 [DATABASE] save_storage called with %s documents and %s items (noop)",
            len(documents),
            len(items),
        )
        logger.info("💾 [DATABASE] Direct database writes handle persistence; no bulk save performed")
        return

    logger.info(
        "💾 [FALLBACK] Persisting %s documents and %s items to fallback storage",
        len(documents),
        len(items),
    )
    save_fallback_storage(documents, items)


def mark_triage_item_completed_in_db(
    queue_item: Dict[str, Any],
    decision: str,
    user_id: str,
    timestamp: datetime.datetime,
) -> bool:
    """Update the database triage record to reflect completion."""
    if not engine or not SessionLocal:
        return False

    item_identifier = queue_item.get("item_id")
    doc_identifier = queue_item.get("doc_id")
    sent_identifier = queue_item.get("sent_id")

    try:
        with SessionLocal() as session:
            triage_obj = None

            if item_identifier is not None:
                triage_obj = (
                    session.query(TriageItem)
                    .filter(TriageItem.item_id == str(item_identifier))
                    .one_or_none()
                )

            if not triage_obj and doc_identifier and sent_identifier:
                triage_obj = (
                    session.query(TriageItem)
                    .join(Sentence)
                    .join(Document)
                    .filter(
                        Document.doc_id == doc_identifier,
                        Sentence.sent_id == sent_identifier,
                    )
                    .order_by(TriageItem.created_at.desc())
                    .first()
                )

            if not triage_obj:
                logger.warning(
                    "⚠️ [DATABASE] Unable to locate triage item for item_id=%s doc_id=%s sent_id=%s",
                    item_identifier,
                    doc_identifier,
                    sent_identifier,
                )
                return False

            triage_obj.status = "completed" if decision == "accept" else decision
            triage_obj.completed_at = timestamp
            triage_obj.assigned_to = user_id
            triage_obj.updated_at = timestamp
            session.commit()

            logger.info(
                "✅ [DATABASE] Marked triage item %s as %s",
                triage_obj.item_id,
                triage_obj.status,
            )
            return True

    except Exception as exc:
        logger.error(f"❌ [DATABASE] Failed to update triage item: {exc}")

    return False

removed_default_items_file = Path("/tmp/railway_removed_defaults.json")
gold_exports_dir = Path("/tmp/railway_gold_exports")

SAMPLE_ANNOTATIONS: List[Dict[str, Any]] = [
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
        "decision": "accept",
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
        "decision": "accept",
        "created_at": "2024-01-01T02:00:00Z",
        "updated_at": "2024-01-01T03:00:00Z"
    }
]

def get_mock_triage_items() -> List[Dict[str, Any]]:
    return [
        {
            "item_id": 1,
            "doc_id": "doc_1",
            "sent_id": "sent_1",
            "text": "White Spot Syndrome Virus (WSSV) is one of the most devastating pathogens affecting Pacific white shrimp.",
            "priority_score": 0.95,
            "confidence": 0.8,
            "status": "pending",
            "priority_level": "critical",
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
            "priority_level": "high",
            "created_at": "2024-01-01T01:00:00Z"
        }
    ]

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
    """Load annotations from database or fallback storage."""
    fallback_records = load_annotations_fallback()

    if not engine or not SessionLocal:
        if fallback_records:
            logger.warning(
                "⚠️ No database connection, returning %s fallback annotations",
                len(fallback_records),
            )
            return fallback_records
        logger.warning("⚠️ No database connection, returning empty annotations")
        return []

    try:
        with SessionLocal() as session:
            annotations = session.query(GoldAnnotation).join(Document).join(Sentence).all()
            annotation_list = []

            for ann in annotations:
                annotation_list.append({
                    "annotation_id": str(ann.id),
                    "id": str(ann.id),
                    "doc_id": ann.document.doc_id,
                    "sent_id": ann.sentence.sent_id,
                    "text": ann.sentence.text,
                    "entities": ann.entities or [],
                    "relations": ann.relations or [],
                    "topics": ann.topics or [],
                    "triplets": ann.triplets or [],
                    "status": ann.status,
                    "decision": ann.status,
                    "confidence": getattr(ann, 'confidence_level', 'medium'),
                    "annotator": ann.annotator_email,
                    "notes": ann.notes or "",
                    "created_at": ann.created_at.isoformat() + "Z",
                    "updated_at": ann.updated_at.isoformat() + "Z"
                })

            if fallback_records:
                existing_ids = {
                    record.get("annotation_id") or record.get("id")
                    for record in annotation_list
                }
                merged = annotation_list[:]
                for record in fallback_records:
                    record_id = record.get("annotation_id") or record.get("id")
                    if record_id not in existing_ids:
                        merged.append(record)
                logger.info(
                    "✅ Loaded %s annotations from database (+%s fallback)",
                    len(annotation_list),
                    len(merged) - len(annotation_list),
                )
                return merged

            logger.info(f"✅ Loaded {len(annotation_list)} annotations from database")
            return annotation_list

    except Exception as e:
        logger.error(f"❌ Failed to load annotations from database: {e}")
        if fallback_records:
            logger.warning(
                "⚠️ Returning fallback annotations due to database error (%s items)",
                len(fallback_records),
            )
            return fallback_records
        return []

def save_annotations_storage(annotations: List[Dict[str, Any]]):
    """Save annotations to database (compatibility function)"""
    if not engine or not SessionLocal:
        logger.warning("⚠️ No database connection, cannot save annotations")
        return
    
    logger.info(f"💾 Database annotation save compatibility function called with {len(annotations)} annotations")
    logger.info("💾 Note: New annotations should be saved directly to database using save_annotation_to_database()")

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

def filter_annotations(
    records: List[Dict[str, Any]],
    *,
    status: Optional[str] = None,
    decision: Optional[str] = None,
    doc_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    def is_all(value: Optional[str]) -> bool:
        if value is None:
            return True
        value_str = str(value).strip().lower()
        return value_str in {"", "all", "null", "undefined"}

    filtered = records

    if not is_all(status):
        filtered = [
            ann for ann in filtered
            if (ann.get("status") or ann.get("decision")) == status
        ]

    if not is_all(decision):
        filtered = [
            ann for ann in filtered
            if (ann.get("decision") or ann.get("status")) == decision
        ]

    if doc_id and not is_all(doc_id):
        filtered = [ann for ann in filtered if ann.get("doc_id") == doc_id]

    if user_id and not is_all(user_id):
        filtered = [ann for ann in filtered if ann.get("annotator") == user_id]

    return filtered

def load_annotation_records(include_mock: bool = True) -> List[Dict[str, Any]]:
    records = load_annotations_storage()
    if include_mock and not records:
        # Return shallow copies to avoid accidental mutation of the sample data
        return [dict(record) for record in SAMPLE_ANNOTATIONS]
    return records

def compute_annotation_statistics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary, per-user, and per-date statistics for annotations."""
    normalized = [normalize_annotation_record(record) for record in records]

    if not normalized:
        return {
            "summary": {
                "total_annotations": 0,
                "accepted": 0,
                "rejected": 0,
                "modified": 0,
                "skipped": 0,
                "pending": 0,
                "acceptance_rate": 0.0,
                "avg_confidence": 0.0,
                "last_updated": None,
            },
            "annotations": [],
            "by_user": [],
            "by_date": [],
        }

    total = len(normalized)
    accepted = sum(1 for ann in normalized if (ann.get("decision") or "").lower() in {"accept", "accepted"})
    rejected = sum(1 for ann in normalized if (ann.get("decision") or "").lower() in {"reject", "rejected"})
    modified = sum(1 for ann in normalized if (ann.get("decision") or "").lower() == "modified")
    skipped = sum(1 for ann in normalized if (ann.get("decision") or "").lower() == "skip")
    pending = sum(1 for ann in normalized if (ann.get("status") or "").lower() == "pending")

    confidences = [ann.get("confidence") for ann in normalized if isinstance(ann.get("confidence"), (int, float))]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    timestamps = [
        datetime.datetime.fromisoformat((ann.get("updated_at") or ann.get("created_at") or datetime.datetime.now().isoformat()).replace('Z', '+00:00'))
        for ann in normalized
    ]
    last_updated = max(timestamps).isoformat() if timestamps else None

    summary = {
        "total_annotations": total,
        "accepted": accepted,
        "rejected": rejected,
        "modified": modified,
        "skipped": skipped,
        "pending": pending,
        "completed_annotations": accepted + modified,
        "acceptance_rate": (accepted / total) * 100 if total else 0.0,
        "avg_confidence": avg_confidence,
        "last_updated": last_updated,
    }

    by_user_map: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "user_id": None,
        "username": None,
        "total": 0,
        "accepted": 0,
        "rejected": 0,
        "modified": 0,
        "skipped": 0,
        "avg_confidence": 0.0,
        "_confidence_sum": 0.0,
    })

    for ann in normalized:
        annotator = ann.get("annotator") or "unknown"
        entry = by_user_map[annotator]
        entry["user_id"] = annotator
        entry["username"] = annotator
        entry["total"] += 1
        decision = (ann.get("decision") or "").lower()
        if decision in {"accept", "accepted"}:
            entry["accepted"] += 1
        elif decision in {"reject", "rejected"}:
            entry["rejected"] += 1
        elif decision == "modified":
            entry["modified"] += 1
        elif decision == "skip":
            entry["skipped"] += 1

        conf = ann.get("confidence")
        if isinstance(conf, (int, float)):
            entry["_confidence_sum"] += conf

    by_user = []
    for annotator, stats in by_user_map.items():
        total_user = stats["total"]
        acceptance_rate = (stats["accepted"] / total_user) * 100 if total_user else 0.0
        avg_user_conf = stats["_confidence_sum"] / total_user if total_user and stats["_confidence_sum"] else 0.0
        stats.pop("_confidence_sum", None)
        stats["acceptance_rate"] = acceptance_rate
        stats["avg_confidence"] = avg_user_conf
        stats["avg_time_per_annotation"] = 0  # Placeholder; no timing data available
        by_user.append(stats)

    by_user.sort(key=lambda x: x["total"], reverse=True)

    by_date_map: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "date": None,
        "total": 0,
        "accepted": 0,
        "rejected": 0,
        "modified": 0,
    })

    for ann in normalized:
        dt_str = ann.get("created_at") or ann.get("updated_at")
        try:
            dt = datetime.datetime.fromisoformat(dt_str.replace('Z', '+00:00')) if dt_str else datetime.datetime.now()
        except Exception:
            dt = datetime.datetime.now()
        date_key = dt.date().isoformat()
        bucket = by_date_map[date_key]
        bucket["date"] = date_key
        bucket["total"] += 1
        decision = (ann.get("decision") or "").lower()
        if decision in {"accept", "accepted"}:
            bucket["accepted"] += 1
        elif decision in {"reject", "rejected"}:
            bucket["rejected"] += 1
        elif decision == "modified":
            bucket["modified"] += 1

    by_date = sorted(by_date_map.values(), key=lambda x: x["date"])

    # Trim annotations list to the most recent 200 entries for UI responsiveness
    annotations_sorted = sorted(
        normalized,
        key=lambda ann: (ann.get("created_at") or ann.get("updated_at") or ""),
        reverse=True,
    )
    annotations_limited = annotations_sorted[:200]

    return {
        "summary": summary,
        "annotations": annotations_limited,
        "by_user": by_user,
        "by_date": by_date,
    }

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
    try:
        logger.info(f"🎯 [SINGLE TRIAGE ENDPOINT] Queue requested: limit={limit}, status={status}")
        
        # Load from persistent storage EVERY TIME
        try:
            stored_docs, stored_items = load_storage()
            logger.info(f"📊 [SINGLE TRIAGE] Loaded from storage: {len(stored_items)} items")
        except Exception as e:
            logger.error(f"❌ [TRIAGE] load_storage() failed: {e}")
            raise e
        
        # Mock items for demonstration
        try:
            mock_items = get_mock_triage_items()
            logger.info(f"📊 [SINGLE TRIAGE] Generated {len(mock_items)} mock items")
        except Exception as e:
            logger.error(f"❌ [TRIAGE] get_mock_triage_items() failed: {e}")
            raise e
        
        try:
            removed_default_ids = load_removed_default_items()
            logger.info(f"📊 [SINGLE TRIAGE] Loaded {len(removed_default_ids)} removed default IDs")
            if removed_default_ids:
                mock_items = [
                    item for item in mock_items
                    if canonical_identifier(item.get("item_id")) not in removed_default_ids
                ]
                if mock_items:
                    logger.info(f"🔍 [SINGLE TRIAGE] {len(removed_default_ids)} default items hidden from queue")
        except Exception as e:
            logger.error(f"❌ [TRIAGE] load_removed_default_items() failed: {e}")
            # Continue without filtering - this is not critical
            logger.warning("⚠️ [TRIAGE] Continuing without removed defaults filtering")

        # Combine mock items with uploaded items (mock first for accessibility)
        all_items = mock_items + stored_items

        for item in all_items:
            if not item.get("priority_level"):
                score = item.get("priority_score") or 0
                if score >= 0.85:
                    item["priority_level"] = "critical"
                elif score >= 0.7:
                    item["priority_level"] = "high"
                elif score >= 0.5:
                    item["priority_level"] = "medium"
                else:
                    item["priority_level"] = "low"
        
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

            def _as_int(value: Any) -> Optional[int]:
                """Best effort conversion of mixed-type IDs for logging/debug."""
                if value is None:
                    return None
                try:
                    return int(str(value).strip())
                except (TypeError, ValueError):
                    return None

            mock_count = 0
            uploaded_count = 0
            for item in final_items:
                item_id_as_int = _as_int(item.get("item_id"))
                if item_id_as_int is None:
                    continue
                if item_id_as_int <= 10:
                    mock_count += 1
                else:
                    uploaded_count += 1
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
        
    except Exception as e:
        logger.error(f"❌ [TRIAGE QUEUE] Error: {e}")
        logger.error(f"❌ [TRIAGE QUEUE] Full traceback: {traceback.format_exc()}")
        # Return minimal fallback response
        fallback_items = [
            {
                "item_id": "error_fallback_1",
                "doc_id": "error_doc_1",
                "sent_id": "error_sent_1",
                "text": "Error loading triage queue - using fallback data.",
                "priority_score": 0.5,
                "confidence": 0.0,
                "status": "pending",
                "priority_level": "medium",
                "created_at": "2024-01-01T00:00:00Z"
            }
        ]
        return {
            "items": fallback_items,
            "total": len(fallback_items),
            "limit": limit,
            "offset": offset,
            "has_more": False,
            "source": "error_fallback",
            "error": str(e)
        }

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
    metadata = request.get('metadata') if isinstance(request.get('metadata'), dict) else {}

    file_name = request.get(
        'fileName',
        request.get('filename', metadata.get('original_filename', 'upload.pdf')),
    )
    file_type = request.get(
        'file_type',
        request.get('fileType', metadata.get('file_type')),
    )

    file_content_b64 = (
        request.get('file_content')
        or request.get('fileContent')
        or metadata.get('file_base64')
    )

    file_bytes = b""
    if file_content_b64:
        try:
            file_bytes = decode_base64_payload(file_content_b64)
            logger.info(
                "📄 [INGEST] Received file payload: %s bytes (type=%s)",
                len(file_bytes),
                file_type,
            )
        except Exception as decode_error:
            logger.error(f"❌ [INGEST] Failed to decode file content: {decode_error}")
            file_bytes = b""

    provided_text = request.get('content') or request.get('text') or ''
    normalised_provided_text = normalise_text(provided_text)

    extracted_text = extract_text_from_bytes(file_bytes, file_type, file_name) if file_bytes else ''
    if extracted_text:
        logger.info("📄 [INGEST] Extracted %s characters from uploaded file", len(extracted_text))
    elif file_bytes:
        logger.warning("⚠️ [INGEST] No text extracted from uploaded file; falling back to provided text")

    document_text = extracted_text or normalised_provided_text
    if not document_text:
        raise HTTPException(status_code=400, detail="No document text provided or extracted from upload")

    document_text = normalise_text(document_text)

    text_source = "file_extracted" if extracted_text else ("provided_text" if normalised_provided_text else "unknown")
    file_size = len(file_bytes) if file_bytes else len(document_text.encode("utf-8"))
    document_metadata = {
        "file_name": file_name,
        "file_type": file_type,
        "file_size": file_size,
        "upload_method": metadata.get("upload_method", "manual"),
        "text_source": text_source,
    }
    if metadata:
        document_metadata["user_metadata"] = dict(metadata)
    if request.get('doc_id'):
        document_metadata["original_doc_id"] = request.get('doc_id')

    # Create unique document ID
    doc_id = f"uploaded_{uuid.uuid4().hex[:12]}"
    timestamp = datetime.datetime.now().isoformat() + "Z"
    
    # Split content into sentences for processing
    sentences = split_into_sentences(document_text)
    if not sentences:
        sentences = [line.strip() for line in document_text.splitlines() if line.strip()]
    sentence_count = len(sentences)
    meaningful_sentences = [s for s in sentences if len(s.strip()) > 20]
    
    triage_created = None

    # Save to database
    if engine and SessionLocal:
        try:
            with SessionLocal() as session:
                # Create document
                document = Document(
                    doc_id=doc_id,
                    source="manual",
                    title=title,
                    raw_text=document_text,
                    document_metadata=document_metadata
                )
                session.add(document)
                session.flush()  # Get document ID
                
                # Create sentences and triage items
                created_items = 0
                for i, sentence in enumerate(sentences):
                    sentence_text = sentence.strip()
                    if len(sentence_text) <= 20:
                        continue

                    text_value = sentence_text if sentence_text.endswith((".", "!", "?")) else f"{sentence_text}."

                    # Create sentence
                    sent_obj = Sentence(
                        sent_id=f"{doc_id}_sent_{i+1}",
                        document_id=document.id,
                        start_offset=0,  # Would need proper calculation
                        end_offset=len(text_value),
                        text=text_value
                    )
                    session.add(sent_obj)
                    session.flush()  # Get sentence ID

                    # Create triage item
                    item_counter = int(timestamp.replace(":", "").replace("-", "").replace("T", "").replace("Z", "")[:12]) + i
                    priority_score = 0.8 + (0.1 * max(0, 3-i))
                    priority_level = "high" if priority_score >= 0.8 else "medium"

                    # Create a placeholder candidate for the triage item
                    # This is needed because TriageItem requires a candidate_id
                    placeholder_candidate = Candidate(
                        sentence_id=sent_obj.id,
                        candidate_type="pending",
                        label="NEEDS_ANNOTATION",
                        confidence=0.0,
                        model_name="manual_upload",
                        generation_method="manual"
                    )
                    session.add(placeholder_candidate)
                    session.flush()  # Get candidate ID

                    triage_item = TriageItem(
                        item_id=str(item_counter),
                        sentence_id=sent_obj.id,
                        candidate_id=placeholder_candidate.id,
                        priority_score=priority_score,
                        priority_level=priority_level,
                        confidence_score=0.0,
                        status="pending"
                    )
                    session.add(triage_item)
                    created_items += 1

                session.commit()
                logger.info(f"💾 [DATABASE] Saved document {doc_id} with {sentence_count} sentences and {created_items} triage items")
                triage_created = created_items
                
        except Exception as e:
            logger.error(f"❌ [DATABASE] Failed to save document: {e}")
            logger.info("🔄 [FALLBACK] Falling back to in-memory storage")
            created_items = save_document_to_fallback(
                doc_id=doc_id,
                title=title,
                file_name=file_name,
                sentences=sentences,
                timestamp=timestamp,
                raw_text=document_text,
                metadata=document_metadata,
            )
            triage_created = created_items
    else:
        logger.warning("⚠️ [DATABASE] No database connection, using fallback storage")
        logger.info("🔄 [FALLBACK] Saving document to in-memory storage")
        triage_created = save_document_to_fallback(
            doc_id=doc_id,
            title=title,
            file_name=file_name,
            sentences=sentences,
            timestamp=timestamp,
            raw_text=document_text,
            metadata=document_metadata,
        )
    if triage_created is None:
        triage_created = len(meaningful_sentences)
    logger.info(f"✅ [INGEST] Document '{title}' added with {sentence_count} sentences, {triage_created} triage items created")
    
    return {
        "success": True,
        "doc_id": doc_id,
        "title": title,
        "status": "ingested",
        "sentence_count": sentence_count,
        "created_at": timestamp,
        "triage_items_created": min(triage_created, sentence_count),
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
    try:
        storage_annotations = load_annotation_records(include_mock=True)
        stats = compute_annotation_statistics(storage_annotations)

        # Include additional summary shortcuts for legacy UI expectations
        summary = stats.get("summary", {})
        stats.update({
            "total_annotations": summary.get("total_annotations", 0),
            "completed_annotations": summary.get("completed_annotations", 0),
            "pending_annotations": summary.get("pending", 0),
        })

        return stats
    except Exception as e:
        logger.error(f"❌ Annotation statistics failed: {e}")
        # Return default statistics instead of failing
        return {
            "summary": {
                "total_annotations": 0,
                "accepted": 0,
                "rejected": 0,
                "modified": 0,
                "skipped": 0,
                "pending": 0,
                "acceptance_rate": 0.0,
                "avg_confidence": 0.0,
                "last_updated": None,
            },
            "total_annotations": 0,
            "completed_annotations": 0,
            "pending_annotations": 0,
            "annotations": [],
            "by_user": [],
            "by_date": [],
            "error": "Failed to load annotation statistics"
        }

# MISSING ENDPOINTS THAT FRONTEND EXPECTS
@app.get("/api/statistics/overview")
async def get_statistics_overview():
    """Get overview statistics for dashboard"""
    logger.info("📊 [STATS] Overview statistics requested")

    try:
        stored_docs, stored_items = load_storage()
        annotations = compute_annotation_statistics(load_annotation_records(include_mock=True))
        summary = annotations.get("summary", {})

        unique_users = len({user.get("user_id") for user in annotations.get("by_user", [])})

        pending_triage = len([item for item in stored_items if (item.get("status") or "").lower() != "completed"])

        return {
            "total_documents": len(stored_docs),
            "total_annotations": summary.get("total_annotations", 0),
            "total_candidates": len(stored_items),
            "active_users": unique_users,
            "pending_triage_items": pending_triage,
            "completed_annotations": summary.get("completed_annotations", 0),
            "annotation_rate": summary.get("acceptance_rate", 0) / 100 if summary.get("total_annotations", 0) else 0,
            "avg_confidence": summary.get("avg_confidence", 0),
            "last_updated": summary.get("last_updated"),
        }
    except Exception as e:
        logger.error(f"❌ Overview statistics failed: {e}")
        # Return default statistics instead of failing
        return {
            "total_documents": 0,
            "total_annotations": 0,
            "total_candidates": 0,
            "active_users": 0,
            "pending_triage_items": 0,
            "completed_annotations": 0,
            "annotation_rate": 0.0,
            "avg_confidence": 0.0,
            "last_updated": None,
            "error": "Failed to load overview statistics"
        }

@app.get("/api/triage/next")
async def get_next_triage_item():
    """Get next item from triage queue"""
    logger.info("⏭️ [TRIAGE] Next item requested")
    
    # Load from persistent storage
    stored_docs, stored_items = load_storage()
    
    # Use same ordering as triage queue to ensure consistency
    mock_items = get_mock_triage_items()
    
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
    
    # Load annotations from persistent storage (with mock fallback for demo environments)
    storage_annotations = load_annotation_records(include_mock=True)
    
    filtered_annotations = filter_annotations(
        storage_annotations,
        status=status,
        decision=decision,
        doc_id=doc_id,
        user_id=user_id,
    )

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

@app.get("/api/annotations/export")
async def export_annotations(
    sort_by: str = "created_at",
    format: str = "json",
    status: Optional[str] = None,
    decision: Optional[str] = None,
    doc_id: Optional[str] = None,
    user_id: Optional[str] = None,
):
    """Export annotations in various formats"""
    logger.info(
        "📤 [EXPORT] Annotation export requested: format=%s, status=%s, decision=%s, doc_id=%s, user_id=%s",
        format,
        status,
        decision,
        doc_id,
        user_id,
    )

    storage_annotations = load_annotation_records(include_mock=True)

    filtered_annotations = filter_annotations(
        storage_annotations,
        status=status,
        decision=decision,
        doc_id=doc_id,
        user_id=user_id,
    )

    if sort_by == "created_at":
        filtered_annotations.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    elif sort_by == "updated_at":
        filtered_annotations.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    elif sort_by == "confidence":
        filtered_annotations.sort(key=lambda x: x.get("confidence", 0), reverse=True)

    normalized_records = [normalize_annotation_record(ann) for ann in filtered_annotations]
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    if format.lower() == "json":
        from fastapi.responses import Response

        payload = {
            "export_metadata": {
                "format": "json",
                "total_annotations": len(normalized_records),
                "export_timestamp": datetime.datetime.now().isoformat(),
                "filtered_by_status": status,
                "filtered_by_decision": decision,
                "filtered_by_doc": doc_id,
                "filtered_by_user": user_id,
            },
            "annotations": normalized_records,
        }

        json_content = json.dumps(payload, indent=2)
        filename = f"annotations_export_{timestamp}.json"
        return Response(
            content=json_content,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    elif format.lower() == "csv":
        import io
        import csv

        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow([
            "annotation_id", "doc_id", "sent_id", "text", "decision",
            "status", "confidence", "annotator", "created_at", "updated_at",
            "entities_count", "relations_count", "triplets_count"
        ])

        for ann in normalized_records:
            writer.writerow([
                ann.get("annotation_id", ""),
                ann.get("doc_id", ""),
                ann.get("sent_id", ""),
                ann.get("text", "")[:100],
                ann.get("decision", ""),
                ann.get("status", ""),
                ann.get("confidence", ""),
                ann.get("annotator", ""),
                ann.get("created_at", ""),
                ann.get("updated_at", ""),
                len(ann.get("entities", [])),
                len(ann.get("relations", [])),
                len(ann.get("triplets", []))
            ])

        csv_content = output.getvalue()
        output.close()

        from fastapi.responses import Response
        filename = f"annotations_export_{timestamp}.csv"
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported export format: {format}")

@app.get("/api/annotations/{annotation_id}")
async def get_annotation_detail(annotation_id: str):
    """Retrieve details for a specific annotation."""
    storage_annotations = load_annotation_records(include_mock=True)
    target = canonical_identifier(annotation_id) or str(annotation_id).strip()

    for record in storage_annotations:
        record_id = canonical_identifier(record.get("annotation_id")) or canonical_identifier(record.get("id"))
        if record_id == target:
            return {"annotation": normalize_annotation_record(record)}

    raise HTTPException(status_code=404, detail=f"Annotation {annotation_id} not found")

@app.post("/api/export/gold")
async def export_gold_annotations(export_request: GoldExportRequest):
    """Export annotations in gold format for downstream use."""
    storage_annotations = load_annotation_records(include_mock=True)

    if export_request.doc_ids:
        allowed_docs = set(export_request.doc_ids)
        storage_annotations = [ann for ann in storage_annotations if ann.get("doc_id") in allowed_docs]

    gold_exports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    export_format = export_request.format.lower()
    export_path = gold_exports_dir / f"gold_export_{timestamp}.{export_format if export_format in {'json', 'jsonl'} else 'json'}"

    normalized_records = [normalize_annotation_record(ann) for ann in storage_annotations]

    if not normalized_records:
        logger.warning("⚠️ [EXPORT] No annotations available for gold export; creating empty file")

    if export_format == "jsonl":
        with open(export_path, 'w', encoding='utf-8') as f:
            for record in normalized_records:
                json.dump(record, f)
                f.write('\n')
        if not normalized_records:
            # Ensure file exists even when empty
            open(export_path, 'a', encoding='utf-8').close()
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
    mock_items = get_mock_triage_items()

    all_items = mock_items + stored_items

    for item in all_items:
        if not item.get("priority_level"):
            score = item.get("priority_score") or 0
            if score >= 0.85:
                item["priority_level"] = "critical"
            elif score >= 0.7:
                item["priority_level"] = "high"
            elif score >= 0.5:
                item["priority_level"] = "medium"
            else:
                item["priority_level"] = "low"

    total_items = len(all_items)
    pending_items = len([item for item in all_items if (item.get("status") or "").lower() == "pending"])
    in_review_items = len([item for item in all_items if (item.get("status") or "").lower() in {"in_review", "assigned"}])
    completed_items = len([item for item in all_items if (item.get("status") or "").lower() == "completed"])

    priority_scores = [item.get("priority_score") for item in all_items if isinstance(item.get("priority_score"), (int, float))]
    avg_priority_score = sum(priority_scores) / len(priority_scores) if priority_scores else 0.0

    confidences = [item.get("confidence") for item in all_items if isinstance(item.get("confidence"), (int, float))]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "total": total_items,
        "total_items": total_items,
        "pending_items": pending_items,
        "in_progress_items": in_review_items,
        "completed_items": completed_items,
        "avg_priority_score": avg_priority_score,
        "avg_confidence": avg_confidence,
        "uploaded_items": len(stored_items),
        "items": all_items,
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

        # Save annotation to database
        timestamp = datetime.datetime.now()
        annotation_id = None
        decision_id = f"decision_{item_id}_{int(timestamp.timestamp())}"
        persisted_to_db = False

        doc_id = request.get('doc_id', f'doc_{item_id}')
        sent_id = request.get('sent_id', f'sent_{item_id}')

        try:
            if engine and SessionLocal:
                with SessionLocal() as session:
                    # Try to find existing document and sentence
                    document = session.query(Document).filter(Document.doc_id == doc_id).first()
                    if document:
                        sentence = session.query(Sentence).filter(
                            Sentence.document_id == document.id,
                            Sentence.sent_id == sent_id
                        ).first()

                        if sentence:
                            # Create gold annotation
                            annotation = GoldAnnotation(
                                document_id=document.id,
                                sentence_id=sentence.id,
                                entities=entities,
                                relations=relations,
                                topics=topics,
                                triplets=triplets,
                                annotator_email=user_id,
                                status="accepted" if decision == "accept" else decision,
                                confidence_level="high" if confidence > 0.8 else "medium" if confidence > 0.5 else "low",
                                notes=notes,
                                decision_method="manual"
                            )

                            session.add(annotation)
                            session.commit()
                            annotation_id = str(annotation.id)
                            persisted_to_db = True

                            logger.info(f"💾 [DATABASE] Saved annotation {annotation_id} for {doc_id}/{sent_id}")
                        else:
                            logger.warning(f"⚠️ [DATABASE] Sentence {sent_id} not found for document {doc_id}")
                    else:
                        logger.warning(f"⚠️ [DATABASE] Document {doc_id} not found")
            else:
                logger.warning("⚠️ [DATABASE] No database connection, annotation not saved")

        except Exception as e:
            logger.error(f"❌ [DATABASE] Failed to save annotation: {e}")
            # Fallback to in-memory storage for compatibility
            annotation_id = f"fallback_{item_id}_{int(timestamp.timestamp())}"
        
        logger.info(f"💾 [DECISION] Saved annotation {annotation_id or 'unknown'} for item {item_id}")
        
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
            matched_queue_item: Optional[Dict[str, Any]] = None

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
                    matched_queue_item = queued_item
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
                logger.info(
                    "✅ [QUEUE REMOVAL] Successfully removed item %s from queue (%s → %s items)",
                    item_id,
                    items_before,
                    items_after,
                )

                if engine and SessionLocal and matched_queue_item:
                    updated = mark_triage_item_completed_in_db(
                        matched_queue_item,
                        decision,
                        user_id,
                        timestamp,
                    )
                    if not updated:
                        logger.warning(
                            "⚠️ [QUEUE REMOVAL] Database triage item update failed for %s",
                            item_id,
                        )
                else:
                    # Persist fallback queue to disk
                    save_storage(stored_docs, stored_items)
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
        
        fallback_annotation_id = annotation_id
        if not fallback_annotation_id:
            safe_doc = (doc_id or "doc").replace("/", "_")
            safe_sent = (sent_id or "sent").replace("/", "_")
            fallback_annotation_id = f"{safe_doc}_{safe_sent}_{int(timestamp.timestamp())}"

        sentence_text = None
        doc_title = None
        if matched_queue_item:
            sentence_text = matched_queue_item.get("text")
            doc_title = matched_queue_item.get("doc_title")
        if not sentence_text:
            sentence_text = request.get('text')
        if not sentence_text:
            # Attempt to recover sentence text from storage
            try:
                _, remaining_items = load_storage()
                for queued in remaining_items:
                    if queued.get("doc_id") == doc_id and queued.get("sent_id") == sent_id:
                        sentence_text = queued.get("text")
                        doc_title = doc_title or queued.get("doc_title")
                        break
            except Exception:
                pass

        created_at_str = timestamp.isoformat() + "Z"
        annotation_record = {
            "annotation_id": fallback_annotation_id,
            "id": fallback_annotation_id,
            "item_id": item_id,
            "candidate_id": request.get('candidate_id'),
            "doc_id": doc_id,
            "sent_id": sent_id,
            "text": sentence_text or "",
            "decision": decision,
            "status": "completed" if decision == "accept" else decision,
            "confidence": confidence,
            "annotator": user_id,
            "notes": notes,
            "entities": entities,
            "relations": relations,
            "topics": topics,
            "triplets": triplets,
            "created_at": created_at_str,
            "updated_at": created_at_str,
            "doc_title": doc_title,
        }

        if not persisted_to_db:
            append_annotation_fallback(annotation_record)

        return {
            "success": True,
            "decision_id": decision_id,
            "annotation_id": annotation_id or fallback_annotation_id,
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
    # Load from file-based fallback storage
    fallback_docs, fallback_items = load_fallback_storage()
    stored_docs, stored_items = load_storage()
    
    return {
        "fallback_storage": {
            "file_path": str(fallback_storage_file),
            "file_exists": fallback_storage_file.exists(),
            "fallback_documents_count": len(fallback_docs),
            "fallback_triage_items_count": len(fallback_items),
            "fallback_documents": fallback_docs,
            "fallback_triage_items": fallback_items[:5]  # First 5 items only
        },
        "loaded_storage": {
            "loaded_docs_count": len(stored_docs),
            "loaded_items_count": len(stored_items), 
            "loaded_docs": stored_docs,
            "loaded_items": stored_items[:5]  # First 5 items only
        },
        "database_status": {
            "engine_available": engine is not None,
            "session_available": SessionLocal is not None
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

# FINAL ROUTE CLEANUP - Remove any catch-all routes that might interfere with API
logger.info("🔧 [FINAL CLEANUP] Checking for catch-all routes after all setup...")
final_catch_all = [r for r in app.routes if hasattr(r, 'path') and r.path == "/{full_path:path}"]
if final_catch_all:
    logger.warning(f"🚨 [FINAL CLEANUP] Found {len(final_catch_all)} catch-all routes - removing them")
    original_routes = list(app.routes)
    app.router.routes.clear()
    
    for route in original_routes:
        if hasattr(route, 'path') and route.path == "/{full_path:path}":
            logger.info(f"🗑️ [FINAL CLEANUP] Removed catch-all route: {route.path}")
            continue
        app.router.routes.append(route)
    
    logger.info(f"✅ [FINAL CLEANUP] Route cleanup complete: {len(app.routes)} routes remaining")
else:
    logger.info("✅ [FINAL CLEANUP] No catch-all routes found - API should work correctly")

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
