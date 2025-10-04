#!/usr/bin/env python3
"""
Populate Triage Queue Script

Processes uploaded documents to generate annotation candidates and populate the triage queue.
This script bridges the gap between document ingestion and annotation workflow.
"""

import sys
import os
import logging
from pathlib import Path
import asyncio

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from services.ingestion.document_ingestion import DocumentIngestionService
from services.candidates.llm_candidate_generator import LLMCandidateGenerator
from services.triage.triage_prioritization import TriagePrioritizationEngine
from services.rules.rule_based_annotator import ShimpAquacultureRuleEngine

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

async def populate_triage_queue(chunking_mode: str = "sentence"):
    """Process all uploaded documents and populate triage queue"""
    
    logger.info(f"🚀 Starting triage queue population (chunking_mode: {chunking_mode})...")
    
    # Initialize services
    logger.info("Initializing services...")
    ingestion_service = DocumentIngestionService(chunking_mode=chunking_mode)
    
    # Initialize rule engine (always available)
    rule_engine = ShimpAquacultureRuleEngine()
    
    # Initialize triage engine
    triage_engine = TriagePrioritizationEngine(
        gold_store_path=project_root / "data/gold"
    )
    
    # LLM generator is optional (requires API key)
    llm_generator = None
    api_key = os.getenv('OPENAI_API_KEY')
    try:
        if api_key:
            llm_generator = LLMCandidateGenerator(
                provider="openai",
                model="gpt-4o-mini",
                api_key=api_key,
                cache_dir=project_root / "data/candidates/.cache"
            )
            logger.info("✓ LLM candidate generator initialized with OpenAI")
        else:
            logger.warning("OPENAI_API_KEY not found in environment")
            raise ValueError("OPENAI_API_KEY required")
    except Exception as e:
        logger.warning(f"LLM generator not available: {e}")
        logger.info("Will use rule-based annotation only")
    
    # Find all raw documents
    raw_dir = project_root / "data/raw"
    if not raw_dir.exists():
        logger.error("No raw documents directory found")
        return
    
    doc_files = list(raw_dir.glob("*.txt"))
    if not doc_files:
        logger.error("No documents found in data/raw/")
        return
    
    logger.info(f"Found {len(doc_files)} documents to process")
    
    total_candidates_generated = 0
    
    # Process each document
    for doc_file in doc_files:
        logger.info(f"Processing document: {doc_file.name}")
        
        try:
            # Ingest document to get sentences
            document = ingestion_service.ingest_text_file(
                doc_file,
                source="uploaded",
                title=doc_file.stem
            )
            
            # Get chunks based on chunking mode
            chunks = document.chunks if hasattr(document, 'chunks') else document.sentences
            chunk_type = getattr(document, 'chunk_type', 'sentence')
            
            logger.info(f"  Document has {len(chunks)} {chunk_type}s")
            
            # Process each chunk (sentence or paragraph)
            for i, chunk in enumerate(chunks):
                # Handle both sentence and paragraph chunks
                if hasattr(chunk, 'sent_id'):  # Sentence
                    chunk_data = {
                        "doc_id": document.doc_id,
                        "sent_id": chunk.sent_id,
                        "text": chunk.text,
                        "title": document.title
                    }
                else:  # Paragraph
                    chunk_data = {
                        "doc_id": document.doc_id,
                        "sent_id": chunk.para_id,  # Use para_id as sent_id for compatibility
                        "text": chunk.text,
                        "title": document.title
                    }
                
                try:
                    # Generate LLM candidates if available
                    llm_candidates = None
                    if llm_generator:
                        try:
                            llm_result = await llm_generator.process_sentence(
                                chunk_data["doc_id"],
                                chunk_data["sent_id"], 
                                chunk_data["text"],
                                chunk_data["title"]
                            )
                            llm_candidates = llm_result["candidates"]
                        except Exception as e:
                            logger.warning(f"LLM processing failed for {chunk_type} {chunk_data['sent_id']}: {e}")
                    
                    # Generate rule-based candidates
                    rule_result = rule_engine.process_sentence(
                        chunk_data["doc_id"],
                        chunk_data["sent_id"],
                        chunk_data["text"]
                    )
                    
                    # Only add to triage if we have interesting content
                    # (either LLM found something or rule engine found entities)
                    has_entities = False
                    if llm_candidates:
                        has_entities = bool(llm_candidates.get("entities", []))
                    if not has_entities and rule_result.get("rule_results"):
                        has_entities = bool(rule_result["rule_results"].get("entities", []))
                    
                    # Add to triage queue (even if no entities, for demonstration)
                    if True:  # Change to `if has_entities:` to only queue chunks with entities
                        doc_metadata = {
                            "doc_id": chunk_data["doc_id"],
                            "sent_id": chunk_data["sent_id"],  # Keep as sent_id for API compatibility
                            "title": document.title,
                            "source": "uploaded",
                            "chunk_type": chunk_type
                        }
                        
                        # Combine candidates
                        combined_candidates = llm_candidates or {}
                        if rule_result.get("rule_results"):
                            # Merge rule results
                            if "entities" not in combined_candidates:
                                combined_candidates["entities"] = []
                            combined_candidates["entities"].extend(
                                rule_result["rule_results"].get("entities", [])
                            )
                        
                        # Add chunk text to candidates
                        combined_candidates["text"] = chunk_data["text"]
                        
                        triage_engine.add_candidates(
                            combined_candidates,
                            doc_metadata,
                            rule_result.get("rule_results")
                        )
                        
                        total_candidates_generated += 1
                        
                        if total_candidates_generated % 10 == 0:
                            logger.info(f"  Generated {total_candidates_generated} candidates so far...")
                
                except Exception as e:
                    logger.error(f"Failed to process {chunk_type} {chunk_data.get('sent_id', 'unknown')}: {e}")
                    continue
                
                # Limit processing for demo (remove this for full processing)
                if total_candidates_generated >= 50:  # Process first 50 chunks for demo
                    logger.info("Demo limit reached (50 candidates). Remove this limit for full processing.")
                    break
            
            if total_candidates_generated >= 50:
                break
                
        except Exception as e:
            logger.error(f"Failed to process document {doc_file.name}: {e}")
            continue
    
    # Print summary
    queue_stats = triage_engine.get_queue_statistics()
    logger.info("🎉 Triage queue population complete!")
    logger.info(f"Total candidates generated: {total_candidates_generated}")
    logger.info(f"Queue statistics: {queue_stats}")
    
    return triage_engine

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Populate triage queue with document candidates")
    parser.add_argument("--chunking-mode", default="sentence", 
                       choices=["sentence", "paragraph"],
                       help="Text chunking mode (default: sentence)")
    
    args = parser.parse_args()
    
    asyncio.run(populate_triage_queue(chunking_mode=args.chunking_mode))