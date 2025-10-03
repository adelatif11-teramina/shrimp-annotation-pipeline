#!/usr/bin/env python3
"""
Populate Triage Queue Script

Processes uploaded documents to generate annotation candidates and populate the triage queue.
This script bridges the gap between document ingestion and annotation workflow.
"""

import sys
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

async def populate_triage_queue():
    """Process all uploaded documents and populate triage queue"""
    
    logger.info("🚀 Starting triage queue population...")
    
    # Initialize services
    logger.info("Initializing services...")
    ingestion_service = DocumentIngestionService()
    
    # Initialize rule engine (always available)
    rule_engine = ShimpAquacultureRuleEngine()
    
    # Initialize triage engine
    triage_engine = TriagePrioritizationEngine(
        gold_store_path=project_root / "data/gold"
    )
    
    # LLM generator is optional (requires API key)
    llm_generator = None
    try:
        llm_generator = LLMCandidateGenerator(
            provider="openai",
            model="gpt-4o-mini",
            cache_dir=project_root / "data/candidates/.cache"
        )
        logger.info("✓ LLM candidate generator initialized")
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
            
            logger.info(f"  Document has {len(document.sentences)} sentences")
            
            # Process each sentence
            for i, sentence in enumerate(document.sentences):
                sentence_data = {
                    "doc_id": document.doc_id,
                    "sent_id": sentence.sent_id,
                    "text": sentence.text,
                    "title": document.title
                }
                
                try:
                    # Generate LLM candidates if available
                    llm_candidates = None
                    if llm_generator:
                        try:
                            llm_result = await llm_generator.process_sentence(
                                sentence_data["doc_id"],
                                sentence_data["sent_id"], 
                                sentence_data["text"],
                                sentence_data["title"]
                            )
                            llm_candidates = llm_result["candidates"]
                        except Exception as e:
                            logger.warning(f"LLM processing failed for sentence {sentence.sent_id}: {e}")
                    
                    # Generate rule-based candidates
                    rule_result = rule_engine.process_sentence(
                        sentence_data["doc_id"],
                        sentence_data["sent_id"],
                        sentence_data["text"]
                    )
                    
                    # Only add to triage if we have interesting content
                    # (either LLM found something or rule engine found entities)
                    has_entities = False
                    if llm_candidates:
                        has_entities = bool(llm_candidates.get("entities", []))
                    if not has_entities and rule_result.get("rule_results"):
                        has_entities = bool(rule_result["rule_results"].get("entities", []))
                    
                    # Add to triage queue (even if no entities, for demonstration)
                    if True:  # Change to `if has_entities:` to only queue sentences with entities
                        doc_metadata = {
                            "doc_id": sentence_data["doc_id"],
                            "sent_id": sentence_data["sent_id"],
                            "title": document.title,
                            "source": "uploaded"
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
                        
                        # Add sentence text to candidates
                        combined_candidates["text"] = sentence_data["text"]
                        
                        triage_engine.add_candidates(
                            combined_candidates,
                            doc_metadata,
                            rule_result.get("rule_results")
                        )
                        
                        total_candidates_generated += 1
                        
                        if total_candidates_generated % 10 == 0:
                            logger.info(f"  Generated {total_candidates_generated} candidates so far...")
                
                except Exception as e:
                    logger.error(f"Failed to process sentence {sentence.sent_id}: {e}")
                    continue
                
                # Limit processing for demo (remove this for full processing)
                if total_candidates_generated >= 50:  # Process first 50 sentences for demo
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
    asyncio.run(populate_triage_queue())