#!/usr/bin/env python3
"""
Generate high-quality annotation candidates using OpenAI GPT-4o-mini.
This script processes the imported documents and creates professional-grade candidates.
"""

import sys
import os
import json
import asyncio
from pathlib import Path
from datetime import datetime

# Add pipeline root to sys.path
pipeline_root = Path(__file__).parent
sys.path.append(str(pipeline_root))

from services.candidates.llm_candidate_generator import LLMCandidateGenerator
from services.ingestion.document_ingestion import DocumentIngestionService

async def main():
    """Generate OpenAI candidates for imported documents"""
    
    print("🚀 Generating High-Quality Candidates with OpenAI GPT-4o-mini\n")
    
    # Get API key from environment
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        return 1
    
    # Initialize services
    print("Initializing services...")
    ingestion_service = DocumentIngestionService(segmenter="regex")
    llm_generator = LLMCandidateGenerator(
        provider="openai",
        model="gpt-4o-mini",
        api_key=api_key,
        temperature=0.1,
        cache_dir=pipeline_root / "data/local/llm_cache"
    )
    
    print("✅ OpenAI GPT-4o-mini generator ready")
    
    # Select documents to process
    raw_dir = pipeline_root / "data/raw"
    priority_files = [
        "test1.txt",      # Research paper on TPD
        "test2.txt",      # Another research paper  
        "noaa.txt",       # NOAA disease guide
        "pathogens-09-00741.txt",  # Pathogen research
        "boyd1.txt"       # Aquaculture textbook
    ]
    
    candidates_dir = pipeline_root / "data/candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    
    total_entities = 0
    total_sentences_processed = 0
    
    for filename in priority_files:
        doc_file = raw_dir / filename
        if not doc_file.exists():
            print(f"⚠️  Skipping {filename} - file not found")
            continue
            
        print(f"\n📄 Processing: {filename}")
        
        try:
            # Ingest document
            document = ingestion_service.ingest_text_file(
                file_path=doc_file,
                source="data-training",
                title=doc_file.stem
            )
            
            print(f"   📊 {len(document.sentences)} sentences total")
            
            # Process sentences in batches (to manage API costs)
            sentences_to_process = document.sentences[:20]  # First 20 sentences
            print(f"   🎯 Processing first {len(sentences_to_process)} sentences...")
            
            doc_candidates = []
            
            for i, sentence in enumerate(sentences_to_process):
                try:
                    print(f"     Sentence {i+1:2d}: {sentence.text[:60]}...")
                    
                    # Extract entities using OpenAI
                    entities = await llm_generator.extract_entities(sentence.text)
                    
                    if entities:
                        candidate = {
                            "doc_id": document.doc_id,
                            "sent_id": sentence.sent_id,
                            "text": sentence.text,
                            "start": sentence.start,
                            "end": sentence.end,
                            "entities": [
                                {
                                    "text": entity.text,
                                    "label": entity.label,
                                    "start": entity.start,
                                    "end": entity.end,
                                    "confidence": entity.confidence
                                }
                                for entity in entities
                            ],
                            "relations": [],  # Can add relation extraction later
                            "topics": [],    # Can add topic classification later
                            "source": "openai_gpt4o_mini",
                            "model": "gpt-4o-mini",
                            "created_at": datetime.now().isoformat()
                        }
                        
                        doc_candidates.append(candidate)
                        total_entities += len(entities)
                        
                        # Show what was found
                        entity_summary = [f"{e.text}→{e.label}" for e in entities[:3]]
                        print(f"                ✅ {len(entities)} entities: {', '.join(entity_summary)}")
                    else:
                        print(f"                ⭕ No entities found")
                        
                    total_sentences_processed += 1
                    
                    # Small delay to be respectful to API
                    await asyncio.sleep(0.1)
                        
                except Exception as e:
                    print(f"                ❌ Error: {e}")
            
            # Save candidates
            if doc_candidates:
                output_file = candidates_dir / f"{document.doc_id}_openai_candidates.jsonl"
                with open(output_file, 'w', encoding='utf-8') as f:
                    for candidate in doc_candidates:
                        f.write(json.dumps(candidate, ensure_ascii=False) + '\n')
                
                print(f"   💾 Saved {len(doc_candidates)} candidates to: {output_file.name}")
            else:
                print(f"   ⚠️  No candidates generated for this document")
                
        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")
    
    print(f"\n🎯 Generation Complete!")
    print(f"   📊 Processed: {total_sentences_processed} sentences")
    print(f"   🏷️  Generated: {total_entities} entity candidates")
    print(f"   💾 Output: {candidates_dir}")
    
    # Quality summary
    if total_entities > 0:
        avg_entities_per_sentence = total_entities / total_sentences_processed
        print(f"   📈 Average: {avg_entities_per_sentence:.1f} entities per sentence")
    
    print(f"\n✅ High-quality OpenAI candidates ready for annotation!")
    print(f"💡 Cost estimate: ~$0.01-0.05 for this processing")
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)