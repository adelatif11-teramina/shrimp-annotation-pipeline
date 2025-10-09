#!/usr/bin/env python3
"""
Process all sentences from the 3 loaded documents and generate candidates.
"""

import sys
import os
import json
import asyncio
import requests
from pathlib import Path
from dotenv import load_dotenv

# Add pipeline root to sys.path
pipeline_root = Path(__file__).parent
sys.path.append(str(pipeline_root))

# Load environment variables from .env file
load_dotenv()

from services.candidates.llm_candidate_generator import LLMCandidateGenerator
from services.ingestion.document_ingestion import DocumentIngestionService

async def process_loaded_documents():
    """Process all sentences from the 3 loaded documents"""
    
    print("🚀 Processing All Loaded Documents for Annotation\n")
    
    # Initialize services
    ingestion_service = DocumentIngestionService(segmenter="regex")
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        print("Please set it in your .env file or environment")
        return
    
    llm_generator = LLMCandidateGenerator(
        provider="openai",
        model="gpt-4o", 
        api_key=api_key,
        temperature=0.1,
        cache_dir=pipeline_root / "data/local/llm_cache"
    )
    
    # Get the 3 loaded documents
    raw_dir = pipeline_root / "data/raw"
    loaded_docs = ["test1.txt", "test2.txt", "noaa.txt"]
    
    api_base = "http://localhost:8000"
    auth_token = "local-admin-2024"
    
    total_candidates = 0
    
    for filename in loaded_docs:
        doc_file = raw_dir / filename
        if not doc_file.exists():
            continue
            
        print(f"📄 Processing: {filename}")
        
        try:
            # Ingest document
            document = ingestion_service.ingest_text_file(
                file_path=doc_file,
                source="data-training", 
                title=doc_file.stem
            )
            
            # Process first 50 sentences (to manage API costs)
            sentences_to_process = document.sentences[:50]
            print(f"   Processing {len(sentences_to_process)} sentences...")
            
            doc_candidates = []
            
            for i, sentence in enumerate(sentences_to_process):
                try:
                    if i % 10 == 0:
                        print(f"     Progress: {i+1}/{len(sentences_to_process)}")
                    
                    # Generate entities with OpenAI
                    entities = await llm_generator.extract_entities(sentence.text)
                    
                    if entities:
                        # Create candidate item for API
                        candidate_data = {
                            "doc_id": document.doc_id,
                            "sent_id": sentence.sent_id,
                            "text": sentence.text,
                            "entities": [
                                {
                                    "text": e.text,
                                    "label": e.label,
                                    "start": e.start,
                                    "end": e.end,
                                    "confidence": e.confidence
                                }
                                for e in entities
                            ],
                            "priority_score": min(0.9, sum(e.confidence for e in entities) / len(entities)),
                            "source": "openai_production"
                        }
                        
                        doc_candidates.append(candidate_data)
                        total_candidates += len(entities)
                    
                    # Small delay to respect API limits
                    await asyncio.sleep(0.05)
                    
                except Exception as e:
                    print(f"     Error on sentence {i}: {e}")
            
            print(f"   ✅ Generated {len(doc_candidates)} annotation items")
            
            # Save candidates to file for queue loading
            output_file = pipeline_root / f"data/candidates/{document.doc_id}_production.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for candidate in doc_candidates:
                    f.write(json.dumps(candidate, ensure_ascii=False) + '\n')
            
            print(f"   💾 Saved to: {output_file.name}")
            
        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")
    
    print(f"\n🎯 Processing Complete!")
    print(f"   📊 Total candidates generated: {total_candidates}")
    print(f"   📁 Files saved to: data/candidates/")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Click ▶️ buttons in the UI to start annotating")
    print(f"   2. Accept/reject/modify the OpenAI suggestions")
    print(f"   3. Export training data when done")
    
    return total_candidates

if __name__ == "__main__":
    total = asyncio.run(process_loaded_documents())
    print(f"\n✅ Ready to annotate {total} entities!")