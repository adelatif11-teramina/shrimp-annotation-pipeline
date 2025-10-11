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
from services.candidates.triplet_workflow import TripletWorkflow
from services.rules.rule_based_annotator import ShimpAquacultureRuleEngine
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
        model="gpt-4o",
        api_key=api_key,
        temperature=0.1,
        cache_dir=pipeline_root / "data/local/llm_cache"
    )

    rule_engine = ShimpAquacultureRuleEngine()
    workflow = TripletWorkflow(llm_generator, rule_engine)
    
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
    total_triplets = 0
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
            
            # Process all sentences
            sentences_to_process = document.sentences
            print(f"   🎯 Processing {len(sentences_to_process)} sentences...")
            
            doc_candidates = []
            
            for i, sentence in enumerate(sentences_to_process):
                try:
                    print(f"     Sentence {i+1:2d}: {sentence.text[:60]}...")
                    
                    # Run the full triplet workflow for this sentence
                    workflow_result = await workflow.process_sentence(
                        document.doc_id,
                        sentence.sent_id,
                        sentence.text,
                        title=document.title
                    )

                    total_sentences_processed += 1
                    total_entities += len(workflow_result.entities)
                    total_triplets += len(workflow_result.triplets)

                    if workflow_result.triplets:
                        triplet_summary = [
                            f"{item.head['text']} -{item.relation}-> {item.tail['text']}"
                            for item in workflow_result.triplets[:2]
                        ]
                        print(
                            f"                ✅ {len(workflow_result.triplets)} triplets | "
                            f"Audit: {workflow_result.audit_overall_verdict.upper()} | "
                            f"Examples: {', '.join(triplet_summary)}"
                        )
                    else:
                        print("                ⭕ No triplets generated")

                    doc_candidates.append(workflow_result.to_dict())

                    # Small delay to be respectful to API
                    await asyncio.sleep(0.1)
                        
                except Exception as e:
                    print(f"                ❌ Error: {e}")
            
            # Save candidates
            if doc_candidates:
                output_file = candidates_dir / f"{document.doc_id}_triplet_candidates.jsonl"
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
    print(f"   🔺 Triplets: {total_triplets} LLM triplet candidates")
    print(f"   💾 Output: {candidates_dir}")

    if total_sentences_processed:
        avg_triplets = total_triplets / total_sentences_processed
        print(f"   📈 Average: {avg_triplets:.1f} triplets per sentence")

    print(f"\n✅ High-quality OpenAI triplet candidates ready for annotation!")
    print(f"💡 Cost estimate: ~$0.01-0.05 for this processing")
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
