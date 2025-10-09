#!/usr/bin/env python3
"""
Start production annotation workflow.
Load documents, generate candidates, and populate the annotation queue.
"""

import sys
import os
import json
import asyncio
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Add pipeline root to sys.path
pipeline_root = Path(__file__).parent
sys.path.append(str(pipeline_root))

# Load environment variables from .env file
load_dotenv()

from services.candidates.llm_candidate_generator import LLMCandidateGenerator
from services.ingestion.document_ingestion import DocumentIngestionService

def load_documents_via_api():
    """Load documents into the annotation system via API"""
    print("📚 Loading documents into annotation system...")
    
    api_base = "http://localhost:8000"
    auth_token = "local-admin-2024"
    
    # Get priority files to load
    raw_dir = pipeline_root / "data/raw"
    priority_files = [
        "test1.txt",      # Research paper on TPD
        "test2.txt",      # Another research paper  
        "noaa.txt",       # NOAA disease guide
    ]
    
    loaded_count = 0
    for filename in priority_files:
        doc_file = raw_dir / filename
        if not doc_file.exists():
            print(f"⚠️  Skipping {filename} - file not found")
            continue
            
        try:
            print(f"   Loading: {filename}")
            
            # Read document
            with open(doc_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Create document payload
            doc_data = {
                "title": doc_file.stem,
                "text": text[:10000],  # First 10k chars to avoid timeout
                "source": "data-training"
            }
            
            # Send to API
            response = requests.post(
                f"{api_base}/documents",
                json=doc_data,
                headers={"Authorization": f"Bearer {auth_token}"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Loaded successfully")
                loaded_count += 1
            else:
                print(f"   ❌ Failed: {response.status_code} - {response.text[:100]}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"📊 Loaded {loaded_count} documents")
    return loaded_count

async def generate_candidates_for_queue():
    """Generate annotation candidates and populate the queue"""
    print("🤖 Generating annotation candidates...")
    
    # Initialize LLM generator
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
    
    # Process sample sentences from candidate files
    candidates_dir = pipeline_root / "data/candidates"
    sample_sentences = [
        "Vibrio parahaemolyticus causes AHPND in Penaeus vannamei post-larvae at 28°C.",
        "Treatment with florfenicol at 10 mg/kg improved survival rate in infected shrimp.",
        "The PvIGF gene was associated with growth rate and disease resistance.",
        "Post-larvae stocked at 15 PL/m² showed better FCR than higher densities.",
        "White spot syndrome virus (WSSV) infection resulted in 80% mortality."
    ]
    
    queue_items = []
    
    for i, sentence in enumerate(sample_sentences):
        try:
            print(f"   Processing sentence {i+1}: {sentence[:50]}...")
            
            # Generate entities with OpenAI
            entities = await llm_generator.extract_entities(sentence)
            
            if entities:
                queue_item = {
                    "doc_id": f"sample_{i}",
                    "sent_id": f"s_{i}",
                    "text": sentence,
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
                    "priority_score": 0.8,
                    "source": "openai_gpt4o_mini"
                }
                
                queue_items.append(queue_item)
                print(f"      ✅ {len(entities)} entities: {', '.join([e.text for e in entities[:3]])}")
            
            # Small delay
            await asyncio.sleep(0.1)
            
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    # Save queue items
    queue_file = pipeline_root / "data/local/queue/annotation_queue.jsonl"
    queue_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(queue_file, 'w', encoding='utf-8') as f:
        for item in queue_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"📊 Generated {len(queue_items)} queue items")
    return len(queue_items)

def main():
    """Start the production annotation workflow"""
    
    print("🚀 Starting Production Annotation Pipeline\n")
    
    # Check services
    print("🔧 Service Status:")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ API Server: Running (port 8000)")
        else:
            print("   ❌ API Server: Error")
            return 1
    except:
        print("   ❌ API Server: Not running")
        return 1
    
    try:
        response = requests.get("http://localhost:3010", timeout=5)
        if response.status_code == 200:
            print("   ✅ React UI: Running (port 3010)")
        else:
            print("   ❌ React UI: Error")
    except:
        print("   ❌ React UI: Not running")
    
    print()
    
    # Load documents
    docs_loaded = load_documents_via_api()
    
    print()
    
    # Generate candidates
    candidates_generated = asyncio.run(generate_candidates_for_queue())
    
    print(f"\n🎯 Production Setup Complete!")
    print(f"   📚 Documents loaded: {docs_loaded}")
    print(f"   🏷️  Candidates generated: {candidates_generated}")
    print(f"   🌐 Web UI: http://localhost:3010")
    print(f"   🔗 API: http://localhost:8000")
    
    print(f"\n✅ Ready for annotation!")
    print(f"💡 Open http://localhost:3010 to start annotating")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)