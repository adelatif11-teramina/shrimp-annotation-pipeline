#!/usr/bin/env python3
"""
Clear dummy data and load real sentences from imported documents into the annotation queue.
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

def clear_dummy_data():
    """Clear the dummy/mock data from database"""
    print("🧹 Clearing dummy data from database...")
    
    try:
        # Connect to database and clear mock data
        from services.api.db_sqlite import SimpleDatabase
        db = SimpleDatabase()
        
        # Clear mock data tables
        with db._get_connection() as conn:
            # Clear triage queue
            conn.execute("DELETE FROM triage_queue WHERE candidate_id >= 100")
            # Clear candidates  
            conn.execute("DELETE FROM candidates WHERE id >= 100")
            # Clear mock sentences
            conn.execute("DELETE FROM sentences WHERE doc_id LIKE 'doc_%'")
            # Clear mock documents
            conn.execute("DELETE FROM documents WHERE doc_id LIKE 'doc_%'")
            conn.commit()
            
        print("   ✅ Dummy data cleared")
        return True
        
    except Exception as e:
        print(f"   ❌ Error clearing data: {e}")
        return False

async def load_real_sentences():
    """Load real sentences from imported documents"""
    print("📚 Loading real sentences from imported documents...")
    
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
    
    # Get real document files
    raw_dir = pipeline_root / "data/raw"
    real_docs = {
        "test1.txt": "TPD Research Paper",
        "test2.txt": "Aquaculture Research", 
        "noaa.txt": "NOAA Disease Guide"
    }
    
    try:
        from services.api.db_sqlite import SimpleDatabase
        db = SimpleDatabase()
        
        total_items = 0
        
        for filename, description in real_docs.items():
            doc_file = raw_dir / filename
            if not doc_file.exists():
                print(f"   ⚠️ Skipping {filename} - not found")
                continue
                
            print(f"   📄 Processing: {filename} ({description})")
            
            # Ingest document
            document = ingestion_service.ingest_text_file(
                file_path=doc_file,
                source="data-training",
                title=description
            )
            
            # Process first 10 meaningful sentences (to keep it manageable)
            meaningful_sentences = []
            for sentence in document.sentences:
                # Skip very short or header-like sentences
                if (len(sentence.text.strip()) > 50 and 
                    not sentence.text.startswith(('|', 'Page', 'Chapter', 'Figure')) and
                    any(word in sentence.text.lower() for word in ['shrimp', 'vibrio', 'disease', 'aquaculture', 'larvae'])):
                    meaningful_sentences.append(sentence)
                    if len(meaningful_sentences) >= 10:
                        break
            
            print(f"      Found {len(meaningful_sentences)} meaningful sentences")
            
            # Generate candidates for each sentence
            for i, sentence in enumerate(meaningful_sentences):
                try:
                    print(f"        Sentence {i+1}: {sentence.text[:60]}...")
                    
                    # Generate entities with OpenAI
                    entities = await llm_generator.extract_entities(sentence.text)
                    
                    if entities:
                        # Insert into database
                        with db._get_connection() as conn:
                            # Insert sentence if not exists
                            conn.execute("""
                                INSERT OR IGNORE INTO sentences (sent_id, doc_id, text, start_offset, end_offset)
                                VALUES (?, ?, ?, ?, ?)
                            """, (sentence.sent_id, document.doc_id, sentence.text, sentence.start, sentence.end))
                            
                            # Get sentence ID
                            cursor = conn.execute("SELECT id FROM sentences WHERE sent_id = ? AND doc_id = ?", 
                                                (sentence.sent_id, document.doc_id))
                            sentence_db_id = cursor.fetchone()[0]
                            
                            # Insert candidate
                            entities_json = json.dumps([{
                                "text": e.text,
                                "label": e.label, 
                                "start": e.start,
                                "end": e.end,
                                "confidence": e.confidence
                            } for e in entities])
                            
                            cursor = conn.execute("""
                                INSERT INTO candidates 
                                (doc_id, sent_id, sentence_id, source, candidate_type, entities, confidence, priority_score)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                document.doc_id, sentence.sent_id, sentence_db_id, "openai_gpt4o_mini", 
                                "entity", entities_json, 
                                sum(e.confidence for e in entities) / len(entities),
                                min(0.9, sum(e.confidence for e in entities) / len(entities))
                            ))
                            
                            candidate_id = cursor.lastrowid
                            
                            # Insert into triage queue
                            priority_score = min(0.9, sum(e.confidence for e in entities) / len(entities))
                            if priority_score > 0.8:
                                priority_level = "high"
                            elif priority_score > 0.6:
                                priority_level = "medium"
                            else:
                                priority_level = "low"
                                
                            conn.execute("""
                                INSERT INTO triage_queue (candidate_id, priority_score, priority_level, status)
                                VALUES (?, ?, ?, ?)
                            """, (candidate_id, priority_score, priority_level, "pending"))
                            
                            conn.commit()
                            
                            total_items += 1
                            print(f"          ✅ Added: {len(entities)} entities")
                    
                    # Small delay for API
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    print(f"          ❌ Error: {e}")
        
        print(f"   📊 Total real items added: {total_items}")
        return total_items
        
    except Exception as e:
        print(f"   ❌ Error loading sentences: {e}")
        return 0

async def main():
    """Clear dummy data and load real sentences"""
    
    print("🚀 Loading Real Data into Annotation Queue\n")
    
    # Clear dummy data
    if not clear_dummy_data():
        return 1
    
    print()
    
    # Load real sentences
    real_items = await load_real_sentences()
    
    print(f"\n🎯 Real Data Loading Complete!")
    print(f"   📊 Real annotation items: {real_items}")
    print(f"   🌐 Refresh the UI: http://localhost:3010")
    
    print(f"\n✅ Queue now contains real sentences from your imported documents!")
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)