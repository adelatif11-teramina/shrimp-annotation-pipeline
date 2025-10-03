#!/usr/bin/env python3
"""
Add real candidates from our OpenAI-generated files to the database.
This will populate the database with real data that the API can use.
"""

import sys
import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime

# Add pipeline root to sys.path
pipeline_root = Path(__file__).parent
sys.path.append(str(pipeline_root))

def add_real_candidates():
    """Add real candidates from generated files to database"""
    print("📊 Adding real OpenAI candidates to database...")
    
    db_path = pipeline_root / "data/local/annotations.db"
    candidates_dir = pipeline_root / "data/candidates"
    
    # Find OpenAI candidate files
    openai_files = list(candidates_dir.glob("*_openai_candidates.jsonl"))
    
    if not openai_files:
        print("   ❌ No OpenAI candidate files found")
        return 0
    
    print(f"   Found {len(openai_files)} OpenAI candidate files")
    
    total_added = 0
    
    with sqlite3.connect(db_path) as conn:
        for candidate_file in openai_files:
            print(f"   📄 Processing: {candidate_file.name}")
            
            try:
                with open(candidate_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f):
                        if not line.strip():
                            continue
                            
                        candidate = json.loads(line)
                        
                        # Insert candidate
                        entity_data = json.dumps(candidate.get("entities", []))
                        confidence = sum(e["confidence"] for e in candidate.get("entities", [])) / max(1, len(candidate.get("entities", [])))
                        
                        cursor = conn.execute("""
                            INSERT INTO candidates 
                            (doc_id, sent_id, source, entity_data, relation_data, topic_data, confidence)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            candidate["doc_id"],
                            candidate["sent_id"], 
                            "llm",  # Match the CHECK constraint
                            entity_data,
                            "[]",  # Empty relations for now
                            "[]",  # Empty topics for now
                            confidence
                        ))
                        
                        candidate_id = cursor.lastrowid
                        
                        # Add to triage queue
                        priority_score = confidence
                        if priority_score > 0.8:
                            priority_level = "high"
                        elif priority_score > 0.6:
                            priority_level = "medium"
                        else:
                            priority_level = "low"
                        
                        conn.execute("""
                            INSERT INTO triage_queue 
                            (doc_id, sent_id, candidate_id, priority_score, priority_level, status)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            candidate["doc_id"],
                            candidate["sent_id"],
                            candidate_id,
                            priority_score,
                            priority_level,
                            "pending"
                        ))
                        
                        total_added += 1
                        
                        if line_num >= 9:  # Limit to first 10 per file
                            break
                
                conn.commit()
                print(f"      ✅ Added {min(10, line_num + 1)} candidates")
                
            except Exception as e:
                print(f"      ❌ Error: {e}")
    
    print(f"   📊 Total candidates added: {total_added}")
    return total_added

def verify_data():
    """Verify the data was added correctly"""
    print("🔍 Verifying database content...")
    
    db_path = pipeline_root / "data/local/annotations.db"
    
    with sqlite3.connect(db_path) as conn:
        # Check candidates
        cursor = conn.execute("SELECT COUNT(*) FROM candidates WHERE source = 'llm'")
        candidate_count = cursor.fetchone()[0]
        print(f"   📊 LLM Candidates: {candidate_count}")
        
        # Check triage queue
        cursor = conn.execute("SELECT COUNT(*) FROM triage_queue WHERE status = 'pending'")
        queue_count = cursor.fetchone()[0]
        print(f"   📋 Queue Items: {queue_count}")
        
        # Sample queue items
        cursor = conn.execute("""
            SELECT tq.doc_id, tq.priority_level, c.entity_data 
            FROM triage_queue tq 
            JOIN candidates c ON tq.candidate_id = c.candidate_id 
            WHERE tq.status = 'pending' 
            LIMIT 3
        """)
        
        samples = cursor.fetchall()
        print(f"   📝 Sample queue items:")
        for doc_id, priority, entities in samples:
            entity_count = len(json.loads(entities))
            print(f"      - {doc_id}: {priority} priority, {entity_count} entities")

def main():
    """Add real candidates to database"""
    
    print("🚀 Adding Real Candidates to Database\n")
    
    # Add candidates
    added = add_real_candidates()
    
    if added > 0:
        print()
        verify_data()
        
        print(f"\n✅ Success! Added {added} real candidates")
        print(f"🔄 Refresh your browser at http://localhost:3010 to see real data")
        print(f"💡 The queue should now show actual sentences from your documents")
    else:
        print(f"\n❌ No candidates were added")
        print(f"💡 Run: python generate_openai_candidates.py first")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)