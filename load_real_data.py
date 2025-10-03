#!/usr/bin/env python3
"""
Load actual imported data into the running database.
This populates the database with real documents and candidates.
"""

import sys
import json
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime

def load_documents_to_db():
    """Load documents from data/raw/ into database"""
    
    # Database path
    db_path = Path("services/api/annotations.db")
    raw_dir = Path("data/raw")
    
    print("📄 Loading documents into database...")
    
    # Connect to database
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                source TEXT,
                raw_text TEXT,
                sentence_count INTEGER DEFAULT 0,
                character_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT DEFAULT '{}'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                sent_id TEXT NOT NULL,
                text TEXT NOT NULL,
                start_offset INTEGER,
                end_offset INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id INTEGER UNIQUE NOT NULL,
                doc_id TEXT NOT NULL,
                sent_id TEXT NOT NULL,
                sentence_text TEXT,
                entity_data TEXT,
                relation_data TEXT,
                topic_data TEXT,
                model_used TEXT DEFAULT 'openai',
                confidence_score REAL DEFAULT 0.8,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS triage_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id INTEGER UNIQUE NOT NULL,
                candidate_id INTEGER NOT NULL,
                doc_id TEXT NOT NULL,
                sent_id TEXT NOT NULL,
                priority_score REAL DEFAULT 0.5,
                priority_level TEXT DEFAULT 'medium',
                status TEXT DEFAULT 'pending',
                assigned_to TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
            )
        ''')
        
        docs_loaded = 0
        sentences_loaded = 0
        
        # Load each document
        for doc_file in raw_dir.glob("*.txt"):
            try:
                with open(doc_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Generate doc_id
                doc_id = hashlib.md5(f"{doc_file.name}:{content[:100]}".encode()).hexdigest()[:12]
                
                # Split into sentences
                sentences = [s.strip() + "." for s in content.split('.') if s.strip() and len(s.strip()) > 20]
                
                # Insert document
                cursor.execute('''
                    INSERT OR REPLACE INTO documents 
                    (doc_id, title, source, raw_text, sentence_count, character_count, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    doc_id,
                    doc_file.stem,
                    "imported",
                    content,
                    len(sentences),
                    len(content),
                    json.dumps({"filename": doc_file.name, "imported_at": datetime.utcnow().isoformat()})
                ))
                
                # Insert sentences
                for i, sentence in enumerate(sentences):
                    sent_id = f"s{i}"
                    cursor.execute('''
                        INSERT OR REPLACE INTO sentences 
                        (doc_id, sent_id, text, start_offset, end_offset)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (doc_id, sent_id, sentence, 0, len(sentence)))
                
                docs_loaded += 1
                sentences_loaded += len(sentences)
                print(f"   ✅ Loaded: {doc_file.name} ({len(sentences)} sentences)")
                
            except Exception as e:
                print(f"   ❌ Error loading {doc_file.name}: {e}")
        
        conn.commit()
        print(f"\n📊 Summary: {docs_loaded} documents, {sentences_loaded} sentences loaded")
        return docs_loaded, sentences_loaded

def load_candidates_to_db():
    """Load OpenAI candidates into database"""
    
    db_path = Path("services/api/annotations.db")
    candidates_dir = Path("data/candidates")
    
    print("\n🤖 Loading OpenAI candidates into database...")
    
    # Connect to database
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        candidates_loaded = 0
        
        # Load OpenAI candidates
        for candidate_file in candidates_dir.glob("*openai_candidates.jsonl"):
            try:
                with open(candidate_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f):
                        if line.strip():
                            try:
                                candidate = json.loads(line)
                                
                                candidate_id = candidate.get('candidate_id', line_num + 1000)
                                doc_id = candidate.get('doc_id', 'unknown')
                                sent_id = candidate.get('sent_id', f's{line_num}')
                                sentence_text = candidate.get('sentence', candidate.get('text', ''))
                                
                                # Extract entities
                                entities = candidate.get('entities', [])
                                relations = candidate.get('relations', [])
                                topics = candidate.get('topics', [])
                                
                                # Insert candidate
                                cursor.execute('''
                                    INSERT OR REPLACE INTO candidates 
                                    (candidate_id, doc_id, sent_id, sentence_text, entity_data, relation_data, topic_data, model_used, confidence_score)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                ''', (
                                    candidate_id,
                                    doc_id,
                                    sent_id,
                                    sentence_text,
                                    json.dumps(entities),
                                    json.dumps(relations),
                                    json.dumps(topics),
                                    'openai',
                                    candidate.get('confidence', 0.8)
                                ))
                                
                                candidates_loaded += 1
                                
                            except json.JSONDecodeError as e:
                                print(f"   ⚠️ Skipping invalid JSON on line {line_num + 1}: {e}")
                                continue
                
                print(f"   ✅ Loaded: {candidate_file.name}")
                
            except Exception as e:
                print(f"   ❌ Error loading {candidate_file.name}: {e}")
        
        conn.commit()
        print(f"\n🎯 Summary: {candidates_loaded} candidates loaded")
        return candidates_loaded

def populate_triage_queue():
    """Populate triage queue with candidates"""
    
    db_path = Path("services/api/annotations.db")
    
    print("\n📋 Populating triage queue...")
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Get all candidates
        cursor.execute('''
            SELECT candidate_id, doc_id, sent_id, entity_data, sentence_text 
            FROM candidates 
            ORDER BY candidate_id
        ''')
        
        candidates = cursor.fetchall()
        queue_items = 0
        
        for i, (candidate_id, doc_id, sent_id, entity_data, sentence_text) in enumerate(candidates):
            try:
                # Calculate priority based on entities
                entities = json.loads(entity_data) if entity_data else []
                priority_score = min(0.95, len(entities) * 0.2 + 0.3)
                
                if priority_score > 0.8:
                    priority_level = "high"
                elif priority_score > 0.5:
                    priority_level = "medium"
                else:
                    priority_level = "low"
                
                # Insert into triage queue
                cursor.execute('''
                    INSERT OR REPLACE INTO triage_queue 
                    (item_id, candidate_id, doc_id, sent_id, priority_score, priority_level, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    i + 1,
                    candidate_id,
                    doc_id,
                    sent_id,
                    priority_score,
                    priority_level,
                    'pending'
                ))
                
                queue_items += 1
                
            except Exception as e:
                print(f"   ⚠️ Error processing candidate {candidate_id}: {e}")
        
        conn.commit()
        print(f"\n⚡ Summary: {queue_items} items added to triage queue")
        return queue_items

def verify_data():
    """Verify loaded data"""
    
    db_path = Path("services/api/annotations.db")
    
    print("\n🔍 Verifying loaded data...")
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Check documents
        cursor.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        
        # Check sentences
        cursor.execute("SELECT COUNT(*) FROM sentences")
        sentence_count = cursor.fetchone()[0]
        
        # Check candidates
        cursor.execute("SELECT COUNT(*) FROM candidates")
        candidate_count = cursor.fetchone()[0]
        
        # Check queue
        cursor.execute("SELECT COUNT(*) FROM triage_queue")
        queue_count = cursor.fetchone()[0]
        
        print(f"   📄 Documents: {doc_count}")
        print(f"   📝 Sentences: {sentence_count}")
        print(f"   🤖 Candidates: {candidate_count}")
        print(f"   📋 Queue items: {queue_count}")
        
        # Show sample queue items
        cursor.execute('''
            SELECT t.item_id, t.doc_id, t.sent_id, t.priority_level, c.sentence_text
            FROM triage_queue t
            JOIN candidates c ON t.candidate_id = c.candidate_id
            ORDER BY t.priority_score DESC
            LIMIT 3
        ''')
        
        samples = cursor.fetchall()
        print(f"\n🎯 Sample queue items:")
        for item_id, doc_id, sent_id, priority, text in samples:
            preview = text[:80] + "..." if len(text) > 80 else text
            print(f"   {item_id}. [{priority}] {doc_id}:{sent_id} - {preview}")

def main():
    """Load all data into database"""
    
    print("🚀 Loading Real Data into Running Database\n")
    
    # Load documents
    docs, sentences = load_documents_to_db()
    
    # Load candidates
    candidates = load_candidates_to_db()
    
    # Populate queue
    queue_items = populate_triage_queue()
    
    # Verify
    verify_data()
    
    print(f"\n✅ Data Loading Complete!")
    print(f"   📊 {docs} documents with {sentences} sentences")
    print(f"   🤖 {candidates} OpenAI candidates")
    print(f"   📋 {queue_items} items in annotation queue")
    print(f"\n💡 The API will now serve real data from your imported documents")
    print(f"🔄 Refresh your browser to see the actual content!")

if __name__ == "__main__":
    main()