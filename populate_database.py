#!/usr/bin/env python3
"""
Directly populate the database with imported documents.
This script bypasses the API authentication and directly inserts documents into SQLite.
"""

import sys
import json
import re
from pathlib import Path
from datetime import datetime

# Add pipeline root to sys.path
pipeline_root = Path(__file__).parent
sys.path.append(str(pipeline_root))

from services.api.db_sqlite import SimpleDatabase

def segment_sentences(text: str) -> list:
    """Simple sentence segmentation using regex"""
    # Basic sentence splitting on period, exclamation, question mark
    sentences = re.split(r'[.!?]+', text)
    
    results = []
    offset = 0
    
    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if not sent:
            continue
            
        # Find the actual start position in original text
        start = text.find(sent, offset)
        if start == -1:
            start = offset
        
        end = start + len(sent)
        
        results.append({
            'sent_id': f's{i}',
            'text': sent,
            'start': start,
            'end': end
        })
        
        offset = end
    
    return results

def main():
    """Populate database with imported documents"""
    
    print("Populating database with imported documents...")
    
    # Initialize database
    db = SimpleDatabase()
    
    # Create default users for API access
    print("Creating default users...")
    try:
        db.create_default_users()
        print("✓ Default users created (admin token: local-admin-2024)")
    except Exception as e:
        print(f"Warning: Could not create users: {e}")
    
    # Get list of imported documents
    raw_dir = pipeline_root / "data/raw"
    text_files = list(raw_dir.glob("*.txt"))
    
    # Filter to only the files we just imported (exclude the original test files)
    original_files = {"doc_1759125903913.txt", "doc_1759126007300.txt", "test_doc_2.txt"}
    imported_files = [f for f in text_files if f.name not in original_files]
    
    print(f"Found {len(imported_files)} imported documents to load")
    
    # Process each document
    loaded_count = 0
    for doc_file in imported_files:  # Process all documents
        try:
            print(f"Loading: {doc_file.name}")
            
            # Read the document text
            with open(doc_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            doc_id = doc_file.stem
            title = doc_file.stem
            
            # Create document record
            metadata = {
                "filename": doc_file.name,
                "source_path": str(doc_file),
                "import_source": "data-training",
                "file_size": len(text)
            }
            
            # Insert document
            doc_data = {
                "doc_id": doc_id,
                "title": title,
                "raw_text": text,
                "source": "data-training",
                "metadata": metadata
            }
            doc_id_created = db.create_document(doc_data)
            
            if doc_id_created:
                print(f"  ✓ Document loaded (ID: {doc_id_created})")
                loaded_count += 1
            else:
                print(f"  ✗ Failed to create document")
        
        except Exception as e:
            print(f"  ✗ Error loading {doc_file.name}: {e}")
    
    print(f"\nLoaded {loaded_count} documents to database")
    
    # Get database statistics
    try:
        stats = db.get_statistics()
        print(f"\nDatabase statistics:")
        print(f"  Documents: {stats.get('total_documents', 0)}")
        print(f"  Sentences: {stats.get('total_sentences', 0)}")
        print(f"  Processed sentences: {stats.get('processed_sentences', 0)}")
    except Exception as e:
        print(f"Error getting statistics: {e}")
    
    print("\nDatabase population completed! 🦐")
    print("\nTo access the API, use: Authorization: Bearer local-admin-2024")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)