#!/usr/bin/env python3
"""
Simple Document Import for Shrimp Annotation Pipeline
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime

# Import database directly
sys.path.insert(0, 'services/api')
from db_sqlite import SimpleDatabase

def simple_text_extract(file_path):
    """Simple text extraction"""
    if file_path.suffix.lower() == '.txt':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()[:5000]  # Limit to 5000 chars for testing
        except:
            return f"Text file: {file_path.name}"
    else:
        # For PDFs, create placeholder content
        return f"PDF Document: {file_path.name}\n\nSample content for annotation testing. This document contains research on shrimp aquaculture, Vibrio bacteria, and disease management in marine environments."

def main():
    print("📥 Simple Document Import")
    
    # Initialize database
    db = SimpleDatabase()
    print("✅ Database connected")
    
    # Source directory
    source_dir = Path("/Users/macbook/Documents/data-training/data/input")
    files = list(source_dir.glob("*.pdf"))[:5] + list(source_dir.glob("*.txt"))  # Import first 5 PDFs + all txt
    
    print(f"📂 Importing {len(files)} files...")
    
    for i, file_path in enumerate(files):
        try:
            print(f"  [{i+1}/{len(files)}] {file_path.name}")
            
            # Extract text
            text_content = simple_text_extract(file_path)
            
            # Create document
            title = file_path.stem.replace('_', ' ').replace('-', ' ').title()
            doc_id = hashlib.md5(f"{title}:{text_content[:50]}".encode()).hexdigest()[:12]
            
            doc_data = {
                "doc_id": doc_id,
                "title": title,
                "source": "imported_document", 
                "raw_text": text_content,
                "metadata": json.dumps({
                    "filename": file_path.name,
                    "imported_at": datetime.utcnow().isoformat()
                })
            }
            
            # Import
            document_id = db.create_document(doc_data)
            print(f"    ✅ Imported (ID: {document_id})")
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
    
    # Show stats
    print("\n📊 Final Statistics:")
    stats = db.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n🎯 Ready for annotation!")
    print("   Start servers: ./start_local.sh")
    print("   Open UI: http://localhost:3000")

if __name__ == "__main__":
    main()