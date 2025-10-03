#!/usr/bin/env python3
"""
Load imported documents into the annotation pipeline database.
This script takes the processed documents and loads them into the SQLite database
so they're available for annotation through the API.
"""

import sys
from pathlib import Path
import requests
import json

# Add pipeline root to sys.path
pipeline_root = Path(__file__).parent
sys.path.append(str(pipeline_root))

from services.ingestion.document_ingestion import DocumentIngestionService

def main():
    """Load processed documents into the database"""
    
    print("Loading documents into annotation database...")
    
    # Initialize the ingestion service to re-process documents
    ingestion_service = DocumentIngestionService(segmenter="regex")
    
    # Get list of imported documents
    raw_dir = pipeline_root / "data/raw"
    text_files = list(raw_dir.glob("*.txt"))
    
    # Filter to only the files we just imported (exclude the original test files)
    original_files = {"doc_1759125903913.txt", "doc_1759126007300.txt", "test_doc_2.txt"}
    imported_files = [f for f in text_files if f.name not in original_files]
    
    print(f"Found {len(imported_files)} imported documents to load")
    
    api_base = "http://localhost:8000"
    
    # Process each document and send to API
    loaded_count = 0
    for doc_file in imported_files[:5]:  # Start with first 5 documents
        try:
            print(f"Loading: {doc_file.name}")
            
            # Read the document text
            with open(doc_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Create document payload
            doc_data = {
                "doc_id": doc_file.stem,
                "text": text,
                "title": doc_file.stem,
                "source": "data-training",
                "metadata": {
                    "filename": doc_file.name,
                    "source_path": str(doc_file),
                    "import_source": "data-training"
                }
            }
            
            # Send to API
            response = requests.post(f"{api_base}/documents/", json=doc_data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"  ✓ Loaded {result.get('sentences', 0)} sentences")
                loaded_count += 1
            else:
                print(f"  ✗ Failed to load: {response.status_code}")
                if response.status_code != 404:  # Don't print error for non-existent endpoint
                    print(f"    Response: {response.text[:200]}")
        
        except Exception as e:
            print(f"  ✗ Error loading {doc_file.name}: {e}")
    
    print(f"\nLoaded {loaded_count} documents to database")
    
    # Check the API status
    try:
        response = requests.get(f"{api_base}/health")
        if response.status_code == 200:
            status = response.json()
            print(f"\nDatabase status:")
            print(f"  Documents: {status.get('total_documents', 0)}")
            print(f"  Sentences: {status.get('total_sentences', 0)}")
        else:
            print("Could not get database status")
    except Exception as e:
        print(f"Error checking database status: {e}")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)