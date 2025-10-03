#!/usr/bin/env python3
"""
Import dataset from data-training project into the shrimp annotation pipeline.
This script copies documents from the data-training/data/output/text directory
and processes them through the document ingestion service.
"""

import os
import sys
import shutil
import json
from pathlib import Path
from datetime import datetime

# Add pipeline root to sys.path
pipeline_root = Path(__file__).parent
sys.path.append(str(pipeline_root))

from services.ingestion.document_ingestion import DocumentIngestionService

def main():
    """Import documents from data-training dataset"""
    
    # Paths
    source_dir = Path("/Users/macbook/Documents/data-training/data/output/text")
    pipeline_raw_dir = Path("/Users/macbook/Documents/shrimp-annotation-pipeline/data/raw")
    
    # Ensure directories exist
    pipeline_raw_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Importing documents from: {source_dir}")
    print(f"Target directory: {pipeline_raw_dir}")
    
    if not source_dir.exists():
        print(f"ERROR: Source directory does not exist: {source_dir}")
        return 1
    
    # Get list of text files
    text_files = list(source_dir.glob("*.txt"))
    print(f"Found {len(text_files)} text files to import")
    
    if not text_files:
        print("No .txt files found in source directory")
        return 1
    
    # Copy files to pipeline raw directory
    imported_files = []
    for source_file in text_files:
        target_file = pipeline_raw_dir / source_file.name
        try:
            shutil.copy2(source_file, target_file)
            imported_files.append(target_file)
            print(f"Copied: {source_file.name}")
        except Exception as e:
            print(f"ERROR copying {source_file.name}: {e}")
    
    print(f"\nSuccessfully copied {len(imported_files)} files")
    
    # Initialize document ingestion service
    print("\nInitializing document ingestion service...")
    try:
        ingestion_service = DocumentIngestionService(
            data_training_path=Path("/Users/macbook/Documents/data-training"),
            segmenter="regex"
        )
        
        # Process each imported file
        print("\nProcessing documents...")
        processed_docs = []
        
        for doc_file in imported_files:
            try:
                print(f"Processing: {doc_file.name}")
                
                # Process document through ingestion service
                document = ingestion_service.ingest_text_file(
                    file_path=doc_file,
                    source="data-training",
                    title=doc_file.stem
                )
                
                processed_docs.append(document)
                print(f"  - Processed {len(document.sentences)} sentences")
                
            except Exception as e:
                print(f"ERROR processing {doc_file.name}: {e}")
        
        print(f"\nProcessing complete!")
        print(f"Imported {len(processed_docs)} documents")
        
        # Summary statistics
        total_sentences = sum(len(doc.sentences) for doc in processed_docs)
        total_chars = sum(len(doc.raw_text) for doc in processed_docs)
        
        print(f"Total sentences: {total_sentences}")
        print(f"Total characters: {total_chars:,}")
        print(f"Average sentences per document: {total_sentences / len(processed_docs):.1f}")
        
        # Save import summary
        summary = {
            "import_date": datetime.now().isoformat(),
            "source_directory": str(source_dir),
            "imported_files": [f.name for f in imported_files],
            "processed_documents": len(processed_docs),
            "total_sentences": total_sentences,
            "total_characters": total_chars,
            "documents": [
                {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "sentences": len(doc.sentences),
                    "characters": len(doc.raw_text),
                    "metadata": doc.metadata
                }
                for doc in processed_docs
            ]
        }
        
        summary_file = pipeline_root / "data/import_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\nImport summary saved to: {summary_file}")
        print("\nDataset import completed successfully! 🦐")
        
        return 0
        
    except Exception as e:
        print(f"ERROR during document ingestion: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)