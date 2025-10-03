#!/usr/bin/env python3
"""
Document Import Script for Shrimp Annotation Pipeline
Imports PDF and text documents from the data-training directory
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import our database directly
sys.path.insert(0, str(project_root / "services" / "api"))
from db_sqlite import SimpleDatabase

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using PyPDF2 or fallback methods"""
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except ImportError:
        print("PyPDF2 not available, using fallback text extraction")
        return f"[PDF Document: {os.path.basename(pdf_path)}]\nContent extraction requires PyPDF2 library."
    except Exception as e:
        print(f"Error extracting PDF {pdf_path}: {e}")
        return f"[PDF Document: {os.path.basename(pdf_path)}]\nError during text extraction: {str(e)}"

def extract_text_from_txt(txt_path):
    """Extract text from text file"""
    try:
        with open(txt_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except Exception as e:
        print(f"Error reading text file {txt_path}: {e}")
        return f"[Text Document: {os.path.basename(txt_path)}]\nError during reading: {str(e)}"

def generate_doc_id(title, content_preview):
    """Generate unique document ID"""
    content = f"{title}:{content_preview[:100]}"
    return hashlib.md5(content.encode()).hexdigest()[:12]

def import_documents(source_dir, db):
    """Import all documents from source directory"""
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return 0
    
    imported_count = 0
    skipped_count = 0
    
    print(f"📂 Scanning directory: {source_path}")
    print("=" * 60)
    
    # Get all PDF and text files
    pdf_files = list(source_path.glob("*.pdf"))
    txt_files = list(source_path.glob("*.txt"))
    
    all_files = pdf_files + txt_files
    
    print(f"Found {len(pdf_files)} PDF files and {len(txt_files)} text files")
    print("")
    
    for file_path in all_files:
        try:
            print(f"📄 Processing: {file_path.name}")
            
            # Extract text based on file type
            if file_path.suffix.lower() == '.pdf':
                text_content = extract_text_from_pdf(file_path)
                source_type = "pdf_document"
            else:
                text_content = extract_text_from_txt(file_path)
                source_type = "text_document"
            
            # Create document data
            title = file_path.stem.replace('_', ' ').replace('-', ' ').title()
            doc_id = generate_doc_id(title, text_content)
            
            doc_data = {
                "doc_id": doc_id,
                "title": title,
                "source": source_type,
                "raw_text": text_content,
                "metadata": {
                    "original_filename": file_path.name,
                    "file_size": file_path.stat().st_size,
                    "import_date": datetime.utcnow().isoformat(),
                    "source_directory": str(source_path),
                    "file_type": file_path.suffix.lower()
                }
            }
            
            # Check if document already exists (simple check)
            try:
                existing_docs = db.get_documents(search=doc_id)
                if existing_docs["total"] > 0:
                    print(f"   ⏭️  Already exists (doc_id: {doc_id})")
                    skipped_count += 1
                    continue
            except Exception:
                # If check fails, continue with import
                pass
            
            # Import document
            document_id = db.create_document(doc_data)
            
            # Create sentences for processing
            sentences = text_content.split('.')
            sentence_count = 0
            
            for i, sentence_text in enumerate(sentences):
                sentence_text = sentence_text.strip()
                if len(sentence_text) > 10:  # Only process meaningful sentences
                    sentence_count += 1
            
            print(f"   ✅ Imported successfully")
            print(f"      📊 Document ID: {doc_id}")
            print(f"      📝 Text length: {len(text_content):,} characters")
            print(f"      📄 Estimated sentences: {sentence_count}")
            print(f"      🆔 Database ID: {document_id}")
            
            imported_count += 1
            
        except Exception as e:
            print(f"   ❌ Error importing {file_path.name}: {e}")
            continue
        
        print("")
    
    print("=" * 60)
    print(f"✅ Import completed:")
    print(f"   📥 Imported: {imported_count} documents")
    print(f"   ⏭️  Skipped: {skipped_count} documents")
    print(f"   📊 Total processed: {imported_count + skipped_count}")
    
    return imported_count

def main():
    print("🚀 Shrimp Annotation Pipeline - Document Import")
    print("=" * 60)
    
    # Initialize database
    try:
        db = SimpleDatabase()
        print("✅ Database connection established")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return 1
    
    # Import documents
    source_directory = "/Users/macbook/Documents/data-training/data/input"
    
    try:
        imported_count = import_documents(source_directory, db)
        
        if imported_count > 0:
            print("")
            print("🎯 Next steps:")
            print("  1. Start the annotation servers: ./start_local.sh")
            print("  2. Open the UI at: http://localhost:3000")
            print("  3. Begin annotation workflow")
            print("")
            print("📊 Database statistics:")
            stats = db.get_statistics()
            for key, value in stats.items():
                print(f"   {key}: {value}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)