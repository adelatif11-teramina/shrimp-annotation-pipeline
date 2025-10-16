#!/usr/bin/env python3
"""
Batch Document Export Script for RAG Storage

Export all documents from the annotation pipeline to JSONL format
for RAG applications and external storage systems.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from services.storage.document_exporter_simple import SimpleDocumentExporter


def main():
    parser = argparse.ArgumentParser(
        description="Export documents from annotation pipeline for RAG storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export all documents to JSONL
  python scripts/export_documents_for_rag.py --all

  # Export specific documents
  python scripts/export_documents_for_rag.py --doc-ids doc1,doc2,doc3

  # Export with custom output path
  python scripts/export_documents_for_rag.py --all --output data/rag_export.jsonl

  # Show summary of available documents
  python scripts/export_documents_for_rag.py --summary

  # Export only annotated documents
  python scripts/export_documents_for_rag.py --annotated-only

  # Export by source type
  python scripts/export_documents_for_rag.py --source paper --source report
        """
    )
    
    # Export options
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Export all documents"
    )
    parser.add_argument(
        "--doc-ids", 
        help="Comma-separated list of document IDs to export"
    )
    parser.add_argument(
        "--annotated-only", 
        action="store_true", 
        help="Export only documents with gold annotations"
    )
    parser.add_argument(
        "--source", 
        action="append", 
        help="Export documents from specific source(s) (can be used multiple times)"
    )
    
    # Output options
    parser.add_argument(
        "--output", 
        help="Output JSONL file path (default: data/content/documents_export_TIMESTAMP.jsonl)"
    )
    parser.add_argument(
        "--individual", 
        action="store_true", 
        help="Save each document as individual JSON file in addition to JSONL"
    )
    
    # Database options
    parser.add_argument(
        "--db-url", 
        help="Database URL (default: sqlite:///data/local/annotations.db)"
    )
    
    # Information options
    parser.add_argument(
        "--summary", 
        action="store_true", 
        help="Show summary of available documents and exit"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Show what would be exported without actually exporting"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.all, args.doc_ids, args.annotated_only, args.source, args.summary]):
        parser.error("Must specify one of: --all, --doc-ids, --annotated-only, --source, or --summary")
    
    try:
        # Create exporter
        if args.verbose:
            print("Connecting to database...")
        
        db_path = "data/local/annotations.db"
        if args.db_url and args.db_url.startswith("sqlite:///"):
            db_path = args.db_url.replace("sqlite:///", "")
        
        exporter = SimpleDocumentExporter(
            db_path=db_path,
            export_base_path="data/content"
        )
        
        # Show summary if requested
        if args.summary:
            summary = exporter.get_export_summary()
            print("📊 Document Export Summary")
            print("=" * 50)
            print(f"Total documents: {summary['total_documents']}")
            print(f"Documents with annotations: {summary['documents_with_annotations']}")
            print(f"Annotation coverage: {summary['annotation_coverage']:.1%}")
            print(f"Export path: {summary['export_base_path']}")
            print("\nBy source:")
            for source, info in summary['by_source'].items():
                print(f"  {source}: {info['count']} docs ({info['with_annotations']} annotated, {info['total_sentences']} sentences)")
            return
        
        # Determine which documents to export
        doc_ids = None
        
        if args.doc_ids:
            doc_ids = [id.strip() for id in args.doc_ids.split(",")]
            if args.verbose:
                print(f"Exporting specific documents: {doc_ids}")
        
        elif args.annotated_only:
            # Get documents with annotations using direct SQL
            import sqlite3
            with sqlite3.connect(exporter.db_path) as conn:
                annotated_docs = conn.execute("""
                    SELECT DISTINCT d.doc_id 
                    FROM documents d
                    JOIN gold_annotations ga ON d.id = ga.document_id
                """).fetchall()
                doc_ids = [row[0] for row in annotated_docs]
            
            if args.verbose:
                print(f"Found {len(doc_ids)} documents with annotations")
        
        elif args.source:
            # Get documents by source using direct SQL
            import sqlite3
            with sqlite3.connect(exporter.db_path) as conn:
                placeholders = ','.join('?' * len(args.source))
                source_docs = conn.execute(
                    f"SELECT doc_id FROM documents WHERE source IN ({placeholders})", 
                    args.source
                ).fetchall()
                doc_ids = [row[0] for row in source_docs]
            
            if args.verbose:
                print(f"Found {len(doc_ids)} documents from sources: {args.source}")
        
        if args.dry_run:
            if doc_ids:
                print(f"Would export {len(doc_ids)} documents:")
                for doc_id in doc_ids[:10]:  # Show first 10
                    print(f"  - {doc_id}")
                if len(doc_ids) > 10:
                    print(f"  ... and {len(doc_ids) - 10} more")
            else:
                summary = exporter.get_export_summary()
                print(f"Would export all {summary['total_documents']} documents")
            return
        
        # Perform export
        print("🚀 Starting document export...")
        
        if args.individual and doc_ids:
            # Export individual files
            print("Exporting individual document files...")
            for doc_id in doc_ids:
                try:
                    export_data = exporter.export_document(doc_id)
                    filepath = exporter.save_document_export(doc_id, export_data)
                    if args.verbose:
                        print(f"  ✓ {doc_id} -> {filepath}")
                except Exception as e:
                    print(f"  ✗ Error exporting {doc_id}: {e}")
        
        # Export to JSONL
        output_file = exporter.export_to_jsonl(
            doc_ids=doc_ids,
            output_file=args.output
        )
        
        print(f"✅ Export completed: {output_file}")
        
        # Show export statistics
        if args.verbose:
            print("\n📈 Export Statistics:")
            with open(output_file, 'r') as f:
                line_count = sum(1 for line in f)
            
            file_size = os.path.getsize(output_file)
            print(f"  Documents exported: {line_count}")
            print(f"  File size: {file_size / 1024 / 1024:.1f} MB")
            print(f"  Average per document: {file_size / line_count / 1024:.1f} KB")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()