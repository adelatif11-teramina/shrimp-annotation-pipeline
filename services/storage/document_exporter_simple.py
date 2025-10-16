"""
Simple Document Export Service for RAG Storage

Exports documents directly from SQLite database using raw SQL queries
to handle schema differences and ensure compatibility.
"""

import json
import os
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path


class SimpleDocumentExporter:
    """Export documents to JSONL format using direct SQLite access"""
    
    def __init__(self, db_path: str = "data/local/annotations.db", export_base_path: str = "data/content"):
        self.db_path = db_path
        self.export_base_path = Path(export_base_path)
        self.export_base_path.mkdir(parents=True, exist_ok=True)
    
    def _get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def export_document(self, doc_id: str) -> Dict[str, Any]:
        """Export a single document with all associated data"""
        
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            
            # Get document
            doc_row = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            
            if not doc_row:
                raise ValueError(f"Document {doc_id} not found")
            
            doc = dict(doc_row)
            
            # Get sentences
            sentences = conn.execute("""
                SELECT * FROM sentences 
                WHERE document_id = ? 
                ORDER BY start_offset
            """, (doc['id'],)).fetchall()
            
            # Build export structure
            export_data = {
                "doc_id": doc['doc_id'],
                "title": doc.get('title'),
                "source": doc['source'],
                "full_content": doc.get('raw_text', ''),
                "content_length": len(doc.get('raw_text', '')),
                "sentence_count": len(sentences),
                "metadata": self._extract_document_metadata(doc),
                "sentences": []
            }
            
            # Process each sentence
            for sentence_row in sentences:
                sentence = dict(sentence_row)
                sentence_data = self._export_sentence(conn, sentence)
                export_data["sentences"].append(sentence_data)
            
            # Add document-level statistics
            export_data["statistics"] = self._calculate_document_stats(conn, doc, len(sentences))
            
            return export_data
    
    def _extract_document_metadata(self, doc: Dict) -> Dict[str, Any]:
        """Extract and parse document metadata"""
        metadata = {}
        
        # Handle JSON metadata field
        if 'metadata' in doc and doc['metadata']:
            try:
                metadata = json.loads(doc['metadata'])
            except (json.JSONDecodeError, TypeError):
                metadata = {}
        elif 'doc_metadata' in doc and doc['doc_metadata']:
            try:
                metadata = json.loads(doc['doc_metadata'])
            except (json.JSONDecodeError, TypeError):
                metadata = {}
        
        return {
            "document_metadata": metadata,
            "status": doc.get('status'),
            "created_at": doc.get('created_at'),
            "updated_at": doc.get('updated_at')
        }
    
    def _export_sentence(self, conn: sqlite3.Connection, sentence: Dict) -> Dict[str, Any]:
        """Export sentence with all annotations and candidates"""
        
        sentence_id = sentence['id']
        
        # Get gold annotations
        gold_annotations = conn.execute("""
            SELECT * FROM gold_annotations 
            WHERE sentence_id = ?
        """, (sentence_id,)).fetchall()
        
        # Get candidates
        candidates = conn.execute("""
            SELECT * FROM candidates 
            WHERE sentence_id = ?
        """, (sentence_id,)).fetchall()
        
        # Get triage items
        triage_items = conn.execute("""
            SELECT * FROM triage_items 
            WHERE sentence_id = ?
        """, (sentence_id,)).fetchall()
        
        sentence_data = {
            "sentence_id": sentence.get('sent_id'),
            "text": sentence['text'],
            "position": {
                "start_offset": sentence['start_offset'],
                "end_offset": sentence['end_offset'],
                "paragraph_id": sentence.get('paragraph_id')
            },
            "annotations": {
                "gold": [self._format_gold_annotation(dict(ann)) for ann in gold_annotations],
                "candidates": [self._format_candidate(dict(cand)) for cand in candidates]
            },
            "triage_status": {
                "items": [self._format_triage_item(dict(item)) for item in triage_items],
                "has_pending": any(dict(item).get('status') == "pending" for item in triage_items),
                "completed_count": sum(1 for item in triage_items if dict(item).get('status') == "completed")
            },
            "created_at": sentence.get('created_at')
        }
        
        return sentence_data
    
    def _format_gold_annotation(self, annotation: Dict) -> Dict[str, Any]:
        """Format gold annotation for export"""
        def parse_json_field(field_value):
            if isinstance(field_value, str) and field_value:
                try:
                    return json.loads(field_value)
                except (json.JSONDecodeError, TypeError):
                    return []
            return field_value or []
        
        return {
            "annotation_id": str(annotation['id']),
            "entities": parse_json_field(annotation.get('entities')),
            "relations": parse_json_field(annotation.get('relations')),
            "topics": parse_json_field(annotation.get('topics')),
            "annotator_email": annotation.get('annotator_email'),
            "status": annotation.get('status'),
            "confidence_level": annotation.get('confidence_level'),
            "notes": annotation.get('notes'),
            "decision_method": annotation.get('decision_method'),
            "created_at": annotation.get('created_at'),
            "updated_at": annotation.get('updated_at'),
            "reviewed_at": annotation.get('reviewed_at')
        }
    
    def _format_candidate(self, candidate: Dict) -> Dict[str, Any]:
        """Format candidate annotation for export"""
        candidate_data = {
            "candidate_id": str(candidate['id']),
            "type": candidate.get('candidate_type'),
            "label": candidate.get('label'),
            "confidence": candidate.get('confidence'),
            "model_name": candidate.get('model_name'),
            "model_version": candidate.get('model_version'),
            "generation_method": candidate.get('generation_method'),
            "created_at": candidate.get('created_at')
        }
        
        # Add type-specific fields
        if candidate.get('candidate_type') == "entity":
            candidate_data.update({
                "text": candidate.get('text'),
                "start_offset": candidate.get('start_offset'),
                "end_offset": candidate.get('end_offset')
            })
        elif candidate.get('candidate_type') == "relation":
            candidate_data.update({
                "head_candidate_id": str(candidate['head_candidate_id']) if candidate.get('head_candidate_id') else None,
                "tail_candidate_id": str(candidate['tail_candidate_id']) if candidate.get('tail_candidate_id') else None,
                "evidence": candidate.get('evidence')
            })
        elif candidate.get('candidate_type') == "topic":
            keywords = candidate.get('keywords')
            if isinstance(keywords, str) and keywords:
                try:
                    keywords = json.loads(keywords)
                except (json.JSONDecodeError, TypeError):
                    keywords = []
            
            candidate_data.update({
                "score": candidate.get('score'),
                "keywords": keywords or []
            })
        
        if candidate.get('rule_pattern'):
            candidate_data["rule_pattern"] = candidate['rule_pattern']
        
        return candidate_data
    
    def _format_triage_item(self, item: Dict) -> Dict[str, Any]:
        """Format triage item for export"""
        return {
            "item_id": item.get('item_id'),
            "priority_score": item.get('priority_score'),
            "priority_level": item.get('priority_level'),
            "status": item.get('status'),
            "assigned_to": item.get('assigned_to'),
            "assigned_at": item.get('assigned_at'),
            "completed_at": item.get('completed_at'),
            "scoring": {
                "confidence_score": item.get('confidence_score'),
                "novelty_score": item.get('novelty_score'),
                "impact_score": item.get('impact_score'),
                "disagreement_score": item.get('disagreement_score'),
                "authority_score": item.get('authority_score')
            }
        }
    
    def _calculate_document_stats(self, conn: sqlite3.Connection, doc: Dict, sentence_count: int) -> Dict[str, Any]:
        """Calculate document-level statistics"""
        
        doc_id = doc['id']
        
        # Count annotations
        gold_count = conn.execute(
            "SELECT COUNT(*) FROM gold_annotations WHERE document_id = ?", (doc_id,)
        ).fetchone()[0]
        
        candidate_count = conn.execute("""
            SELECT COUNT(*) FROM candidates c
            JOIN sentences s ON c.sentence_id = s.id
            WHERE s.document_id = ?
        """, (doc_id,)).fetchone()[0]
        
        completed_annotations = conn.execute("""
            SELECT COUNT(*) FROM gold_annotations 
            WHERE document_id = ? AND status IN ('accepted', 'completed')
        """, (doc_id,)).fetchone()[0]
        
        content = doc.get('raw_text', '')
        
        return {
            "sentence_count": sentence_count,
            "character_count": len(content),
            "word_count": len(content.split()) if content else 0,
            "annotation_counts": {
                "gold_annotations": gold_count,
                "candidate_annotations": candidate_count,
                "completed_annotations": completed_annotations
            },
            "annotation_coverage": completed_annotations / sentence_count if sentence_count > 0 else 0
        }
    
    def save_document_export(self, doc_id: str, export_data: Dict[str, Any]) -> str:
        """Save document export to file"""
        filename = f"{doc_id}.json"
        filepath = self.export_base_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
    
    def export_to_jsonl(self, doc_ids: List[str] = None, output_file: str = None) -> str:
        """Export multiple documents to JSONL format"""
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.export_base_path / f"documents_export_{timestamp}.jsonl"
        else:
            output_file = Path(output_file)
        
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            
            # Get documents to export
            if doc_ids:
                placeholders = ','.join('?' * len(doc_ids))
                documents = conn.execute(
                    f"SELECT * FROM documents WHERE doc_id IN ({placeholders})", doc_ids
                ).fetchall()
            else:
                documents = conn.execute("SELECT * FROM documents").fetchall()
        
        # Export each document
        exported_count = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc_row in documents:
                try:
                    doc = dict(doc_row)
                    export_data = self.export_document(doc['doc_id'])
                    f.write(json.dumps(export_data, ensure_ascii=False) + '\n')
                    exported_count += 1
                except Exception as e:
                    print(f"Error exporting document {doc['doc_id']}: {e}")
                    continue
        
        print(f"Exported {exported_count} documents to {output_file}")
        return str(output_file)
    
    def get_export_summary(self) -> Dict[str, Any]:
        """Get summary of available documents for export"""
        
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            
            # Get all documents
            documents = conn.execute("SELECT * FROM documents").fetchall()
            
            # Count by source
            doc_counts = {}
            total_with_annotations = 0
            
            for doc_row in documents:
                doc = dict(doc_row)
                source = doc['source']
                
                if source not in doc_counts:
                    doc_counts[source] = {
                        "count": 0,
                        "with_annotations": 0,
                        "total_sentences": 0
                    }
                
                doc_counts[source]["count"] += 1
                
                # Check if has annotations
                has_annotations = conn.execute(
                    "SELECT COUNT(*) FROM gold_annotations WHERE document_id = ?", (doc['id'],)
                ).fetchone()[0] > 0
                
                if has_annotations:
                    doc_counts[source]["with_annotations"] += 1
                    total_with_annotations += 1
                
                # Count sentences
                sentence_count = conn.execute(
                    "SELECT COUNT(*) FROM sentences WHERE document_id = ?", (doc['id'],)
                ).fetchone()[0]
                doc_counts[source]["total_sentences"] += sentence_count
        
        total_docs = len(documents)
        
        return {
            "total_documents": total_docs,
            "documents_with_annotations": total_with_annotations,
            "annotation_coverage": total_with_annotations / total_docs if total_docs > 0 else 0,
            "by_source": doc_counts,
            "export_base_path": str(self.export_base_path)
        }


if __name__ == "__main__":
    # CLI interface for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Export documents for RAG storage")
    parser.add_argument("--doc-id", help="Export specific document by ID")
    parser.add_argument("--all", action="store_true", help="Export all documents")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--summary", action="store_true", help="Show export summary")
    parser.add_argument("--db-path", help="Database path", default="data/local/annotations.db")
    
    args = parser.parse_args()
    
    exporter = SimpleDocumentExporter(args.db_path)
    
    if args.summary:
        summary = exporter.get_export_summary()
        print("Export Summary:")
        print(json.dumps(summary, indent=2))
    
    elif args.doc_id:
        export_data = exporter.export_document(args.doc_id)
        if args.output:
            exporter.save_document_export(args.doc_id, export_data)
            print(f"Exported document {args.doc_id} to {args.output}")
        else:
            print(json.dumps(export_data, indent=2))
    
    elif args.all:
        output_file = exporter.export_to_jsonl(output_file=args.output)
        print(f"Exported all documents to {output_file}")
    
    else:
        parser.print_help()