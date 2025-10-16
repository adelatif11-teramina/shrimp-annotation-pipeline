"""
Document Export Service for RAG Storage

Exports documents, sentences, annotations, and metadata to JSONL format
for future RAG applications and external storage.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from sqlalchemy.orm import Session
from sqlalchemy import create_engine, and_

try:
    from services.database.models import (
        Document, Sentence, GoldAnnotation, Candidate, 
        TriageItem, AnnotationEvent
    )
except ImportError:
    # Fallback for simple database schema
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, ForeignKey, JSON
    import uuid
    
    Base = declarative_base()
    
    class Document(Base):
        __tablename__ = "documents"
        id = Column(Integer, primary_key=True, autoincrement=True)
        doc_id = Column(String(100), unique=True, nullable=False)
        title = Column(String(500))
        source = Column(String(100), nullable=False)
        raw_text = Column(Text)
        doc_metadata = Column(Text)  # JSON as text in SQLite
        status = Column(String(20))
        created_at = Column(DateTime)
        updated_at = Column(DateTime)
    
    class Sentence(Base):
        __tablename__ = "sentences"
        id = Column(Integer, primary_key=True, autoincrement=True)
        sent_id = Column(String(100), nullable=False)
        document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
        start_offset = Column(Integer, nullable=False)
        end_offset = Column(Integer, nullable=False)
        text = Column(Text, nullable=False)
        paragraph_id = Column(Integer)
        created_at = Column(DateTime)
    
    # Simplified classes for compatibility
    class GoldAnnotation(Base):
        __tablename__ = "gold_annotations"
        id = Column(Integer, primary_key=True, autoincrement=True)
        document_id = Column(Integer, ForeignKey("documents.id"))
        sentence_id = Column(Integer, ForeignKey("sentences.id"))
        entities = Column(Text)  # JSON as text
        relations = Column(Text)  # JSON as text
        topics = Column(Text)  # JSON as text
        annotator_email = Column(String(255))
        status = Column(String(50))
        confidence_level = Column(String(20))
        notes = Column(Text)
        decision_method = Column(String(50))
        created_at = Column(DateTime)
        updated_at = Column(DateTime)
        reviewed_at = Column(DateTime)
    
    class Candidate(Base):
        __tablename__ = "candidates"
        id = Column(Integer, primary_key=True, autoincrement=True)
        sentence_id = Column(Integer, ForeignKey("sentences.id"))
        candidate_type = Column(String(20))
        text = Column(String(500))
        label = Column(String(100))
        start_offset = Column(Integer)
        end_offset = Column(Integer)
        confidence = Column(Float)
        head_candidate_id = Column(Integer)
        tail_candidate_id = Column(Integer)
        evidence = Column(Text)
        score = Column(Float)
        keywords = Column(Text)  # JSON as text
        model_name = Column(String(100))
        model_version = Column(String(50))
        generation_method = Column(String(50))
        rule_pattern = Column(String(200))
        created_at = Column(DateTime)
    
    class TriageItem(Base):
        __tablename__ = "triage_items"
        id = Column(Integer, primary_key=True, autoincrement=True)
        item_id = Column(String(255))
        sentence_id = Column(Integer, ForeignKey("sentences.id"))
        candidate_id = Column(Integer, ForeignKey("candidates.id"))
        priority_score = Column(Float)
        priority_level = Column(String(20))
        status = Column(String(50))
        assigned_to = Column(String(255))
        assigned_at = Column(DateTime)
        completed_at = Column(DateTime)
        confidence_score = Column(Float)
        novelty_score = Column(Float)
        impact_score = Column(Float)
        disagreement_score = Column(Float)
        authority_score = Column(Float)


class DocumentExporter:
    """Export documents to JSONL format for RAG storage"""
    
    def __init__(self, db_session: Session, export_base_path: str = "data/content"):
        self.session = db_session
        self.export_base_path = Path(export_base_path)
        self.export_base_path.mkdir(parents=True, exist_ok=True)
    
    def export_document(self, doc_id: str) -> Dict[str, Any]:
        """Export a single document with all associated data"""
        
        # Get document
        doc = self.session.query(Document).filter(Document.doc_id == doc_id).first()
        if not doc:
            raise ValueError(f"Document {doc_id} not found")
        
        # Get all sentences for this document
        sentences = self.session.query(Sentence)\
            .filter(Sentence.document_id == doc.id)\
            .order_by(Sentence.start_offset).all()
        
        # Handle metadata field (could be doc_metadata or document_metadata)
        metadata = getattr(doc, 'document_metadata', None) or getattr(doc, 'doc_metadata', None)
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except (json.JSONDecodeError, TypeError):
                metadata = {}
        
        # Build export structure
        export_data = {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "source": doc.source,
            "full_content": doc.raw_text,
            "content_length": len(doc.raw_text) if doc.raw_text else 0,
            "sentence_count": len(sentences),
            "metadata": {
                "document_metadata": metadata or {},
                "status": getattr(doc, 'status', None),
                "created_at": doc.created_at.isoformat() if doc.created_at else None,
                "updated_at": doc.updated_at.isoformat() if doc.updated_at else None
            },
            "sentences": []
        }
        
        # Process each sentence
        for sentence in sentences:
            sentence_data = self._export_sentence(sentence)
            export_data["sentences"].append(sentence_data)
        
        # Add document-level statistics
        export_data["statistics"] = self._calculate_document_stats(doc, sentences)
        
        return export_data
    
    def _export_sentence(self, sentence: Sentence) -> Dict[str, Any]:
        """Export sentence with all annotations and candidates"""
        
        # Get gold annotations for this sentence
        gold_annotations = self.session.query(GoldAnnotation)\
            .filter(GoldAnnotation.sentence_id == sentence.id).all()
        
        # Get candidates for this sentence
        candidates = self.session.query(Candidate)\
            .filter(Candidate.sentence_id == sentence.id).all()
        
        # Get triage items for this sentence
        triage_items = self.session.query(TriageItem)\
            .filter(TriageItem.sentence_id == sentence.id).all()
        
        sentence_data = {
            "sentence_id": sentence.sent_id,
            "text": sentence.text,
            "position": {
                "start_offset": sentence.start_offset,
                "end_offset": sentence.end_offset,
                "paragraph_id": sentence.paragraph_id
            },
            "annotations": {
                "gold": [self._format_gold_annotation(ann) for ann in gold_annotations],
                "candidates": [self._format_candidate(cand) for cand in candidates]
            },
            "triage_status": {
                "items": [self._format_triage_item(item) for item in triage_items],
                "has_pending": any(item.status == "pending" for item in triage_items),
                "completed_count": sum(1 for item in triage_items if item.status == "completed")
            },
            "created_at": sentence.created_at.isoformat()
        }
        
        return sentence_data
    
    def _format_gold_annotation(self, annotation: GoldAnnotation) -> Dict[str, Any]:
        """Format gold annotation for export"""
        # Handle JSON fields that might be stored as text in SQLite
        def parse_json_field(field_value):
            if isinstance(field_value, str):
                try:
                    return json.loads(field_value)
                except (json.JSONDecodeError, TypeError):
                    return field_value
            return field_value or []
        
        return {
            "annotation_id": str(annotation.id),
            "entities": parse_json_field(annotation.entities),
            "relations": parse_json_field(annotation.relations),
            "topics": parse_json_field(annotation.topics),
            "annotator_email": annotation.annotator_email,
            "status": annotation.status,
            "confidence_level": annotation.confidence_level,
            "notes": annotation.notes,
            "decision_method": annotation.decision_method,
            "created_at": annotation.created_at.isoformat() if annotation.created_at else None,
            "updated_at": annotation.updated_at.isoformat() if annotation.updated_at else None,
            "reviewed_at": annotation.reviewed_at.isoformat() if annotation.reviewed_at else None
        }
    
    def _format_candidate(self, candidate: Candidate) -> Dict[str, Any]:
        """Format candidate annotation for export"""
        candidate_data = {
            "candidate_id": str(candidate.id),
            "type": candidate.candidate_type,
            "label": candidate.label,
            "confidence": candidate.confidence,
            "model_name": candidate.model_name,
            "model_version": candidate.model_version,
            "generation_method": candidate.generation_method,
            "created_at": candidate.created_at.isoformat()
        }
        
        # Add type-specific fields
        if candidate.candidate_type == "entity":
            candidate_data.update({
                "text": candidate.text,
                "start_offset": candidate.start_offset,
                "end_offset": candidate.end_offset
            })
        elif candidate.candidate_type == "relation":
            candidate_data.update({
                "head_candidate_id": str(candidate.head_candidate_id) if candidate.head_candidate_id else None,
                "tail_candidate_id": str(candidate.tail_candidate_id) if candidate.tail_candidate_id else None,
                "evidence": candidate.evidence
            })
        elif candidate.candidate_type == "topic":
            # Handle keywords that might be stored as JSON text
            keywords = candidate.keywords
            if isinstance(keywords, str):
                try:
                    keywords = json.loads(keywords)
                except (json.JSONDecodeError, TypeError):
                    keywords = []
            
            candidate_data.update({
                "score": candidate.score,
                "keywords": keywords or []
            })
        
        if candidate.rule_pattern:
            candidate_data["rule_pattern"] = candidate.rule_pattern
        
        return candidate_data
    
    def _format_triage_item(self, item: TriageItem) -> Dict[str, Any]:
        """Format triage item for export"""
        return {
            "item_id": item.item_id,
            "priority_score": item.priority_score,
            "priority_level": item.priority_level,
            "status": item.status,
            "assigned_to": item.assigned_to,
            "assigned_at": item.assigned_at.isoformat() if item.assigned_at else None,
            "completed_at": item.completed_at.isoformat() if item.completed_at else None,
            "scoring": {
                "confidence_score": item.confidence_score,
                "novelty_score": item.novelty_score,
                "impact_score": item.impact_score,
                "disagreement_score": item.disagreement_score,
                "authority_score": item.authority_score
            }
        }
    
    def _calculate_document_stats(self, doc: Document, sentences: List[Sentence]) -> Dict[str, Any]:
        """Calculate document-level statistics"""
        
        # Count annotations by type
        gold_count = self.session.query(GoldAnnotation)\
            .filter(GoldAnnotation.document_id == doc.id).count()
        
        candidate_count = self.session.query(Candidate)\
            .join(Sentence, Candidate.sentence_id == Sentence.id)\
            .filter(Sentence.document_id == doc.id).count()
        
        # Count by annotation status
        completed_annotations = self.session.query(GoldAnnotation)\
            .filter(and_(
                GoldAnnotation.document_id == doc.id,
                GoldAnnotation.status.in_(["accepted", "completed"])
            )).count()
        
        return {
            "sentence_count": len(sentences),
            "character_count": len(doc.raw_text),
            "word_count": len(doc.raw_text.split()),
            "annotation_counts": {
                "gold_annotations": gold_count,
                "candidate_annotations": candidate_count,
                "completed_annotations": completed_annotations
            },
            "annotation_coverage": completed_annotations / len(sentences) if sentences else 0
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
        
        # Get documents to export
        query = self.session.query(Document)
        if doc_ids:
            query = query.filter(Document.doc_id.in_(doc_ids))
        
        documents = query.all()
        
        # Export each document
        exported_count = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in documents:
                try:
                    export_data = self.export_document(doc.doc_id)
                    f.write(json.dumps(export_data, ensure_ascii=False) + '\n')
                    exported_count += 1
                except Exception as e:
                    print(f"Error exporting document {doc.doc_id}: {e}")
                    continue
        
        print(f"Exported {exported_count} documents to {output_file}")
        return str(output_file)
    
    def get_export_summary(self) -> Dict[str, Any]:
        """Get summary of available documents for export"""
        
        # Count documents by source
        doc_counts = {}
        docs = self.session.query(Document).all()
        
        for doc in docs:
            source = doc.source
            if source not in doc_counts:
                doc_counts[source] = {
                    "count": 0,
                    "with_annotations": 0,
                    "total_sentences": 0
                }
            
            doc_counts[source]["count"] += 1
            
            # Check if has annotations
            has_annotations = self.session.query(GoldAnnotation)\
                .filter(GoldAnnotation.document_id == doc.id).first() is not None
            
            if has_annotations:
                doc_counts[source]["with_annotations"] += 1
            
            # Count sentences
            sentence_count = self.session.query(Sentence)\
                .filter(Sentence.document_id == doc.id).count()
            doc_counts[source]["total_sentences"] += sentence_count
        
        total_docs = len(docs)
        total_with_annotations = sum(info["with_annotations"] for info in doc_counts.values())
        
        return {
            "total_documents": total_docs,
            "documents_with_annotations": total_with_annotations,
            "annotation_coverage": total_with_annotations / total_docs if total_docs > 0 else 0,
            "by_source": doc_counts,
            "export_base_path": str(self.export_base_path)
        }


def create_exporter(db_url: str = None, export_path: str = "data/content") -> DocumentExporter:
    """Factory function to create DocumentExporter with database connection"""
    
    if db_url is None:
        # Use default SQLite for development
        db_url = "sqlite:///data/local/annotations.db"
    
    engine = create_engine(db_url)
    from sqlalchemy.orm import sessionmaker
    Session = sessionmaker(bind=engine)
    session = Session()
    
    return DocumentExporter(session, export_path)


if __name__ == "__main__":
    # CLI interface for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Export documents for RAG storage")
    parser.add_argument("--doc-id", help="Export specific document by ID")
    parser.add_argument("--all", action="store_true", help="Export all documents")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--summary", action="store_true", help="Show export summary")
    parser.add_argument("--db-url", help="Database URL")
    
    args = parser.parse_args()
    
    exporter = create_exporter(args.db_url)
    
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