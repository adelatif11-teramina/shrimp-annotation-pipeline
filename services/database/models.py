"""
Database Models for Annotation Pipeline

SQLAlchemy ORM models for all pipeline data structures.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import json

from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()

class Document(Base):
    """Document model for ingested documents"""
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_id = Column(String(255), unique=True, nullable=False, index=True)
    source = Column(String(50), nullable=False)  # paper, report, hatchery_log
    title = Column(String(500))
    pub_date = Column(DateTime)
    raw_text = Column(Text, nullable=False)
    document_metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    sentences = relationship("Sentence", back_populates="document", cascade="all, delete-orphan")
    annotations = relationship("GoldAnnotation", back_populates="document")

class Sentence(Base):
    """Sentence model for document segmentation"""
    __tablename__ = "sentences"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sent_id = Column(String(100), nullable=False, index=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    start_offset = Column(Integer, nullable=False)
    end_offset = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    paragraph_id = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="sentences")
    candidates = relationship("Candidate", back_populates="sentence")
    annotations = relationship("GoldAnnotation", back_populates="sentence")
    triage_items = relationship("TriageItem", back_populates="sentence")

class Candidate(Base):
    """LLM/Rule-based candidate annotations"""
    __tablename__ = "candidates"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sentence_id = Column(UUID(as_uuid=True), ForeignKey("sentences.id"), nullable=False)
    candidate_type = Column(String(20), nullable=False)  # entity, relation, topic
    
    # Candidate data
    text = Column(String(500))  # For entities
    label = Column(String(100), nullable=False)
    start_offset = Column(Integer)  # For entities
    end_offset = Column(Integer)   # For entities
    confidence = Column(Float, nullable=False)
    
    # For relations
    head_candidate_id = Column(UUID(as_uuid=True), ForeignKey("candidates.id"))
    tail_candidate_id = Column(UUID(as_uuid=True), ForeignKey("candidates.id"))
    evidence = Column(Text)
    
    # For topics
    score = Column(Float)
    keywords = Column(JSON)  # List of keywords
    
    # Generation metadata
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50))
    generation_method = Column(String(50), nullable=False)  # llm, rule, hybrid
    rule_pattern = Column(String(200))  # For rule-based candidates
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    sentence = relationship("Sentence", back_populates="candidates")
    head_relation = relationship("Candidate", remote_side=[id], foreign_keys=[head_candidate_id])
    tail_relation = relationship("Candidate", remote_side=[id], foreign_keys=[tail_candidate_id])
    triage_items = relationship("TriageItem", back_populates="candidate")

class GoldAnnotation(Base):
    """Gold standard annotations from human annotators"""
    __tablename__ = "gold_annotations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    sentence_id = Column(UUID(as_uuid=True), ForeignKey("sentences.id"), nullable=False)
    
    # Annotation data (stored as JSON for flexibility)
    entities = Column(JSON, default=[])
    relations = Column(JSON, default=[])
    topics = Column(JSON, default=[])
    
    # Annotation metadata
    annotator_email = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False)  # accepted, modified, rejected, needs_review
    confidence_level = Column(String(20))  # high, medium, low
    notes = Column(Text)
    
    # Decision tracking
    decision_method = Column(String(50))  # manual, auto_accept, bulk_accept
    source_candidate_id = Column(UUID(as_uuid=True), ForeignKey("candidates.id"))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    reviewed_at = Column(DateTime)
    
    # Relationships
    document = relationship("Document", back_populates="annotations")
    sentence = relationship("Sentence", back_populates="annotations")
    source_candidate = relationship("Candidate")

class TriageItem(Base):
    """Triage queue items for prioritization"""
    __tablename__ = "triage_items"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    item_id = Column(String(255), unique=True, nullable=False, index=True)
    sentence_id = Column(UUID(as_uuid=True), ForeignKey("sentences.id"), nullable=False)
    candidate_id = Column(UUID(as_uuid=True), ForeignKey("candidates.id"), nullable=False)
    
    # Priority scoring
    priority_score = Column(Float, nullable=False)
    priority_level = Column(String(20), nullable=False)  # CRITICAL, HIGH, MEDIUM, LOW, MINIMAL
    
    # Scoring components
    confidence_score = Column(Float, default=0.0)
    novelty_score = Column(Float, default=0.0)
    impact_score = Column(Float, default=0.0)
    disagreement_score = Column(Float, default=0.0)
    authority_score = Column(Float, default=0.0)
    
    # Status tracking
    status = Column(String(50), default="pending")  # pending, in_review, completed, skipped
    assigned_to = Column(String(255))  # annotator email
    assigned_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    sentence = relationship("Sentence", back_populates="triage_items")
    candidate = relationship("Candidate", back_populates="triage_items")

class AnnotationEvent(Base):
    """Tracking events for metrics and monitoring"""
    __tablename__ = "annotation_events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_id = Column(String(255), unique=True, nullable=False)
    event_type = Column(String(50), nullable=False)  # started, completed, modified, rejected
    
    # Related items
    triage_item_id = Column(UUID(as_uuid=True), ForeignKey("triage_items.id"))
    annotator_email = Column(String(255), nullable=False)
    
    # Performance metrics
    processing_time = Column(Float)  # seconds
    decision = Column(String(50))
    
    # Event metadata
    document_metadata = Column(JSON, default={})
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

class AutoAcceptRule(Base):
    """Auto-accept rules configuration"""
    __tablename__ = "auto_accept_rules"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    rule_id = Column(String(100), unique=True, nullable=False)
    rule_name = Column(String(200), nullable=False)
    
    # Rule conditions
    entity_types = Column(JSON)  # List of applicable entity types
    relation_types = Column(JSON)  # List of applicable relation types
    min_confidence = Column(Float, default=0.95)
    min_agreement = Column(Float, default=0.9)
    source_authority_min = Column(Float, default=0.8)
    requires_rule_support = Column(Boolean, default=True)
    max_novelty = Column(Float, default=0.3)
    
    # Rule status
    enabled = Column(Boolean, default=True)
    
    # Performance tracking
    auto_accepted = Column(Integer, default=0)
    false_positives = Column(Integer, default=0)
    precision = Column(Float, default=1.0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class AutoAcceptDecision(Base):
    """Auto-accept decisions log"""
    __tablename__ = "auto_accept_decisions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    triage_item_id = Column(UUID(as_uuid=True), ForeignKey("triage_items.id"), nullable=False)
    rule_id = Column(UUID(as_uuid=True), ForeignKey("auto_accept_rules.id"))
    
    decision = Column(String(50), nullable=False)  # auto_accept, human_review, auto_reject
    confidence = Column(Float, nullable=False)
    reasoning = Column(Text)
    
    # Validation tracking
    was_correct = Column(Boolean)  # Set after human validation
    validated_by = Column(String(255))
    validated_at = Column(DateTime)
    
    timestamp = Column(DateTime, default=datetime.utcnow)

class ModelTrainingRun(Base):
    """Model training and retraining tracking"""
    __tablename__ = "model_training_runs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(String(100), unique=True, nullable=False)
    
    # Training metadata
    model_type = Column(String(50), nullable=False)  # scibert_ner, relation_classifier, topic_classifier, auto_accept_classifier
    training_data_version = Column(String(50))
    gold_annotation_count = Column(Integer)
    
    # Automated retraining fields
    trigger_conditions = Column(JSON)  # List of trigger conditions that caused this training
    confidence_score = Column(Float)  # Confidence in the retraining decision
    data_snapshot = Column(JSON)  # Snapshot of data metrics at training time
    triggered_by = Column(String(255))  # User or system that triggered training
    
    # Training configuration
    hyperparameters = Column(JSON, default={})
    training_config = Column(JSON, default={})
    
    # Performance metrics
    train_metrics = Column(JSON, default={})
    val_metrics = Column(JSON, default={})
    test_metrics = Column(JSON, default={})
    
    # Status tracking
    status = Column(String(50), nullable=False, default="initiated")  # initiated, queued, running, completed, failed
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    error_message = Column(Text)
    
    # Model artifacts
    model_path = Column(String(500))
    model_version = Column(String(50))
    
    created_at = Column(DateTime, default=datetime.utcnow)

# Database utility functions
class DatabaseManager:
    """Database management utilities"""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create_document(self, doc_id: str, source: str, title: str, 
                       raw_text: str, sentences: List[Dict], 
                       metadata: Dict = None) -> Document:
        """Create document with sentences"""
        doc = Document(
            doc_id=doc_id,
            source=source,
            title=title,
            raw_text=raw_text,
            metadata=metadata or {}
        )
        
        self.session.add(doc)
        self.session.flush()  # Get the document ID
        
        # Add sentences
        for sent_data in sentences:
            sentence = Sentence(
                sent_id=sent_data["sent_id"],
                document_id=doc.id,
                start_offset=sent_data["start"],
                end_offset=sent_data["end"],
                text=sent_data["text"],
                paragraph_id=sent_data.get("paragraph_id")
            )
            self.session.add(sentence)
        
        self.session.commit()
        return doc
    
    def create_candidates(self, sentence_id: str, candidates_data: Dict) -> List[Candidate]:
        """Create candidates for a sentence"""
        candidates = []
        
        # Entity candidates
        for entity_data in candidates_data.get("entities", []):
            candidate = Candidate(
                sentence_id=sentence_id,
                candidate_type="entity",
                text=entity_data["text"],
                label=entity_data["label"],
                start_offset=entity_data["start"],
                end_offset=entity_data["end"],
                confidence=entity_data["confidence"],
                model_name=entity_data.get("model", "unknown"),
                generation_method=entity_data.get("method", "llm")
            )
            candidates.append(candidate)
            self.session.add(candidate)
        
        self.session.flush()  # Get candidate IDs for relations
        
        # Relation candidates
        entity_candidates = {i: candidates[i] for i in range(len(candidates))}
        
        for relation_data in candidates_data.get("relations", []):
            head_cid = relation_data.get("head_cid", 0)
            tail_cid = relation_data.get("tail_cid", 0)
            
            if head_cid in entity_candidates and tail_cid in entity_candidates:
                candidate = Candidate(
                    sentence_id=sentence_id,
                    candidate_type="relation",
                    label=relation_data["label"],
                    confidence=relation_data["confidence"],
                    head_candidate_id=entity_candidates[head_cid].id,
                    tail_candidate_id=entity_candidates[tail_cid].id,
                    evidence=relation_data.get("evidence"),
                    model_name=relation_data.get("model", "unknown"),
                    generation_method=relation_data.get("method", "llm")
                )
                candidates.append(candidate)
                self.session.add(candidate)
        
        # Topic candidates
        for topic_data in candidates_data.get("topics", []):
            candidate = Candidate(
                sentence_id=sentence_id,
                candidate_type="topic",
                label=topic_data["topic_id"],
                confidence=topic_data.get("score", 0.5),
                score=topic_data.get("score", 0.5),
                keywords=topic_data.get("keywords", []),
                model_name=topic_data.get("model", "unknown"),
                generation_method=topic_data.get("method", "llm")
            )
            candidates.append(candidate)
            self.session.add(candidate)
        
        self.session.commit()
        return candidates
    
    def create_gold_annotation(self, document_id: str, sentence_id: str,
                              entities: List[Dict], relations: List[Dict],
                              topics: List[Dict], annotator_email: str,
                              status: str = "accepted") -> GoldAnnotation:
        """Create gold annotation"""
        annotation = GoldAnnotation(
            document_id=document_id,
            sentence_id=sentence_id,
            entities=entities,
            relations=relations,
            topics=topics,
            annotator_email=annotator_email,
            status=status
        )
        
        self.session.add(annotation)
        self.session.commit()
        return annotation
    
    def get_documents_for_annotation(self, limit: int = 10) -> List[Document]:
        """Get documents that need annotation"""
        return self.session.query(Document)\
            .filter(~Document.annotations.any())\
            .limit(limit).all()
    
    def get_triage_queue(self, limit: int = 10, 
                        annotator: str = None) -> List[TriageItem]:
        """Get items from triage queue"""
        query = self.session.query(TriageItem)\
            .filter(TriageItem.status == "pending")\
            .order_by(TriageItem.priority_score.desc())
        
        if annotator:
            query = query.filter(TriageItem.assigned_to == annotator)
        
        return query.limit(limit).all()
    
    def get_annotation_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get annotation statistics"""
        from sqlalchemy import func
        from datetime import timedelta
        
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        # Count annotations by status
        status_counts = self.session.query(
            GoldAnnotation.status,
            func.count(GoldAnnotation.id)
        ).filter(GoldAnnotation.created_at >= cutoff)\
         .group_by(GoldAnnotation.status).all()
        
        # Count by annotator
        annotator_counts = self.session.query(
            GoldAnnotation.annotator_email,
            func.count(GoldAnnotation.id)
        ).filter(GoldAnnotation.created_at >= cutoff)\
         .group_by(GoldAnnotation.annotator_email).all()
        
        return {
            "period_days": days,
            "status_distribution": dict(status_counts),
            "annotator_distribution": dict(annotator_counts),
            "total_annotations": sum(count for _, count in status_counts)
        }