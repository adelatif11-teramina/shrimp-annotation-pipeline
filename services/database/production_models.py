"""
Production SQLAlchemy Models for Annotation Pipeline
Clean, production-ready models for Alembic migrations
"""

from sqlalchemy import Column, Integer, String, Text, Boolean, Float, DateTime, ForeignKey, JSON, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()

class User(Base):
    """User model for authentication and authorization"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True)
    hashed_password = Column(String(255))
    role = Column(String(20), nullable=False, default="annotator")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_login = Column(DateTime(timezone=True))
    
    __table_args__ = (
        Index("idx_user_active_role", "is_active", "role"),
    )

class Document(Base):
    """Document model for storing raw documents"""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    doc_id = Column(String(100), unique=True, index=True, nullable=False)
    title = Column(String(500))
    source = Column(String(100), nullable=False, index=True)
    raw_text = Column(Text)
    doc_metadata = Column(JSON)
    status = Column(String(20), default="active", index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index("idx_document_source_status", "source", "status"),
        Index("idx_document_created", "created_at"),
    )

class Sentence(Base):
    """Sentence model for individual text segments"""
    __tablename__ = "sentences"
    
    id = Column(Integer, primary_key=True, index=True)
    sent_id = Column(String(100), nullable=False, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    text = Column(Text, nullable=False)
    start_offset = Column(Integer)
    end_offset = Column(Integer)
    paragraph_id = Column(String(50))
    processed = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index("idx_sentence_doc_processed", "document_id", "processed"),
        Index("idx_sentence_doc_sent", "document_id", "sent_id"),
    )

class Candidate(Base):
    """Candidate annotations from LLM or rules"""
    __tablename__ = "candidates"
    
    id = Column(Integer, primary_key=True, index=True)
    sentence_id = Column(Integer, ForeignKey("sentences.id"), nullable=False)
    source = Column(String(50), nullable=False, index=True)
    candidate_type = Column(String(20), nullable=False, index=True)
    
    # Candidate data
    entities = Column(JSON)
    relations = Column(JSON)
    topics = Column(JSON)
    
    # Scoring
    confidence = Column(Float, index=True)
    priority_score = Column(Float, index=True)
    
    # Processing status
    status = Column(String(20), default="pending", index=True)
    processed = Column(Boolean, default=False, index=True)
    
    # Metadata
    model_info = Column(JSON)
    processing_time = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index("idx_candidate_status_priority", "status", "priority_score"),
        Index("idx_candidate_source_type", "source", "candidate_type"),
        Index("idx_candidate_confidence", "confidence"),
    )

class Annotation(Base):
    """Human annotations (gold standard)"""
    __tablename__ = "annotations"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    sentence_id = Column(Integer, ForeignKey("sentences.id"), nullable=False)
    candidate_id = Column(Integer, ForeignKey("candidates.id"))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Annotation content
    annotation_type = Column(String(20), nullable=False, index=True)
    entities = Column(JSON)
    relations = Column(JSON)
    topics = Column(JSON)
    
    # Decision and quality
    decision = Column(String(20), nullable=False, index=True)
    confidence = Column(Float)
    quality_score = Column(Float)
    
    # Metadata
    notes = Column(Text)
    time_spent = Column(Float)
    revision_count = Column(Integer, default=0)
    
    # Version control
    version = Column(Integer, default=1)
    is_latest = Column(Boolean, default=True, index=True)
    superseded_by = Column(Integer, ForeignKey("annotations.id"))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index("idx_annotation_user_decision", "user_id", "decision"),
        Index("idx_annotation_doc_sent", "document_id", "sentence_id"),
        Index("idx_annotation_latest", "is_latest", "decision"),
    )