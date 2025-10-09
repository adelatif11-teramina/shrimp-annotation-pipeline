"""
Complete SQLAlchemy database setup for local development
This replaces the broken local_models.py with a working implementation
"""

import os
import json
import hashlib
from datetime import datetime, date
from typing import Optional, Dict, Any, List
from pathlib import Path

from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, Boolean, ForeignKey, JSON, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/local/annotations.db")

# Create engine with proper configuration
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False  # Set to True for SQL debugging
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# SQLite JSON type
class SQLiteJSON(JSON):
    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value)
        return value
        
    def process_result_value(self, value, dialect):
        if value is not None:
            return json.loads(value)
        return value

# Database Models
class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String, unique=True, nullable=False, index=True)
    source = Column(String, nullable=False)
    title = Column(String)
    pub_date = Column(String)
    raw_text = Column(Text, nullable=False)
    document_metadata = Column(SQLiteJSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    sentences = relationship("Sentence", back_populates="document", cascade="all, delete-orphan")
    candidates = relationship("Candidate", back_populates="document", cascade="all, delete-orphan")
    annotations = relationship("GoldAnnotation", back_populates="document")

class Sentence(Base):
    __tablename__ = "sentences"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    sent_id = Column(String, nullable=False)
    doc_id = Column(String, ForeignKey("documents.doc_id"), nullable=False)
    start_offset = Column(Integer)
    end_offset = Column(Integer)
    text = Column(Text, nullable=False)
    paragraph_id = Column(Integer)
    sentence_metadata = Column(SQLiteJSON, default=dict)
    processed = Column(Boolean, default=False)
    
    # Relationships
    document = relationship("Document", back_populates="sentences")
    candidates = relationship("Candidate", back_populates="sentence", cascade="all, delete-orphan")

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False)
    token = Column(String, unique=True, nullable=False)
    role = Column(String, nullable=False)  # admin, annotator, reviewer
    email = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    annotations = relationship("GoldAnnotation", back_populates="user")
    triage_assignments = relationship("TriageItem", back_populates="assigned_user")

class Candidate(Base):
    __tablename__ = "candidates"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String, ForeignKey("documents.doc_id"), nullable=False)
    sent_id = Column(String, nullable=False)
    sentence_id = Column(Integer, ForeignKey("sentences.id"))
    source = Column(String, nullable=False)  # 'llm', 'rule', 'manual'
    candidate_type = Column(String, nullable=False)  # 'entity', 'relation', 'topic'
    
    # Annotation data
    entities = Column(SQLiteJSON, default=list)
    relations = Column(SQLiteJSON, default=list)
    topics = Column(SQLiteJSON, default=list)
    
    # Scoring
    confidence = Column(Float, default=0.0)
    priority_score = Column(Float, default=0.0)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    processed = Column(Boolean, default=False)
    
    # Relationships
    document = relationship("Document", back_populates="candidates")
    sentence = relationship("Sentence", back_populates="candidates")
    triage_items = relationship("TriageItem", back_populates="candidate", cascade="all, delete-orphan")
    annotations = relationship("GoldAnnotation", back_populates="candidate")

class TriageItem(Base):
    __tablename__ = "triage_queue"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id"), nullable=False)
    
    # Priority and scoring
    priority_score = Column(Float, nullable=False)
    priority_level = Column(String, nullable=False)  # critical, high, medium, low
    
    # Status tracking
    status = Column(String, default='pending')  # pending, in_review, completed, skipped
    assigned_to = Column(Integer, ForeignKey("users.id"))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    assigned_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Relationships
    candidate = relationship("Candidate", back_populates="triage_items")
    assigned_user = relationship("User", back_populates="triage_assignments")

class GoldAnnotation(Base):
    __tablename__ = "gold_annotations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String, ForeignKey("documents.doc_id"), nullable=False)
    sent_id = Column(String, nullable=False)
    candidate_id = Column(Integer, ForeignKey("candidates.id"))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # Annotation content
    annotation_type = Column(String, nullable=False)  # 'entity', 'relation', 'topic', 'combined'
    entities = Column(SQLiteJSON, default=list)
    relations = Column(SQLiteJSON, default=list)
    topics = Column(SQLiteJSON, default=list)
    
    # Decision tracking
    decision = Column(String, nullable=False)  # 'accept', 'reject', 'modify'
    confidence = Column(Float, default=0.0)
    notes = Column(Text)
    
    # Quality metrics
    time_spent = Column(Float)  # seconds
    difficulty = Column(String)  # easy, medium, hard
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="annotations")
    user = relationship("User", back_populates="annotations")
    candidate = relationship("Candidate", back_populates="annotations")

class AnnotationSession(Base):
    __tablename__ = "annotation_sessions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime)
    
    # Session metrics
    items_completed = Column(Integer, default=0)
    items_skipped = Column(Integer, default=0)
    average_time_per_item = Column(Float, default=0.0)
    
    # Session metadata
    session_type = Column(String, default='annotation')  # annotation, review, training
    notes = Column(Text)

class SystemStats(Base):
    __tablename__ = "system_stats"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String, nullable=False)  # YYYY-MM-DD
    
    # Document stats
    total_documents = Column(Integer, default=0)
    total_sentences = Column(Integer, default=0)
    processed_sentences = Column(Integer, default=0)
    
    # Annotation stats
    total_candidates = Column(Integer, default=0)
    total_annotations = Column(Integer, default=0)
    accepted_annotations = Column(Integer, default=0)
    rejected_annotations = Column(Integer, default=0)
    
    # Queue stats
    queue_size = Column(Integer, default=0)
    average_priority = Column(Float, default=0.0)
    
    # Performance stats
    annotations_per_hour = Column(Float, default=0.0)
    average_confidence = Column(Float, default=0.0)
    
    created_at = Column(DateTime, default=datetime.utcnow)

# Database utility functions
def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """Create all database tables"""
    # Ensure data directory exists
    db_path = Path(DATABASE_URL.replace("sqlite:///", ""))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created successfully")

def reset_database():
    """Reset database (careful!)"""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    print("✓ Database reset complete")

def init_database():
    """Initialize database with tables and default data"""
    create_tables()
    
    # Create session
    db = SessionLocal()
    
    try:
        # Create default users if they don't exist
        default_users = [
            {"username": "admin", "token": "local-admin-2024", "role": "admin", "email": "admin@local.dev"},
            {"username": "annotator1", "token": "anno-team-001", "role": "annotator", "email": "ann1@local.dev"},
            {"username": "annotator2", "token": "anno-team-002", "role": "annotator", "email": "ann2@local.dev"},
            {"username": "reviewer", "token": "review-lead-003", "role": "reviewer", "email": "review@local.dev"},
        ]
        
        for user_data in default_users:
            existing_user = db.query(User).filter(User.username == user_data["username"]).first()
            if not existing_user:
                user = User(**user_data)
                db.add(user)
        
        # Create system stats entry for today
        today = date.today().isoformat()
        existing_stats = db.query(SystemStats).filter(SystemStats.date == today).first()
        if not existing_stats:
            stats = SystemStats(date=today)
            db.add(stats)
        
        db.commit()
        print("✓ Default users and data created")
        
    except Exception as e:
        db.rollback()
        print(f"⚠ Error initializing database: {e}")
    finally:
        db.close()

def get_session():
    """Get a database session"""
    return SessionLocal()

if __name__ == "__main__":
    init_database()