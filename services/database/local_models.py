"""
SQLite-compatible Database Models for Local Development
Works completely offline without PostgreSQL
"""

import os
from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.sql import func
import json

# Use SQLite for local development
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/local/annotations.db")

# Create engine with SQLite compatibility
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Custom JSON type for SQLite compatibility
class JSONType(JSON):
    """JSON type that works with SQLite"""
    def process_bind_param(self, value, dialect):
        if value is not None:
            value = json.dumps(value)
        return value
        
    def process_result_value(self, value, dialect):
        if value is not None:
            value = json.loads(value)
        return value

# Database Models
class Document(Base):
    """Document model"""
    __tablename__ = "documents"
    
    doc_id = Column(String, primary_key=True)
    source = Column(String, nullable=False)
    title = Column(String)
    pub_date = Column(String)
    raw_text = Column(Text, nullable=False)
    metadata = Column(JSONType, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    sentences = relationship("Sentence", back_populates="document", cascade="all, delete-orphan")
    candidates = relationship("Candidate", back_populates="document")
    annotations = relationship("GoldAnnotation", back_populates="document")

class Sentence(Base):
    """Sentence model"""
    __tablename__ = "sentences"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    sent_id = Column(String, nullable=False)
    doc_id = Column(String, ForeignKey("documents.doc_id"), nullable=False)
    start_offset = Column(Integer)
    end_offset = Column(Integer) 
    text = Column(Text, nullable=False)
    paragraph_id = Column(Integer)
    metadata = Column(JSONType, default={})
    
    # Relationships
    document = relationship("Document", back_populates="sentences")
    candidates = relationship("Candidate", back_populates="sentence")
    annotations = relationship("GoldAnnotation", back_populates="sentence")

class Candidate(Base):
    """Candidate annotation from LLM or rules"""
    __tablename__ = "candidates"
    
    candidate_id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String, ForeignKey("documents.doc_id"))
    sent_id = Column(String)
    sentence_id = Column(Integer, ForeignKey("sentences.id"))
    source = Column(String)  # 'llm', 'rule', 'manual'
    entity_data = Column(JSONType, default=[])
    relation_data = Column(JSONType, default=[])
    topic_data = Column(JSONType, default=[])
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="candidates")
    sentence = relationship("Sentence", back_populates="candidates")
    annotations = relationship("GoldAnnotation", back_populates="candidate")
    triage_items = relationship("TriageItem", back_populates="candidate")

class GoldAnnotation(Base):
    """Gold standard annotation"""
    __tablename__ = "gold_annotations"
    
    annotation_id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String, ForeignKey("documents.doc_id"))
    sent_id = Column(String)
    sentence_id = Column(Integer, ForeignKey("sentences.id"))
    candidate_id = Column(Integer, ForeignKey("candidates.candidate_id"))
    annotation_type = Column(String)  # 'entity', 'relation', 'topic'
    annotation_data = Column(JSONType, nullable=False)
    annotator = Column(String)
    confidence = Column(Float)
    decision = Column(String)  # 'accept', 'reject', 'modify'
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="annotations")
    sentence = relationship("Sentence", back_populates="annotations")
    candidate = relationship("Candidate", back_populates="annotations")

class TriageItem(Base):
    """Triage queue item"""
    __tablename__ = "triage_queue"
    
    item_id = Column(Integer, primary_key=True, autoincrement=True)
    doc_id = Column(String)
    sent_id = Column(String)
    candidate_id = Column(Integer, ForeignKey("candidates.candidate_id"))
    priority_score = Column(Float)
    priority_level = Column(String)
    status = Column(String, default='pending')  # 'pending', 'in_review', 'completed'
    assigned_to = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    # Relationships
    candidate = relationship("Candidate", back_populates="triage_items")

class User(Base):
    """User model for simple authentication"""
    __tablename__ = "users"
    
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False)
    token = Column(String, unique=True, nullable=False)
    role = Column(String)  # 'admin', 'annotator', 'reviewer'
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    stats = relationship("AnnotationStat", back_populates="user")

class AnnotationStat(Base):
    """Annotation statistics"""
    __tablename__ = "annotation_stats"
    
    stat_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    date = Column(String)
    annotations_count = Column(Integer, default=0)
    accept_count = Column(Integer, default=0)
    reject_count = Column(Integer, default=0)
    modify_count = Column(Integer, default=0)
    avg_confidence = Column(Float)
    
    # Relationships
    user = relationship("User", back_populates="stats")

# Database utilities
def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database with tables"""
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created")
    
def reset_db():
    """Reset database (careful!)"""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    print("✓ Database reset complete")

# Cache implementation (replaces Redis)
class LocalCache:
    """Simple in-memory cache to replace Redis"""
    
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            self.access_times[key] = datetime.utcnow()
            return self.cache[key]
        return None
        
    def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache"""
        # Evict oldest if cache is full
        if len(self.cache) >= self.max_size:
            oldest = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest]
            del self.access_times[oldest]
            
        self.cache[key] = value
        self.access_times[key] = datetime.utcnow()
        
    def delete(self, key: str):
        """Delete from cache"""
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]
            
    def clear(self):
        """Clear entire cache"""
        self.cache.clear()
        self.access_times.clear()

# Global cache instance
cache = LocalCache()

# Queue implementation (replaces Redis queue)
class LocalQueue:
    """Simple file-based queue to replace Redis"""
    
    def __init__(self, queue_dir="./data/local/queue"):
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        
    def push(self, item: Dict) -> str:
        """Add item to queue"""
        item_id = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        file_path = self.queue_dir / f"{item_id}.json"
        
        with open(file_path, 'w') as f:
            json.dump(item, f)
            
        return item_id
        
    def pop(self) -> Optional[Dict]:
        """Get next item from queue"""
        files = sorted(self.queue_dir.glob("*.json"))
        
        if files:
            file_path = files[0]
            with open(file_path, 'r') as f:
                item = json.load(f)
            file_path.unlink()  # Delete file
            return item
            
        return None
        
    def size(self) -> int:
        """Get queue size"""
        return len(list(self.queue_dir.glob("*.json")))
        
    def clear(self):
        """Clear queue"""
        for file_path in self.queue_dir.glob("*.json"):
            file_path.unlink()

# Global queue instance
from pathlib import Path
queue = LocalQueue()

if __name__ == "__main__":
    # Initialize database when run directly
    init_db()
    print("Local database initialized successfully!")