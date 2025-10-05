#!/usr/bin/env python3
"""
Railway Database Setup Script

Sets up PostgreSQL database for Railway deployment.
Creates tables and initial data for production environment.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from services.database.models import Base, AutoAcceptRule

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_database_url():
    """Get database URL for Railway deployment"""
    if os.getenv('RAILWAY_ENVIRONMENT'):
        # Railway PostgreSQL
        db_url = os.getenv('DATABASE_URL')
        if db_url and db_url.startswith('postgres://'):
            # Railway sometimes uses postgres:// but SQLAlchemy needs postgresql://
            db_url = db_url.replace('postgres://', 'postgresql://', 1)
        return db_url
    else:
        # Local development
        user = os.getenv('POSTGRES_USER', 'postgres')
        password = os.getenv('POSTGRES_PASSWORD', 'postgres')
        host = os.getenv('POSTGRES_HOST', 'localhost')
        port = os.getenv('POSTGRES_PORT', '5432')
        db = os.getenv('POSTGRES_DB', 'shrimp_annotation')
        return f"postgresql://{user}:{password}@{host}:{port}/{db}"

def test_connection(engine):
    """Test database connection"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            logger.info(f"✅ Connected to PostgreSQL: {version[:50]}...")
            return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False

def create_tables(engine):
    """Create all tables"""
    try:
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Tables created successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Table creation failed: {e}")
        return False

def create_initial_data(engine):
    """Create initial configuration data"""
    from sqlalchemy.orm import sessionmaker
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        # Check if auto-accept rules already exist
        existing_rules = session.query(AutoAcceptRule).count()
        if existing_rules > 0:
            logger.info(f"Auto-accept rules already exist ({existing_rules} rules)")
            return True
        
        # Create default auto-accept rules
        default_rules = [
            {
                "rule_id": "high_confidence_entities",
                "rule_name": "High Confidence Entity Auto-Accept",
                "entity_types": ["SPECIES", "PATHOGEN", "CHEMICAL", "EQUIPMENT"],
                "relation_types": [],
                "min_confidence": 0.95,
                "min_agreement": 0.9,
                "source_authority_min": 0.8,
                "requires_rule_support": True,
                "max_novelty": 0.2,
                "enabled": True
            },
            {
                "rule_id": "common_relations",
                "rule_name": "Common Relations Auto-Accept",
                "entity_types": [],
                "relation_types": ["causes", "treated_with", "located_in"],
                "min_confidence": 0.90,
                "min_agreement": 0.85,
                "source_authority_min": 0.7,
                "requires_rule_support": True,
                "max_novelty": 0.3,
                "enabled": True
            }
        ]
        
        for rule_data in default_rules:
            rule = AutoAcceptRule(**rule_data)
            session.add(rule)
        
        session.commit()
        logger.info(f"✅ Created {len(default_rules)} default auto-accept rules")
        return True
        
    except Exception as e:
        session.rollback()
        logger.error(f"❌ Initial data creation failed: {e}")
        return False
    finally:
        session.close()

def setup_indexes(engine):
    """Create additional indexes for performance"""
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_sentences_document_id ON sentences(document_id);",
        "CREATE INDEX IF NOT EXISTS idx_candidates_sentence_id ON candidates(sentence_id);",
        "CREATE INDEX IF NOT EXISTS idx_candidates_type_confidence ON candidates(candidate_type, confidence);",
        "CREATE INDEX IF NOT EXISTS idx_triage_status_priority ON triage_items(status, priority_score DESC);",
        "CREATE INDEX IF NOT EXISTS idx_gold_annotations_annotator ON gold_annotations(annotator_email);",
        "CREATE INDEX IF NOT EXISTS idx_gold_annotations_created_at ON gold_annotations(created_at);",
        "CREATE INDEX IF NOT EXISTS idx_annotation_events_timestamp ON annotation_events(timestamp);"
    ]
    
    try:
        with engine.connect() as conn:
            for index_sql in indexes:
                conn.execute(text(index_sql))
                conn.commit()
        logger.info("✅ Performance indexes created")
        return True
    except Exception as e:
        logger.error(f"❌ Index creation failed: {e}")
        return False

def main():
    """Main setup function"""
    logger.info("🚀 Starting Railway database setup...")
    
    # Get database URL
    database_url = get_database_url()
    if not database_url:
        logger.error("❌ No database URL configured")
        return False
    
    logger.info(f"Database URL: {database_url[:50]}...")
    
    # Create engine
    try:
        engine = create_engine(
            database_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False
        )
    except Exception as e:
        logger.error(f"❌ Failed to create database engine: {e}")
        return False
    
    # Test connection
    if not test_connection(engine):
        return False
    
    # Create tables
    if not create_tables(engine):
        return False
    
    # Create indexes
    if not setup_indexes(engine):
        return False
    
    # Create initial data
    if not create_initial_data(engine):
        return False
    
    logger.info("🎉 Railway database setup completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)