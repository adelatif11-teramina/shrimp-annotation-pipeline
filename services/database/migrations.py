"""
Database Migration System

Handles database schema creation, updates, and data migration.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
from alembic.config import Config
from alembic import command
from alembic.script import ScriptDirectory
from alembic.runtime.migration import MigrationContext

from .models import Base, Document, Sentence, Candidate, GoldAnnotation, TriageItem, AnnotationEvent, AutoAcceptRule, AutoAcceptDecision, ModelTrainingRun

logger = logging.getLogger(__name__)

class DatabaseMigrator:
    """Database migration and setup manager"""
    
    def __init__(self, database_url: str):
        """
        Initialize migrator.
        
        Args:
            database_url: SQLAlchemy database URL
        """
        self.database_url = database_url
        self.engine = create_engine(database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Setup Alembic
        self.alembic_cfg = Config()
        self.alembic_cfg.set_main_option("script_location", str(Path(__file__).parent / "alembic"))
        self.alembic_cfg.set_main_option("sqlalchemy.url", database_url)
        
        logger.info(f"Initialized database migrator for {database_url}")
    
    def init_alembic(self):
        """Initialize Alembic migration environment"""
        alembic_dir = Path(__file__).parent / "alembic"
        if not alembic_dir.exists():
            command.init(self.alembic_cfg, str(alembic_dir))
            logger.info("Initialized Alembic migration directory")
    
    def create_all_tables(self):
        """Create all tables from models"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Created all database tables")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def drop_all_tables(self):
        """Drop all tables (use with caution!)"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Dropped all database tables")
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            raise
    
    def get_current_revision(self) -> Optional[str]:
        """Get current Alembic revision"""
        try:
            with self.engine.connect() as connection:
                context = MigrationContext.configure(connection)
                return context.get_current_revision()
        except Exception as e:
            logger.error(f"Failed to get current revision: {e}")
            return None
    
    def get_available_revisions(self) -> List[str]:
        """Get list of available migration revisions"""
        try:
            script_dir = ScriptDirectory.from_config(self.alembic_cfg)
            revisions = [rev.revision for rev in script_dir.walk_revisions()]
            return list(reversed(revisions))  # Order from oldest to newest
        except Exception as e:
            logger.error(f"Failed to get available revisions: {e}")
            return []
    
    def upgrade_to_head(self):
        """Upgrade database to latest revision"""
        try:
            command.upgrade(self.alembic_cfg, "head")
            logger.info("Upgraded database to latest revision")
        except Exception as e:
            logger.error(f"Failed to upgrade database: {e}")
            raise
    
    def downgrade_to_revision(self, revision: str):
        """Downgrade database to specific revision"""
        try:
            command.downgrade(self.alembic_cfg, revision)
            logger.info(f"Downgraded database to revision {revision}")
        except Exception as e:
            logger.error(f"Failed to downgrade database: {e}")
            raise
    
    def create_migration(self, message: str):
        """Create new migration file"""
        try:
            command.revision(self.alembic_cfg, message=message, autogenerate=True)
            logger.info(f"Created migration: {message}")
        except Exception as e:
            logger.error(f"Failed to create migration: {e}")
            raise
    
    def check_database_exists(self) -> bool:
        """Check if database exists and is accessible"""
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database not accessible: {e}")
            return False
    
    def get_table_info(self) -> Dict[str, Dict]:
        """Get information about existing tables"""
        inspector = inspect(self.engine)
        tables = {}
        
        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            indexes = inspector.get_indexes(table_name)
            foreign_keys = inspector.get_foreign_keys(table_name)
            
            tables[table_name] = {
                "columns": {col["name"]: col["type"] for col in columns},
                "indexes": [idx["name"] for idx in indexes],
                "foreign_keys": [fk["name"] for fk in foreign_keys]
            }
        
        return tables
    
    def setup_fresh_database(self):
        """Setup fresh database with all tables and initial data"""
        logger.info("Setting up fresh database...")
        
        # Create all tables
        self.create_all_tables()
        
        # Insert initial data
        self._insert_initial_data()
        
        # Mark as current Alembic revision
        try:
            command.stamp(self.alembic_cfg, "head")
            logger.info("Stamped database with current Alembic revision")
        except Exception as e:
            logger.warning(f"Failed to stamp Alembic revision: {e}")
        
        logger.info("Fresh database setup completed")
    
    def _insert_initial_data(self):
        """Insert initial reference data"""
        session = self.SessionLocal()
        
        try:
            # Insert default auto-accept rules
            default_rules = [
                {
                    "rule_id": "species_high_conf",
                    "rule_name": "High-confidence species annotations",
                    "entity_types": ["SPECIES"],
                    "min_confidence": 0.95,
                    "min_agreement": 0.9,
                    "requires_rule_support": True,
                    "max_novelty": 0.2
                },
                {
                    "rule_id": "pathogen_high_conf",
                    "rule_name": "High-confidence pathogen annotations", 
                    "entity_types": ["PATHOGEN"],
                    "min_confidence": 0.95,
                    "min_agreement": 0.9,
                    "requires_rule_support": True,
                    "max_novelty": 0.2
                },
                {
                    "rule_id": "disease_known",
                    "rule_name": "Known disease entities",
                    "entity_types": ["DISEASE"],
                    "min_confidence": 0.9,
                    "min_agreement": 0.85,
                    "requires_rule_support": True,
                    "max_novelty": 0.1
                },
                {
                    "rule_id": "causes_relation_strong",
                    "rule_name": "Strong causal relations",
                    "relation_types": ["causes"],
                    "min_confidence": 0.93,
                    "min_agreement": 0.9,
                    "requires_rule_support": True,
                    "max_novelty": 0.3
                }
            ]
            
            for rule_data in default_rules:
                existing_rule = session.query(AutoAcceptRule).filter_by(rule_id=rule_data["rule_id"]).first()
                if not existing_rule:
                    rule = AutoAcceptRule(**rule_data)
                    session.add(rule)
            
            session.commit()
            logger.info("Inserted initial auto-accept rules")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to insert initial data: {e}")
            raise
        finally:
            session.close()
    
    def migrate_legacy_data(self, legacy_data_path: Path):
        """Migrate data from legacy JSON files"""
        logger.info(f"Migrating legacy data from {legacy_data_path}")
        
        session = self.SessionLocal()
        
        try:
            # Migrate gold annotations from JSON files
            if (legacy_data_path / "gold").exists():
                self._migrate_gold_annotations(session, legacy_data_path / "gold")
            
            # Migrate existing documents
            if (legacy_data_path / "raw").exists():
                self._migrate_documents(session, legacy_data_path / "raw")
            
            session.commit()
            logger.info("Legacy data migration completed")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to migrate legacy data: {e}")
            raise
        finally:
            session.close()
    
    def _migrate_gold_annotations(self, session, gold_path: Path):
        """Migrate gold annotations from JSON files"""
        import json
        
        for json_file in gold_path.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    gold_data = json.load(f)
                
                # Create or find document
                doc_id = gold_data.get("doc_id", json_file.stem)
                document = session.query(Document).filter_by(doc_id=doc_id).first()
                
                if not document:
                    # Create minimal document
                    document = Document(
                        doc_id=doc_id,
                        source="migrated",
                        title=gold_data.get("title", doc_id),
                        raw_text=gold_data.get("text", ""),
                        metadata={"migrated_from": str(json_file)}
                    )
                    session.add(document)
                    session.flush()
                
                # Create or find sentence
                sent_id = gold_data.get("sent_id", "s0")
                sentence = session.query(Sentence).filter_by(
                    document_id=document.id, 
                    sent_id=sent_id
                ).first()
                
                if not sentence:
                    text = gold_data.get("text", "")
                    sentence = Sentence(
                        sent_id=sent_id,
                        document_id=document.id,
                        start_offset=0,
                        end_offset=len(text),
                        text=text
                    )
                    session.add(sentence)
                    session.flush()
                
                # Create gold annotation
                annotation = GoldAnnotation(
                    document_id=document.id,
                    sentence_id=sentence.id,
                    entities=gold_data.get("entities", []),
                    relations=gold_data.get("relations", []),
                    topics=gold_data.get("topics", []),
                    annotator_email=gold_data.get("annotator", "migrated@system.local"),
                    status=gold_data.get("status", "accepted"),
                    decision_method="migrated"
                )
                session.add(annotation)
                
                logger.debug(f"Migrated gold annotation: {json_file.name}")
                
            except Exception as e:
                logger.error(f"Failed to migrate {json_file}: {e}")
    
    def _migrate_documents(self, session, raw_path: Path):
        """Migrate raw documents"""
        from services.ingestion.document_ingestion import DocumentIngestionService
        
        ingestion_service = DocumentIngestionService()
        
        for text_file in raw_path.glob("*.txt"):
            try:
                # Check if already exists
                doc_id = text_file.stem
                existing_doc = session.query(Document).filter_by(doc_id=doc_id).first()
                
                if existing_doc:
                    continue
                
                # Ingest document
                document_obj = ingestion_service.ingest_text_file(
                    text_file,
                    source="migrated",
                    title=text_file.stem
                )
                
                # Convert to database objects
                db_doc = Document(
                    doc_id=document_obj.doc_id,
                    source=document_obj.source,
                    title=document_obj.title,
                    raw_text=document_obj.raw_text,
                    metadata=document_obj.metadata
                )
                session.add(db_doc)
                session.flush()
                
                # Add sentences
                for sent in document_obj.sentences:
                    db_sent = Sentence(
                        sent_id=sent.sent_id,
                        document_id=db_doc.id,
                        start_offset=sent.start,
                        end_offset=sent.end,
                        text=sent.text,
                        paragraph_id=sent.paragraph_id
                    )
                    session.add(db_sent)
                
                logger.debug(f"Migrated document: {text_file.name}")
                
            except Exception as e:
                logger.error(f"Failed to migrate document {text_file}: {e}")
    
    def backup_database(self, backup_path: Path):
        """Create database backup"""
        import subprocess
        
        try:
            # Extract connection details from URL
            # Format: postgresql://user:password@host:port/database
            url_parts = self.database_url.replace("postgresql://", "").split("/")
            db_name = url_parts[-1]
            
            backup_file = backup_path / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Use pg_dump for backup
            cmd = [
                "pg_dump",
                self.database_url,
                "-f", str(backup_file),
                "--no-owner",
                "--no-privileges"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Database backup created: {backup_file}")
                return backup_file
            else:
                logger.error(f"Backup failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None
    
    def get_migration_status(self) -> Dict[str, any]:
        """Get detailed migration status"""
        current_rev = self.get_current_revision()
        available_revs = self.get_available_revisions()
        table_info = self.get_table_info()
        
        return {
            "current_revision": current_rev,
            "available_revisions": available_revs,
            "is_up_to_date": current_rev == available_revs[-1] if available_revs else True,
            "tables": list(table_info.keys()),
            "table_count": len(table_info),
            "database_accessible": self.check_database_exists()
        }


# Command line interface
def main():
    """CLI for database migration"""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Database migration utility")
    parser.add_argument("--database-url", 
                       default=os.getenv("DATABASE_URL", "postgresql://annotator:secure_password_change_me@localhost:5432/annotations"),
                       help="Database URL")
    parser.add_argument("--action", 
                       choices=["init", "create", "upgrade", "downgrade", "status", "backup", "migrate-legacy"],
                       required=True,
                       help="Migration action")
    parser.add_argument("--message", help="Migration message")
    parser.add_argument("--revision", help="Target revision")
    parser.add_argument("--legacy-data-path", help="Path to legacy data for migration")
    parser.add_argument("--backup-path", default="./backups", help="Backup directory")
    
    args = parser.parse_args()
    
    # Initialize migrator
    migrator = DatabaseMigrator(args.database_url)
    
    if args.action == "init":
        migrator.init_alembic()
        migrator.setup_fresh_database()
        
    elif args.action == "create":
        if not args.message:
            print("--message required for create action")
            return
        migrator.create_migration(args.message)
        
    elif args.action == "upgrade":
        migrator.upgrade_to_head()
        
    elif args.action == "downgrade":
        if not args.revision:
            print("--revision required for downgrade action")
            return
        migrator.downgrade_to_revision(args.revision)
        
    elif args.action == "status":
        status = migrator.get_migration_status()
        print("Migration Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
            
    elif args.action == "backup":
        backup_file = migrator.backup_database(Path(args.backup_path))
        if backup_file:
            print(f"Backup created: {backup_file}")
        else:
            print("Backup failed")
            
    elif args.action == "migrate-legacy":
        if not args.legacy_data_path:
            print("--legacy-data-path required for migrate-legacy action")
            return
        migrator.migrate_legacy_data(Path(args.legacy_data_path))
        print("Legacy data migration completed")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()