"""
Tests for database operations
"""

import pytest
from datetime import datetime
from services.database.simple_db import SimpleDatabase

class TestSimpleDatabase:
    """Test SQLite database operations"""
    
    @pytest.fixture
    def db(self, temp_dir):
        """Create test database instance"""
        db_path = temp_dir / "test.db"
        return SimpleDatabase(str(db_path))
    
    def test_database_initialization(self, db):
        """Test database initialization and table creation"""
        # Check if database is accessible
        stats = db.get_statistics()
        assert stats is not None
        assert "total_documents" in stats
    
    def test_user_operations(self, db):
        """Test user CRUD operations"""
        # Create user
        user_data = {
            "username": "test_user",
            "token": "test_token",
            "role": "annotator",
            "email": "test@example.com"
        }
        user = db.create_user(**user_data)
        assert user["username"] == "test_user"
        assert "id" in user
        
        # Get user by token
        fetched_user = db.get_user_by_token("test_token")
        assert fetched_user["username"] == "test_user"
        
        # Update user
        db.update_user(user["id"], email="newemail@example.com")
        updated = db.get_user_by_token("test_token")
        assert updated["email"] == "newemail@example.com"
    
    def test_document_operations(self, db):
        """Test document CRUD operations"""
        # Create document
        doc_data = {
            "doc_id": "test_doc_001",
            "title": "Test Document",
            "source": "test",
            "raw_text": "This is a test document about shrimp farming."
        }
        doc = db.create_document(**doc_data)
        assert doc["doc_id"] == "test_doc_001"
        
        # Get document
        fetched = db.get_document("test_doc_001")
        assert fetched["title"] == "Test Document"
        
        # List documents
        docs = db.get_documents(limit=10)
        assert "documents" in docs
        assert len(docs["documents"]) > 0
        
        # Delete document
        result = db.delete_document("test_doc_001")
        assert result["status"] == "deleted"
    
    def test_sentence_operations(self, db):
        """Test sentence operations"""
        # First create a document
        doc = db.create_document(
            doc_id="doc_002",
            title="Test",
            source="test",
            raw_text="Test text"
        )
        
        # Create sentence
        sentence_data = {
            "sent_id": "sent_001",
            "doc_id": "doc_002",
            "text": "Vibrio parahaemolyticus causes AHPND.",
            "start_offset": 0,
            "end_offset": 38
        }
        sentence = db.create_sentence(**sentence_data)
        assert sentence["sent_id"] == "sent_001"
        
        # Get sentences for document
        sentences = db.get_sentences_for_document("doc_002")
        assert len(sentences) > 0
        assert sentences[0]["text"] == sentence_data["text"]
    
    def test_candidate_operations(self, db):
        """Test candidate operations"""
        # Create document and sentence first
        db.create_document(
            doc_id="doc_003",
            title="Test",
            source="test",
            raw_text="Test"
        )
        
        # Create candidate
        candidate_data = {
            "doc_id": "doc_003",
            "sent_id": "sent_001",
            "source": "llm",
            "candidate_type": "entity",
            "entities": [{"text": "test", "label": "TEST"}],
            "confidence": 0.85,
            "priority_score": 0.75
        }
        candidate = db.create_candidate(**candidate_data)
        assert candidate["source"] == "llm"
        assert candidate["confidence"] == 0.85
    
    def test_annotation_operations(self, db):
        """Test annotation operations"""
        # Setup: create user, document, and candidate
        user = db.create_user(
            username="annotator",
            token="anno_token",
            role="annotator"
        )
        
        db.create_document(
            doc_id="doc_004",
            title="Test",
            source="test",
            raw_text="Test"
        )
        
        candidate = db.create_candidate(
            doc_id="doc_004",
            sent_id="sent_001",
            source="llm",
            candidate_type="entity"
        )
        
        # Create annotation
        annotation_data = {
            "doc_id": "doc_004",
            "sent_id": "sent_001",
            "candidate_id": candidate["id"],
            "user_id": user["id"],
            "annotation_type": "entity",
            "entities": [{"text": "test", "label": "TEST"}],
            "decision": "accepted",
            "confidence": 0.90
        }
        annotation = db.create_annotation(**annotation_data)
        assert annotation["decision"] == "accepted"
        
        # Get annotations
        annotations = db.get_annotations(doc_id="doc_004")
        assert len(annotations) > 0
        
        # Get user annotations
        user_annotations = db.get_user_annotations(user["id"])
        assert len(user_annotations) > 0
    
    def test_triage_queue_operations(self, db):
        """Test triage queue operations"""
        # Setup
        db.create_document(
            doc_id="doc_005",
            title="Test",
            source="test",
            raw_text="Test"
        )
        
        candidate = db.create_candidate(
            doc_id="doc_005",
            sent_id="sent_001",
            source="llm",
            candidate_type="entity",
            priority_score=0.85
        )
        
        # Add to triage queue
        triage_item = db.add_to_triage_queue(
            candidate_id=candidate["id"],
            priority_score=0.85,
            priority_level="high"
        )
        assert triage_item["priority_level"] == "high"
        
        # Get triage queue
        queue = db.get_triage_queue(limit=10)
        assert len(queue) > 0
        
        # Update triage item status
        db.update_triage_status(triage_item["id"], "completed")
        updated = db.get_triage_item(triage_item["id"])
        assert updated["status"] == "completed"
    
    def test_statistics(self, db):
        """Test statistics calculation"""
        # Create some test data
        db.create_document(
            doc_id="stat_doc",
            title="Statistics Test",
            source="test",
            raw_text="Test document for statistics"
        )
        
        user = db.create_user(
            username="stat_user",
            token="stat_token",
            role="annotator"
        )
        
        # Get statistics
        stats = db.get_statistics()
        assert isinstance(stats, dict)
        assert "total_documents" in stats
        assert "total_users" in stats
        assert stats["total_documents"] >= 1
        assert stats["total_users"] >= 1
    
    def test_transaction_rollback(self, db):
        """Test transaction rollback on error"""
        # Try to create a duplicate user (should fail)
        db.create_user(username="dup_user", token="token1", role="annotator")
        
        # This should fail due to unique constraint
        with pytest.raises(Exception):
            db.create_user(username="dup_user", token="token2", role="annotator")
        
        # Original user should still exist
        user = db.get_user_by_token("token1")
        assert user["username"] == "dup_user"