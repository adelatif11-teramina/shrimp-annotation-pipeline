"""
Tests for API endpoints
"""

import pytest
from fastapi import status

class TestDocumentEndpoints:
    """Test document-related endpoints"""
    
    def test_create_document(self, test_client, sample_document):
        """Test document creation"""
        response = test_client.post("/documents", json=sample_document)
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["doc_id"] == sample_document["doc_id"]
        assert data["title"] == sample_document["title"]
    
    def test_get_document(self, test_client, sample_document):
        """Test fetching a document"""
        # First create the document
        test_client.post("/documents", json=sample_document)
        
        # Then fetch it
        response = test_client.get(f"/documents/{sample_document['doc_id']}")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["doc_id"] == sample_document["doc_id"]
    
    def test_list_documents(self, test_client):
        """Test listing documents"""
        response = test_client.get("/documents")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert isinstance(data, list)
    
    def test_delete_document(self, test_client, sample_document):
        """Test document deletion"""
        # Create document
        test_client.post("/documents", json=sample_document)
        
        # Delete it
        response = test_client.delete(f"/documents/{sample_document['doc_id']}")
        assert response.status_code == status.HTTP_200_OK
    
    def test_document_not_found(self, test_client):
        """Test fetching non-existent document"""
        response = test_client.get("/documents/nonexistent")
        assert response.status_code == status.HTTP_404_NOT_FOUND

class TestCandidateEndpoints:
    """Test candidate generation endpoints"""
    
    @pytest.mark.asyncio
    async def test_generate_candidates(self, test_client, mock_llm_generator):
        """Test candidate generation for a sentence"""
        request_data = {
            "doc_id": "test_doc",
            "sent_id": "sent_001",
            "text": "Vibrio parahaemolyticus causes AHPND.",
            "title": "Test Document"
        }
        
        response = test_client.post("/candidates/generate", json=request_data)
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["doc_id"] == request_data["doc_id"]
        assert data["sent_id"] == request_data["sent_id"]
        assert "candidates" in data
    
    @pytest.mark.asyncio
    async def test_batch_generate_candidates(self, test_client, mock_llm_generator):
        """Test batch candidate generation"""
        request_data = {
            "sentences": [
                {
                    "doc_id": "doc1",
                    "sent_id": "s1",
                    "text": "Test sentence 1"
                },
                {
                    "doc_id": "doc1",
                    "sent_id": "s2",
                    "text": "Test sentence 2"
                }
            ],
            "batch_size": 2
        }
        
        response = test_client.post("/candidates/batch", json=request_data)
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2

class TestTriageEndpoints:
    """Test triage queue endpoints"""
    
    def test_get_triage_queue(self, test_client):
        """Test fetching triage queue"""
        response = test_client.get("/triage/queue")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_triage_queue_with_filters(self, test_client):
        """Test fetching triage queue with filters"""
        response = test_client.get("/triage/queue?limit=5&priority_filter=high")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 5

class TestAnnotationEndpoints:
    """Test annotation endpoints"""
    
    def test_submit_annotation(self, test_client, auth_headers):
        """Test submitting an annotation decision"""
        decision_data = {
            "candidate_id": 1,
            "decision": "accepted",
            "entities": [],
            "relations": [],
            "topics": [],
            "confidence": 0.95,
            "notes": "Test annotation"
        }
        
        response = test_client.post(
            "/annotations/submit",
            json=decision_data,
            headers=auth_headers
        )
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_401_UNAUTHORIZED]
    
    def test_get_annotation_history(self, test_client):
        """Test fetching annotation history"""
        response = test_client.get("/annotations/history")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert isinstance(data, list)

class TestStatsEndpoints:
    """Test statistics endpoints"""
    
    def test_get_stats(self, test_client):
        """Test fetching statistics"""
        response = test_client.get("/stats")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "total_documents" in data
        assert "total_annotations" in data
    
    def test_get_user_stats(self, test_client, auth_headers):
        """Test fetching user-specific statistics"""
        response = test_client.get("/stats/user", headers=auth_headers)
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_401_UNAUTHORIZED]

class TestHealthEndpoints:
    """Test health check endpoints"""
    
    def test_health_check(self, test_client):
        """Test basic health check"""
        response = test_client.get("/health")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_readiness_check(self, test_client):
        """Test readiness check"""
        response = test_client.get("/ready")
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "database" in data
        assert "cache" in data