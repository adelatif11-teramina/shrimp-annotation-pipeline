"""
Tests for API endpoints
"""

import pytest
from fastapi import status

class TestDocumentEndpoints:
    """Test document-related endpoints"""

    def test_ingest_document(self, test_client, sample_document):
        """Document ingestion persists text and returns summary"""
        response = test_client.post("/documents/ingest", json=sample_document)
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "doc_id" in data
        assert data["sentence_count"] >= 1

    def test_get_document_placeholder(self, test_client, sample_document):
        """Placeholder document endpoint returns stub payload"""
        test_client.post("/documents/ingest", json=sample_document)

        response = test_client.get(f"/documents/{sample_document['doc_id']}")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["doc_id"] == sample_document["doc_id"]
        assert data["status"] == "not_implemented"

    def test_list_documents(self, test_client):
        """Listing documents returns metadata collection"""
        response = test_client.get("/documents")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "documents" in data
        assert "count" in data
        assert isinstance(data["documents"], list)

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
    
    def test_submit_annotation_decision(self, test_client):
        """Annotation decisions accept payload and respond with success"""
        decision_data = {
            "item_id": "doc1_sent1_entity_0",
            "decision": "accepted",
            "annotator": "tester",
            "final_annotation": {"entities": []},
            "notes": "Test annotation"
        }

        response = test_client.post("/annotations/decisions", json=decision_data)
        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "success"

class TestStatsEndpoints:
    """Test statistics endpoints"""
    
    def test_get_system_statistics(self, test_client):
        """Statistics endpoint returns service breakdown"""
        response = test_client.get("/statistics/overview")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "timestamp" in data
        assert "services" in data

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
        assert "services" in data
