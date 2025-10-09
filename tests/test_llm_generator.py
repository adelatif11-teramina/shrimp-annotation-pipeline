"""
Tests for LLM candidate generator
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from services.candidates.llm_candidate_generator import (
    LLMCandidateGenerator,
    EntityCandidate,
    RelationCandidate,
    TopicCandidate
)

class TestLLMCandidateGenerator:
    """Test LLM candidate generator functionality"""
    
    @pytest.fixture
    def generator(self, temp_dir):
        """Create LLM generator instance for testing"""
        return LLMCandidateGenerator(
            provider="openai",
            model="gpt-5",
            api_key="test_key",
            temperature=0.1,
            cache_dir=temp_dir / "cache"
        )
    
    def test_initialization(self, generator):
        """Test generator initialization"""
        assert generator.provider == "openai"
        assert generator.model == "gpt-5"
        assert generator.temperature == 0.1
        assert generator.cache_dir.exists()
    
    def test_cache_key_generation(self, generator):
        """Test cache key generation"""
        text = "Test sentence"
        task = "entities"
        
        key1 = generator._get_cache_key(text, task)
        key2 = generator._get_cache_key(text, task)
        key3 = generator._get_cache_key("Different", task)
        
        assert key1 == key2  # Same input = same key
        assert key1 != key3  # Different input = different key
    
    def test_cache_operations(self, generator):
        """Test cache save and load"""
        cache_key = "test_cache_key"
        test_data = {"entities": [{"text": "test", "label": "TEST"}]}
        
        # Save to cache
        generator._save_to_cache(cache_key, test_data)
        
        # Load from cache
        loaded = generator._load_from_cache(cache_key)
        assert loaded == test_data
    
    def test_span_validation_exact_match(self, generator):
        """Test entity span validation with exact match"""
        sentence = "Vibrio parahaemolyticus causes AHPND in shrimp."
        
        # Exact match
        start, end = generator._validate_entity_span(
            sentence, "Vibrio parahaemolyticus", 0, 23
        )
        assert start == 0
        assert end == 23
        assert sentence[start:end] == "Vibrio parahaemolyticus"
    
    def test_span_validation_case_insensitive(self, generator):
        """Test entity span validation with case differences"""
        sentence = "Vibrio parahaemolyticus causes AHPND in shrimp."
        
        # Case insensitive match
        start, end = generator._validate_entity_span(
            sentence, "vibrio parahaemolyticus", 0, 23
        )
        assert start == 0
        assert end == 23
    
    def test_span_validation_find_in_sentence(self, generator):
        """Test finding entity in sentence when positions are wrong"""
        sentence = "The pathogen Vibrio parahaemolyticus causes disease."
        
        # Wrong positions but text exists
        start, end = generator._validate_entity_span(
            sentence, "Vibrio parahaemolyticus", 0, 23
        )
        assert start == 13
        assert end == 36
        assert sentence[start:end] == "Vibrio parahaemolyticus"
    
    def test_span_validation_failure(self, generator):
        """Test span validation failure when text not found"""
        sentence = "Some other text without the entity."
        
        start, end = generator._validate_entity_span(
            sentence, "Vibrio parahaemolyticus", 0, 23
        )
        assert start is None
        assert end is None
    
    def test_text_similarity(self, generator):
        """Test text similarity calculation"""
        assert generator._text_similarity("test", "test") == 1.0
        assert generator._text_similarity("test", "Test") == 0.75  # 3/4 chars match
        assert generator._text_similarity("test", "best") == 0.75
        assert generator._text_similarity("test", "xxxx") == 0.0
        assert generator._text_similarity("", "test") == 0.0
    
    @pytest.mark.asyncio
    @patch('services.candidates.llm_candidate_generator.openai.OpenAI')
    async def test_extract_entities_with_mock(self, mock_openai, generator):
        """Test entity extraction with mocked OpenAI response"""
        # Mock OpenAI response
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "entities": [
                {
                    "text": "Vibrio parahaemolyticus",
                    "label": "PATHOGEN",
                    "start": 0,
                    "end": 23,
                    "confidence": 0.95
                }
            ]
        })
        
        mock_client.chat.completions.create = Mock(return_value=mock_response)
        generator.client = mock_client
        
        # Test extraction
        sentence = "Vibrio parahaemolyticus causes AHPND."
        entities = await generator.extract_entities(sentence)
        
        assert len(entities) == 1
        assert entities[0].text == "Vibrio parahaemolyticus"
        assert entities[0].label == "PATHOGEN"
        assert entities[0].confidence == 0.95
    
    @pytest.mark.asyncio
    async def test_extract_relations(self, generator):
        """Test relation extraction"""
        sentence = "Vibrio parahaemolyticus causes AHPND."
        entities = [
            EntityCandidate(0, "Vibrio parahaemolyticus", "PATHOGEN", 0, 23, 0.95),
            EntityCandidate(1, "AHPND", "DISEASE", 31, 36, 0.92)
        ]
        
        with patch.object(generator, '_call_openai', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {
                "relations": [
                    {
                        "head": 0,
                        "tail": 1,
                        "label": "causes",
                        "confidence": 0.90
                    }
                ]
            }
            
            relations = await generator.extract_relations(sentence, entities)
            
            assert len(relations) == 1
            assert relations[0].label == "causes"
            assert relations[0].head_cid == 0
            assert relations[0].tail_cid == 1
    
    @pytest.mark.asyncio
    async def test_suggest_topics(self, generator):
        """Test topic suggestion"""
        text = "Research on AHPND disease management in shrimp farming."
        
        with patch.object(generator, '_call_openai', new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {
                "topics": [
                    {
                        "topic_id": "T_DISEASE",
                        "label": "Disease Management",
                        "score": 0.85,
                        "keywords": ["AHPND", "disease", "management"]
                    }
                ]
            }
            
            topics = await generator.suggest_topics(text)
            
            assert len(topics) == 1
            assert topics[0].topic_id == "T_DISEASE"
            assert topics[0].score == 0.85
    
    @pytest.mark.asyncio
    async def test_process_sentence(self, generator):
        """Test complete sentence processing"""
        with patch.object(generator, 'extract_entities', new_callable=AsyncMock) as mock_entities:
            with patch.object(generator, 'extract_relations', new_callable=AsyncMock) as mock_relations:
                with patch.object(generator, 'suggest_topics', new_callable=AsyncMock) as mock_topics:
                    mock_entities.return_value = []
                    mock_relations.return_value = []
                    mock_topics.return_value = []
                    
                    result = await generator.process_sentence(
                        "doc1", "sent1", "Test sentence", "Test Title"
                    )
                    
                    assert result["doc_id"] == "doc1"
                    assert result["sent_id"] == "sent1"
                    assert "candidates" in result
                    assert "processing_time" in result
    
    @pytest.mark.asyncio
    async def test_process_batch(self, generator):
        """Test batch processing"""
        sentences = [
            {"doc_id": "d1", "sent_id": "s1", "text": "Sentence 1"},
            {"doc_id": "d1", "sent_id": "s2", "text": "Sentence 2"}
        ]
        
        with patch.object(generator, 'process_sentence', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = {
                "doc_id": "d1",
                "sent_id": "s1",
                "candidates": {},
                "processing_time": 0.1
            }
            
            results = await generator.process_batch(sentences, batch_size=2)
            
            assert len(results) == 2
            assert mock_process.call_count == 2