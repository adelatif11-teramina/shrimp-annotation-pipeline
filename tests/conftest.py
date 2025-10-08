"""
Pytest configuration and shared fixtures
"""

import os
import sys
import asyncio
import tempfile
from pathlib import Path
from typing import Generator, AsyncGenerator
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set test environment
os.environ["ENVIRONMENT"] = "testing"
os.environ["DB_PASSWORD"] = "test_password"
os.environ["JWT_SECRET_KEY"] = "test_secret_key_for_testing_only"
os.environ["OPENAI_API_KEY"] = "test_openai_key"

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def mock_settings(monkeypatch, temp_dir):
    """Mock application settings for testing"""
    from config.settings import Settings
    
    test_settings = Settings(
        environment="testing",
        debug=True,
        db_password="test_password",
        jwt_secret_key="test_secret_key",
        data_dir=temp_dir / "data",
        cache_dir=temp_dir / "cache",
        export_dir=temp_dir / "exports",
        log_dir=temp_dir / "logs",
        openai_api_key="test_key",
        rate_limit_enabled=False  # Disable rate limiting in tests
    )
    test_settings.ensure_directories()
    
    monkeypatch.setattr("config.settings.get_settings", lambda: test_settings)
    return test_settings

@pytest.fixture
def test_db(temp_dir):
    """Create test SQLite database"""
    from services.database.models import Base
    
    db_path = temp_dir / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(bind=engine)
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    yield SessionLocal
    
    # Cleanup
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def db_session(test_db) -> Generator[Session, None, None]:
    """Create database session for testing"""
    session = test_db()
    try:
        yield session
    finally:
        session.close()

@pytest.fixture
def test_client(mock_settings, mock_monitoring, disable_psutil):
    """Create FastAPI test client"""
    from services.api.annotation_api import app
    
    with TestClient(app) as client:
        yield client

@pytest.fixture
def auth_headers():
    """Generate authentication headers for testing"""
    return {"Authorization": "Bearer test_token"}

@pytest.fixture
def sample_document():
    """Sample document for testing"""
    return {
        "doc_id": "test_doc_001",
        "title": "Test Document",
        "text": "Vibrio parahaemolyticus causes AHPND in Penaeus vannamei.",
        "source": "test"
    }

@pytest.fixture
def sample_entities():
    """Sample entities for testing"""
    return [
        {
            "text": "Vibrio parahaemolyticus",
            "label": "PATHOGEN",
            "start": 0,
            "end": 23,
            "confidence": 0.95
        },
        {
            "text": "AHPND",
            "label": "DISEASE",
            "start": 31,
            "end": 36,
            "confidence": 0.92
        },
        {
            "text": "Penaeus vannamei",
            "label": "SPECIES",
            "start": 40,
            "end": 56,
            "confidence": 0.98
        }
    ]

@pytest.fixture
def sample_relations():
    """Sample relations for testing"""
    return [
        {
            "head": 0,  # Vibrio parahaemolyticus
            "tail": 1,  # AHPND
            "label": "causes",
            "confidence": 0.90
        },
        {
            "head": 1,  # AHPND
            "tail": 2,  # Penaeus vannamei
            "label": "affects",
            "confidence": 0.88
        }
    ]

@pytest.fixture
async def mock_llm_generator(monkeypatch):
    """Mock LLM generator for testing"""
    from services.candidates.llm_candidate_generator import LLMCandidateGenerator
    
    class MockLLMGenerator:
        async def extract_entities(self, sentence):
            return []
        
        async def extract_relations(self, sentence, entities):
            return []
        
        async def suggest_topics(self, text, title=None):
            return []
        
        async def process_sentence(self, doc_id, sent_id, sentence, title=None):
            return {
                "doc_id": doc_id,
                "sent_id": sent_id,
                "candidates": {
                    "entities": [],
                    "relations": [],
                    "topics": []
                }
            }
    
    monkeypatch.setattr(
        "services.candidates.llm_candidate_generator.LLMCandidateGenerator",
        MockLLMGenerator
    )
    
    return MockLLMGenerator()

@pytest.fixture
def mock_redis(monkeypatch):
    """Mock Redis for testing"""
    class MockRedis:
        def __init__(self):
            self.data = {}
        
        def get(self, key):
            return self.data.get(key)
        
        def set(self, key, value, ex=None):
            self.data[key] = value
            return True
        
        def delete(self, key):
            if key in self.data:
                del self.data[key]
                return 1
            return 0
        
        def exists(self, key):
            return key in self.data
    
    mock_redis_instance = MockRedis()
    monkeypatch.setattr("redis.Redis", lambda **kwargs: mock_redis_instance)
    return mock_redis_instance

@pytest.fixture
def mock_monitoring(monkeypatch):
    """Mock monitoring components to prevent psutil issues in tests"""
    
    class MockApplicationMonitor:
        def __init__(self, *args, **kwargs):
            pass
        
        async def start_monitoring(self, interval=60):
            pass
        
        async def stop_monitoring(self):
            pass
        
        def metrics(self):
            return MockMetricsCollector()
    
    class MockMetricsCollector:
        def increment_counter(self, *args, **kwargs):
            pass
        
        def set_gauge(self, *args, **kwargs):
            pass
        
        def observe_histogram(self, *args, **kwargs):
            pass
    
    # Mock the monitoring module
    monkeypatch.setattr("utils.monitoring.ApplicationMonitor", MockApplicationMonitor)
    monkeypatch.setattr("utils.monitoring.monitor", MockApplicationMonitor())
    
    return MockApplicationMonitor()

@pytest.fixture
def disable_psutil(monkeypatch):
    """Disable psutil to prevent floating point exceptions in tests"""
    
    class MockProcess:
        def cpu_percent(self, interval=None):
            return 10.0
    
    class MockVirtualMemory:
        percent = 50.0
    
    class MockDiskUsage:
        used = 1000000
        total = 10000000
    
    def mock_cpu_percent(interval=None):
        return 10.0
    
    def mock_virtual_memory():
        return MockVirtualMemory()
    
    def mock_disk_usage(path):
        return MockDiskUsage()
    
    def mock_getloadavg():
        return [0.5, 0.6, 0.7]
    
    # Mock psutil functions that can cause floating point exceptions
    monkeypatch.setattr("psutil.cpu_percent", mock_cpu_percent)
    monkeypatch.setattr("psutil.virtual_memory", mock_virtual_memory)
    monkeypatch.setattr("psutil.disk_usage", mock_disk_usage)
    monkeypatch.setattr("psutil.getloadavg", mock_getloadavg)