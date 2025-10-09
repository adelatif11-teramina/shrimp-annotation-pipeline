"""
Application Settings Management
Centralized configuration using environment variables with validation
"""

import os
from pathlib import Path
from typing import Optional, List
from functools import lru_cache
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Environment
    environment: str = Field("development", env="ENVIRONMENT")
    debug: bool = Field(False, env="DEBUG")
    log_level: str = Field("info", env="LOG_LEVEL")
    
    # API Configuration
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_workers: int = Field(4, env="API_WORKERS")
    frontend_url: str = Field("http://localhost:3000", env="FRONTEND_URL")
    
    # Database
    db_host: str = Field("localhost", env="DB_HOST")
    db_port: int = Field(5432, env="DB_PORT")
    db_name: str = Field("annotations", env="DB_NAME")
    db_user: str = Field("annotator", env="DB_USER")
    db_password: Optional[str] = Field(None, env="DB_PASSWORD")  # Optional for Railway
    
    @property
    def database_url(self) -> str:
        """Get database URL from environment or construct PostgreSQL URL"""
        # Check if DATABASE_URL is set in environment (takes precedence)
        env_url = os.getenv("DATABASE_URL")
        if env_url:
            return env_url
        
        # Otherwise construct PostgreSQL URL
        return f"postgresql://{self.db_user}:{self.db_password}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    @property
    def sqlite_database_url(self) -> str:
        """SQLite URL for local development"""
        return f"sqlite:///{self.data_dir}/local/annotations.db"
    
    # Redis
    redis_host: str = Field("localhost", env="REDIS_HOST")
    redis_port: int = Field(6379, env="REDIS_PORT")
    redis_password: Optional[str] = Field(None, env="REDIS_PASSWORD")
    
    @property
    def redis_url(self) -> str:
        """Construct Redis URL"""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}"
        return f"redis://{self.redis_host}:{self.redis_port}"
    
    # Security
    jwt_secret_key: str = Field(..., env="JWT_SECRET_KEY")  # Required
    jwt_algorithm: str = Field("HS256", env="JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(30, env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")
    jwt_refresh_token_expire_days: int = Field(7, env="JWT_REFRESH_TOKEN_EXPIRE_DAYS")
    
    # OpenAI
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o", env="OPENAI_MODEL")
    openai_temperature: float = Field(0.1, env="OPENAI_TEMPERATURE")
    
    # Ollama
    ollama_host: str = Field("http://localhost:11434", env="OLLAMA_HOST")
    ollama_model: str = Field("llama3.2:3b", env="OLLAMA_MODEL")
    
    # Label Studio
    label_studio_url: str = Field("http://localhost:8080", env="LABEL_STUDIO_URL")
    label_studio_api_key: Optional[str] = Field(None, env="LABEL_STUDIO_API_KEY")
    
    # Rate Limiting
    rate_limit_enabled: bool = Field(True, env="RATE_LIMIT_ENABLED")
    rate_limit_per_minute: int = Field(100, env="RATE_LIMIT_PER_MINUTE")
    rate_limit_per_hour: int = Field(1000, env="RATE_LIMIT_PER_HOUR")
    
    # Storage Paths
    data_dir: Path = Field(Path("./data"), env="DATA_DIR")
    cache_dir: Path = Field(Path("./data/local/llm_cache"), env="CACHE_DIR")
    export_dir: Path = Field(Path("./data/exports"), env="EXPORT_DIR")
    log_dir: Path = Field(Path("./logs"), env="LOG_DIR")
    
    # Feature Flags
    enable_auto_accept: bool = Field(True, env="ENABLE_AUTO_ACCEPT")
    auto_accept_max_rate: float = Field(0.3, env="AUTO_ACCEPT_MAX_RATE")
    enable_websocket: bool = Field(True, env="ENABLE_WEBSOCKET")
    enable_metrics: bool = Field(True, env="ENABLE_METRICS")
    
    # Monitoring (Optional)
    sentry_dsn: Optional[str] = Field(None, env="SENTRY_DSN")
    datadog_api_key: Optional[str] = Field(None, env="DATADOG_API_KEY")
    
    @validator("environment", allow_reuse=True)
    def validate_environment(cls, v):
        allowed = ["development", "staging", "production", "testing"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}")
        return v
    
    @validator("log_level", allow_reuse=True)
    def validate_log_level(cls, v):
        allowed = ["debug", "info", "warning", "error", "critical"]
        if v.lower() not in allowed:
            raise ValueError(f"Log level must be one of {allowed}")
        return v.lower()
    
    @validator("data_dir", "cache_dir", "export_dir", "log_dir", pre=True, allow_reuse=True)
    def resolve_path(cls, v):
        if isinstance(v, str):
            return Path(v).resolve()
        return v
    
    def ensure_directories(self):
        """Create required directories if they don't exist"""
        for dir_path in [self.data_dir, self.cache_dir, self.export_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @property
    def is_production(self) -> bool:
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        return self.environment == "development"
    
    @property
    def cors_origins(self) -> List[str]:
        """Get CORS allowed origins based on environment"""
        if self.is_production:
            return [self.frontend_url]
        return ["*"]  # Allow all in development
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        
@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    Use dependency injection in FastAPI:
    
    from config.settings import get_settings
    
    @app.get("/")
    def read_root(settings: Settings = Depends(get_settings)):
        return {"env": settings.environment}
    """
    settings = Settings()
    settings.ensure_directories()
    return settings

# Export singleton instance
settings = get_settings()

# Validation on import
def validate_required_settings():
    """Validate that all required settings are present"""
    errors = []
    
    if not settings.db_password and settings.is_production:
        errors.append("DB_PASSWORD is required in production")
    
    if not settings.jwt_secret_key:
        errors.append("JWT_SECRET_KEY is required")
    
    if settings.is_production:
        if settings.jwt_secret_key == "your-secret-key-here-change-in-production":
            errors.append("JWT_SECRET_KEY must be changed from default in production")
        
        if not settings.openai_api_key and not settings.ollama_host:
            errors.append("Either OPENAI_API_KEY or OLLAMA_HOST must be configured")
    
    if errors:
        error_msg = "Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
        if settings.is_production:
            raise ValueError(error_msg)
        else:
            print(f"⚠️  Warning: {error_msg}")

# Run validation
if __name__ != "__main__":
    validate_required_settings()
