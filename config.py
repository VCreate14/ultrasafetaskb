from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Configuration
    ULTRASAFE_API_URL: str = "https://api.us.inc/usf/v1/hiring/chat/completions"
    ULTRASAFE_MODEL: str = "usf1-mini"
    ULTRASAFE_API_KEY: str  # This will be loaded from .env file
    
    # Qdrant Configuration
    QDRANT_URL: str
    QDRANT_API_KEY: str
    
    # Embeddings Configuration
    EMBEDDINGS_MODEL: str = "all-MiniLM-L6-v2"
    
    # Search Configuration
    SEARCH_LIMIT: int = 10
    MIN_RELEVANCE_SCORE: float = 0.5
    
    # RAG Configuration
    MAX_CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_TOKENS: int = 1000
    TEMPERATURE: float = 0.7
    
    # Agent Configuration
    ENABLE_WEB_SEARCH: bool = True
    STREAM_RESPONSE: bool = False
    
    # File paths
    DATA_DIR: str = "data"
    CACHE_DIR: str = "cache"
    OUTPUT_DIR: str = "outputs/reports"
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    
    # API Server Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    API_RELOAD: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings() 