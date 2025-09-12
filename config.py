# config.py
import os
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings"""
    
    # API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Database
    database_url: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://postgres:password@localhost:5432/dental"
    )
    
    # Redis
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    cache_ttl: int = int(os.getenv("CACHE_TTL", "3600"))
    
    # Application
    app_env: str = os.getenv("APP_ENV", "development")
    debug: bool = os.getenv("DEBUG", "True").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # API Settings
    max_tokens: int = int(os.getenv("MAX_TOKENS", "2000"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    
    # Security
    secret_key: str = os.getenv("SECRET_KEY", "change-me-in-production")
    
    # Default tenant for testing
    default_tenant_id: str = os.getenv(
        "TENANT_ID_DEFAULT", 
        "11111111-1111-1111-1111-111111111111"
    )
    
    # Model settings
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    llm_model_advanced: str = "gpt-4o"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Create global settings instance
settings = Settings()

# Validate required settings
if not settings.openai_api_key:
    raise ValueError("OPENAI_API_KEY is required. Please set it in your .env file")

# PHI Patterns for redaction
PHI_PATTERNS = {
    'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
    'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'dob': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
    'mrn': r'\b(MRN|mrn)[:\s]*\d+\b'
}