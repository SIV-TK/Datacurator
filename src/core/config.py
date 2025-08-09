"""
Configuration management for the data curation system.
"""
import os
from typing import Dict, Any, Optional, List
from pydantic import Field, field_validator, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Application settings."""
    # Application
    APP_NAME: str = "Data Curator"
    APP_VERSION: str = "0.1.0"
    APP_ENV: str = os.getenv("APP_ENV", "development")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Database
    DATABASE_DRIVER: str = os.getenv("DATABASE_DRIVER", "sqlite")
    DATABASE_HOST: str = os.getenv("DATABASE_HOST", "localhost")
    DATABASE_PORT: str = os.getenv("DATABASE_PORT", "5432")
    DATABASE_USER: str = os.getenv("DATABASE_USER", "postgres")
    DATABASE_PASSWORD: str = os.getenv("DATABASE_PASSWORD", "postgres")
    DATABASE_NAME: str = os.getenv("DATABASE_NAME", "data_curator")
    
    # Database connection args
    database_connect_args: Dict[str, Any] = {}
    
    # AI Services
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    HF_API_KEY: Optional[str] = os.getenv("HF_API_KEY")
    
    # Web Scraping
    DEFAULT_REQUEST_HEADERS: Dict[str, str] = {
        "User-Agent": "DataCurator/0.1.0 (+https://example.com/bot)",
        "Accept-Language": "en-US,en;q=0.5",
    }
    REQUEST_TIMEOUT: int = 30
    RESPECT_ROBOTS_TXT: bool = True
    
    # Data processing
    MAX_CONTENT_LENGTH: int = 1_000_000  # Characters
    DEFAULT_CHUNK_SIZE: int = 1000
    LANGUAGES: List[str] = ["en"]  # Default supported languages
    
    # Advanced cleaner settings
    ADVANCED_CLEANER_WORKING_DIR: str = "data"
    ADVANCED_CLEANER_TARGET_LANGUAGE: str = "en"
    ADVANCED_CLEANER_MIN_WORDS: int = 5  # Lowered from 30 for testing
    ADVANCED_CLEANER_MAX_SPECIAL_CHAR_RATIO: float = 0.1
    ADVANCED_CLEANER_USE_SPACY: bool = True
    ADVANCED_CLEANER_USE_LANGUAGE_TOOL: bool = True
    
    # Advanced scraper settings
    SCRAPER_NUM_THREADS: int = 8
    SCRAPER_USE_SELENIUM_FALLBACK: bool = True
    SCRAPER_RATE_LIMIT_DELAY: float = 0.5
    SCRAPER_TARGET_LANGUAGE: str = "en"
    SCRAPER_MAX_SUB_URLS: int = 100
    SCRAPER_WORKING_DIR: str = "data"
    
    # Paths
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", os.path.join(os.getcwd(), "data")))
    EXPORT_DIR: Path = Path(os.getenv("EXPORT_DIR", os.path.join(os.getcwd(), "exports")))
    
    # API
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_PREFIX: str = "/api/v1"
    
    @computed_field
    @property
    def DATABASE_URL(self) -> str:
        """Build database URL from components."""
        driver = self.DATABASE_DRIVER
        user = self.DATABASE_USER
        password = self.DATABASE_PASSWORD
        host = self.DATABASE_HOST
        port = self.DATABASE_PORT
        db = self.DATABASE_NAME
        
        if driver == "sqlite":
            return f"sqlite:////{self.DATA_DIR}/database.sqlite"
        
        return f"{driver}://{user}:{password}@{host}:{port}/{db}"
    
    @field_validator("database_connect_args", mode="before")
    @classmethod
    def set_connect_args(cls, v: Dict[str, Any], info) -> Dict[str, Any]:
        """Set connection arguments based on database driver."""
        driver = info.data.get("DATABASE_DRIVER", "")
        
        if driver == "sqlite":
            return {"check_same_thread": False}
        elif driver == "postgresql":
            return {"connect_timeout": 10}
        
        return v if v else {}
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def ensure_directories():
    """Ensure required directories exist."""
    settings = get_settings()
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    os.makedirs(settings.EXPORT_DIR, exist_ok=True)
