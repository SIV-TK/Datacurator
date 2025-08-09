"""
API models for the data curation system.
This module defines the request and response models for the API.
"""
from pydantic import BaseModel, Field, HttpUrl, validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum


class SourceType(str, Enum):
    WEB = "web"
    FILE = "file"
    API = "api"
    DATABASE = "database"
    MANUAL = "manual"


class JobType(str, Enum):
    SCRAPE = "scrape"
    CLEAN = "clean"
    VALIDATE = "validate"
    TRANSFORM = "transform"
    DEDUPLICATE = "deduplicate"
    ANALYZE = "analyze"
    EXPORT = "export"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DatasetCreate(BaseModel):
    name: str
    description: Optional[str] = None
    source_type: SourceType
    config: Optional[Dict[str, Any]] = None


class DatasetResponse(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    source_type: str
    created_at: datetime
    updated_at: datetime
    total_records: int
    cleaned_records: int
    config: Optional[Dict[str, Any]] = None

    class Config:
        orm_mode = True


class DataRecordCreate(BaseModel):
    content: str
    external_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DataRecordResponse(BaseModel):
    id: int
    dataset_id: int
    external_id: Optional[str] = None
    content: str
    processed_content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    is_valid: bool
    quality_score: Optional[float] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class AnnotationCreate(BaseModel):
    key: str
    value: str
    confidence: Optional[float] = 1.0
    source: Optional[str] = "manual"


class AnnotationResponse(BaseModel):
    id: int
    record_id: int
    key: str
    value: str
    confidence: float
    source: str
    created_at: datetime

    class Config:
        orm_mode = True


class ProcessingJobCreate(BaseModel):
    dataset_id: int
    job_type: JobType
    params: Optional[Dict[str, Any]] = None


class ProcessingJobResponse(BaseModel):
    id: int
    dataset_id: int
    job_type: str
    status: str
    params: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    class Config:
        orm_mode = True


class WebScrapingConfig(BaseModel):
    urls: List[HttpUrl]
    selectors: Dict[str, str]
    follow_links: Optional[bool] = False
    max_pages: Optional[int] = 10
    delay: Optional[float] = 1.0
    headers: Optional[Dict[str, str]] = None


class FileImportConfig(BaseModel):
    file_paths: List[str]
    file_type: str
    encoding: Optional[str] = "utf-8"
    delimiter: Optional[str] = None  # For CSV files


class CleaningConfig(BaseModel):
    remove_html: Optional[bool] = True
    fix_encoding: Optional[bool] = True
    remove_duplicates: Optional[bool] = True
    language_filter: Optional[List[str]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    quality_threshold: Optional[float] = 0.7
    custom_rules: Optional[List[Dict[str, Any]]] = None
    use_ai_filter: Optional[bool] = False
    ai_config: Optional[Dict[str, Any]] = None


class ValidationResult(BaseModel):
    valid_records: int
    invalid_records: int
    validation_criteria: Dict[str, Any]
    quality_distribution: Dict[str, int]


class DataStatisticsResponse(BaseModel):
    dataset_id: int
    record_count: int
    avg_content_length: Optional[float] = None
    language_distribution: Optional[Dict[str, int]] = None
    quality_distribution: Optional[Dict[str, int]] = None
    duplicate_count: int
    last_updated: datetime

    class Config:
        orm_mode = True
