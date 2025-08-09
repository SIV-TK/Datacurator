"""
Database models for the data curation system.
This module defines the schema and data models for storing and tracking datasets.
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, Float, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import datetime

Base = declarative_base()

class Dataset(Base):
    """Dataset model to track collections of data."""
    __tablename__ = 'datasets'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    source_type = Column(String(50), nullable=False)  # web, file, api, etc.
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    total_records = Column(Integer, default=0)
    cleaned_records = Column(Integer, default=0)
    config = Column(JSON, nullable=True)  # Store configuration as JSON
    
    # Relationships
    records = relationship("DataRecord", back_populates="dataset", cascade="all, delete-orphan")
    processing_jobs = relationship("ProcessingJob", back_populates="dataset")
    
    def __repr__(self):
        return f"<Dataset(name='{self.name}', records={self.total_records})>"


class DataRecord(Base):
    """Individual data records within a dataset."""
    __tablename__ = 'data_records'
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)
    external_id = Column(String(255), nullable=True)  # ID from external source
    content = Column(Text, nullable=False)  # Raw content
    processed_content = Column(Text, nullable=True)  # Processed/cleaned content
    record_metadata = Column(JSON, nullable=True)  # Additional metadata
    is_valid = Column(Boolean, default=True)
    quality_score = Column(Float, nullable=True)  # Quality score (0-1)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="records")
    annotations = relationship("Annotation", back_populates="record", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<DataRecord(id={self.id}, valid={self.is_valid})>"


class Annotation(Base):
    """Annotations or labels for data records."""
    __tablename__ = 'annotations'
    
    id = Column(Integer, primary_key=True)
    record_id = Column(Integer, ForeignKey('data_records.id'), nullable=False)
    key = Column(String(100), nullable=False)  # Annotation type/key
    value = Column(Text, nullable=False)  # Annotation value
    confidence = Column(Float, default=1.0)  # Confidence score (0-1)
    source = Column(String(50), default='manual')  # manual, ai, rule, etc.
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    record = relationship("DataRecord", back_populates="annotations")
    
    def __repr__(self):
        return f"<Annotation(key='{self.key}', source='{self.source}')>"


class ProcessingJob(Base):
    """Track data processing jobs."""
    __tablename__ = 'processing_jobs'
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)
    job_type = Column(String(50), nullable=False)  # scrape, clean, validate, etc.
    status = Column(String(20), default='pending')  # pending, running, completed, failed
    params = Column(JSON, nullable=True)  # Job parameters
    result = Column(JSON, nullable=True)  # Job results
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="processing_jobs")
    
    def __repr__(self):
        return f"<ProcessingJob(type='{self.job_type}', status='{self.status}')>"


class DataStatistics(Base):
    """Statistics about datasets."""
    __tablename__ = 'data_statistics'
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False, unique=True)
    record_count = Column(Integer, default=0)
    avg_content_length = Column(Float, nullable=True)
    language_distribution = Column(JSON, nullable=True)  # {language: count}
    quality_distribution = Column(JSON, nullable=True)  # {score_range: count}
    duplicate_count = Column(Integer, default=0)
    last_updated = Column(DateTime, default=datetime.datetime.utcnow)
    
    def __repr__(self):
        return f"<DataStatistics(dataset_id={self.dataset_id})>"
