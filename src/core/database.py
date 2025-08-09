"""
Database connection and session management.
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from contextlib import contextmanager
from loguru import logger

from ..core.config import get_settings

settings = get_settings()

# Create engine based on settings
DATABASE_URL = settings.DATABASE_URL
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    connect_args=settings.database_connect_args,
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db_session = scoped_session(SessionLocal)

# Base class for all models
Base = declarative_base()


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_transaction():
    """Context manager for database transactions."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database transaction error: {e}")
        raise
    finally:
        session.close()


def init_db():
    """Initialize database tables."""
    from ..models.database import Base
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")


def reset_db():
    """Reset database (drop and recreate all tables)."""
    from ..models.database import Base
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    logger.warning("Database reset completed")
