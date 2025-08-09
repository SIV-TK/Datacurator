"""
Logging configuration for the data curation system.
"""
import os
import sys
import json
from pathlib import Path
from loguru import logger
from ..core.config import get_settings

settings = get_settings()

# Remove default logger
logger.remove()

# Add console logger
log_level = settings.LOG_LEVEL
logger.add(
    sys.stderr,
    level=log_level,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
)

# Add file logger
log_dir = Path(settings.DATA_DIR) / "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = log_dir / "data_curator.log"

logger.add(
    log_file,
    rotation="10 MB",
    retention="1 week",
    compression="zip",
    level=log_level,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
)

# Add JSON logger for structured logs
json_log_file = log_dir / "data_curator.json"
logger.add(
    json_log_file,
    rotation="10 MB",
    retention="1 week",
    compression="zip",
    level=log_level,
    serialize=True,
)

logger.info(f"Logging initialized with level: {log_level}")
logger.debug(f"Application environment: {settings.APP_ENV}")


def get_logger(name: str):
    """
    Get a logger for a specific module.
    
    Args:
        name: Module name
        
    Returns:
        Logger instance
    """
    return logger.bind(name=name)
