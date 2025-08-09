"""
Web UI for the Data Curator application.
"""
import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from fastapi import FastAPI, Request, Form, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from datetime import datetime
import shutil
from pydantic import BaseModel, HttpUrl
import asyncio
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.config import get_settings, ensure_directories
from src.core.database import get_db, SessionLocal
from src.core.logging import logger
from src.models.database import Dataset, DataRecord
from src.models.api import WebScrapingConfig, FileImportConfig, CleaningConfig

# Create the application
app = FastAPI(
    title="Data Curator",
    description="Web UI for data curation and processing",
    version="0.1.0",
)

# Get settings
settings = get_settings()

# Mount static files
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

# Set up templates
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# Track background jobs
background_jobs = {}


# API Models
class EnhancedExtractionRequest(BaseModel):
    """Model for enhanced document extraction requests."""
    input_method: str  # "files", "folder", "existing"
    processing_method: str  # "basic", "enhanced", "ai"
    
    # Enhanced extraction options
    max_tokens: Optional[int] = 512
    min_chunk_size: int = 100
    chunk_overlap: int = 50
    target_language: Optional[str] = "en"  # "auto" for no filtering
    text_quality: float = 0.7
    enable_ocr: bool = True
    num_workers: Optional[int] = 4
    
    # Processing options
    extract_text: bool = True
    preserve_formatting: bool = True
    extract_metadata: bool = True
    ocr_images: bool = False
    generate_summary: bool = False
    extract_keywords: bool = False
    categorize_content: bool = False
    sentiment_analysis: bool = False
    
    # Advanced settings
    quality_threshold: float = 0.5
    max_file_size: int = 50  # MB
    parallel_processing: bool = True
    preserve_structure: bool = False
    
    # Output configuration
    output_name: Optional[str] = None
    output_format: str = "jsonl"


# Web pages
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/datasets", response_class=HTMLResponse)
async def list_datasets(request: Request):
    """Render the datasets page."""
    return templates.TemplateResponse("datasets.html", {"request": request})


@app.get("/scrape", response_class=HTMLResponse)
async def scrape_page(request: Request):
    """Render the scraping page."""
    return templates.TemplateResponse("scrape.html", {"request": request})


@app.get("/process", response_class=HTMLResponse)
async def process_page(request: Request):
    """Render the file processing page."""
    return templates.TemplateResponse("process.html", {"request": request})


@app.get("/clean", response_class=HTMLResponse)
async def clean_page(request: Request):
    """Render the data cleaning page."""
    return templates.TemplateResponse("clean.html", {"request": request})


@app.get("/jobs", response_class=HTMLResponse)
async def jobs_page(request: Request):
    """Render the jobs page."""
    return templates.TemplateResponse("jobs.html", {"request": request})


# API endpoints
@app.get("/api/datasets", response_class=JSONResponse)
async def api_list_datasets():
    """Get all datasets."""
    db = next(get_db())
    datasets = db.query(Dataset).all()
    
    result = []
    for dataset in datasets:
        result.append({
            "id": dataset.id,
            "name": dataset.name,
            "description": dataset.description,
            "source_type": dataset.source_type,
            "total_records": dataset.total_records,
            "cleaned_records": dataset.cleaned_records,
            "created_at": dataset.created_at.isoformat(),
            "updated_at": dataset.updated_at.isoformat(),
        })
    
    return result


@app.get("/api/files", response_class=JSONResponse)
async def api_list_files():
    """Get all available files from datasets for processing."""
    try:
        db = next(get_db())
        datasets = db.query(Dataset).all()
        
        files = []
        for dataset in datasets:
            records = db.query(DataRecord).filter(DataRecord.dataset_id == dataset.id).limit(10).all()
            for record in records:
                files.append({
                    "id": record.id,
                    "name": f"{dataset.name} - Record {record.id}",
                    "size": len(record.content) if record.content else 0,
                    "type": dataset.source_type or "unknown",
                    "dataset_id": dataset.id,
                    "dataset_name": dataset.name,
                    "preview": record.content[:100] + "..." if record.content and len(record.content) > 100 else record.content
                })
        
        return {"files": files}
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        return {"files": []}


@app.get("/api/jobs/{job_id}", response_class=JSONResponse)
async def api_get_job(job_id: str):
    """Get the status of a background job."""
    if job_id not in background_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return background_jobs[job_id]


@app.get("/api/jobs", response_class=JSONResponse)
async def api_list_jobs():
    """List all background jobs."""
    return list(background_jobs.values())


def start_server():
    """Start the web server."""
    # Ensure required directories exist
    ensure_directories()
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    start_server()
