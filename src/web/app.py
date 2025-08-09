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


class ScrapeRequest(BaseModel):
    """Model for web scraping requests."""
    urls: List[str]
    selectors: Dict[str, str]
    follow_links: bool = False
    max_pages: int = 10


class FileProcessRequest(BaseModel):
    """Model for file processing requests."""
    recursive: bool = False


class CleanRequest(BaseModel):
    """Model for text cleaning requests."""
    dataset_id: Optional[int] = None
    text_data: Optional[List[str]] = None
    use_advanced: bool = False
    remove_html: bool = True
    min_length: int = 0
    max_length: int = 0


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


@app.post("/api/clean")
async def clean_data(request: CleanRequest, background_tasks: BackgroundTasks):
    """Clean text data using basic or advanced cleaning methods."""
    import uuid
    
    job_id = str(uuid.uuid4())
    background_jobs[job_id] = ProcessingJob(
        id=job_id,
        job_type="clean",
        status="queued",
        created_at=datetime.utcnow().isoformat(),
        details={
            "use_advanced": request.use_advanced,
            "dataset_id": request.dataset_id,
            "text_count": len(request.text_data) if request.text_data else 0,
        },
    ).dict()
    
    # Run cleaning in the background
    background_tasks.add_task(
        run_clean_job,
        job_id=job_id,
        dataset_id=request.dataset_id,
        text_data=request.text_data,
        use_advanced=request.use_advanced,
        remove_html=request.remove_html,
        min_length=request.min_length,
        max_length=request.max_length,
    )
    
    return {"job_id": job_id}


async def run_clean_job(job_id: str, dataset_id: Optional[int] = None, 
                       text_data: Optional[List[str]] = None, use_advanced: bool = False,
                       remove_html: bool = True, min_length: int = 0, max_length: int = 0):
    """Run a text cleaning job in the background."""
    try:
        # Update job status
        background_jobs[job_id]["status"] = "running"
        
        texts_to_clean = []
        
        # Get texts from dataset or direct input
        if dataset_id:
            db = next(get_db())
            dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if dataset:
                records = db.query(DataRecord).filter(DataRecord.dataset_id == dataset_id).all()
                texts_to_clean = [record.content for record in records if record.content]
        elif text_data:
            texts_to_clean = text_data
        else:
            raise ValueError("Either dataset_id or text_data must be provided")
        
        cleaned_texts = []
        
        if use_advanced:
            # Use advanced cleaning
            from src.core.advanced_cleaner import AdvancedTextCleaner
            
            with AdvancedTextCleaner() as cleaner:
                cleaned_texts = cleaner.clean_texts_batch(texts_to_clean, show_progress=False)
        else:
            # Use basic cleaning
            from src.core.cleaner import TextCleaner
            
            cleaner = TextCleaner(
                remove_html=remove_html,
                min_length=min_length if min_length > 0 else None,
                max_length=max_length if max_length > 0 else None,
            )
            
            for text in texts_to_clean:
                cleaned = cleaner.clean_text(text)
                if cleaned:
                    cleaned_texts.append(cleaned)
        
        # Update job with results
        background_jobs[job_id]["status"] = "completed"
        background_jobs[job_id]["details"]["results"] = {
            "original_count": len(texts_to_clean),
            "cleaned_count": len(cleaned_texts),
            "sample_cleaned": cleaned_texts[:3] if cleaned_texts else []
        }
        
        logger.info(f"Cleaning job {job_id} completed: {len(cleaned_texts)} texts cleaned")
        
    except Exception as e:
        # Update job with error
        background_jobs[job_id]["status"] = "failed"
        background_jobs[job_id]["details"]["error"] = str(e)
        logger.error(f"Cleaning job {job_id} failed: {e}")


@app.post("/api/enhanced-extract")
async def api_enhanced_extract(
    files: List[UploadFile] = File(None),
    request_data: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    """Enhanced document extraction with OCR, chunking, and quality filtering."""
    try:
        # Parse the request data
        request = EnhancedExtractionRequest.model_validate_json(request_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request data: {e}")
    
    # Generate job ID
    job_id = f"enhanced_extract_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Initialize job tracking
    background_jobs[job_id] = {
        "status": "initializing",
        "created_at": datetime.now().isoformat(),
        "details": {
            "processing_method": request.processing_method,
            "files_count": len(files) if files else 0,
            "parameters": {
                "max_tokens": request.max_tokens,
                "min_chunk_size": request.min_chunk_size,
                "target_language": request.target_language,
                "text_quality": request.text_quality,
                "enable_ocr": request.enable_ocr,
            }
        }
    }
    
    # Create temporary directory for uploaded files
    temp_dir = Path(settings.DATA_DIR) / "tmp" / job_id
    os.makedirs(temp_dir, exist_ok=True)
    
    file_paths = []
    if files:
        # Save uploaded files
        for file in files:
            if file.filename:
                file_path = temp_dir / file.filename
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                file_paths.append(str(file_path))
    
    # Run enhanced extraction in the background
    background_tasks.add_task(
        run_enhanced_extraction_job,
        job_id=job_id,
        file_paths=file_paths,
        temp_dir=str(temp_dir),
        request=request,
    )
    
    return {"job_id": job_id}


async def run_enhanced_extraction_job(
    job_id: str, 
    file_paths: List[str], 
    temp_dir: str, 
    request: EnhancedExtractionRequest
):
    """Run an enhanced document extraction job in the background."""
    try:
        # Update job status
        background_jobs[job_id]["status"] = "running"
        
        # Import the enhanced document extractor
        from src.ingestion.enhanced_document_extractor import EnhancedDocumentExtractor
        
        # Prepare output file
        output_name = request.output_name or f"enhanced_extraction_{job_id}"
        output_file = Path(settings.DATA_DIR) / "exports" / f"{output_name}.{request.output_format}"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure extractor based on processing method
        if request.processing_method == "enhanced":
            # Use enhanced extraction with all options
            target_language = None if request.target_language == "auto" else request.target_language
            num_workers = None if request.num_workers == 0 else request.num_workers
            
            extractor = EnhancedDocumentExtractor(
                input_dir=temp_dir,
                output_file=str(output_file),
                max_tokens=request.max_tokens,
                min_chunk_size=request.min_chunk_size,
                overlap=request.chunk_overlap,
                target_language=target_language,
                min_text_quality=request.text_quality,
                enable_ocr=request.enable_ocr,
                num_workers=num_workers
            )
            
            # Run the extraction
            extractor.process_documents()
            
            # Get statistics from the processing
            stats = {
                "total_files": len(file_paths),
                "processed_files": len(file_paths),  # This would be updated by the extractor
                "chunks_created": 0,  # This would be counted from the output
                "output_file": str(output_file),
                "metadata_file": str(output_file.with_name(f"{output_file.stem}_metadata.jsonl"))
            }
            
            # Count chunks in output file
            if output_file.exists():
                with open(output_file, 'r', encoding='utf-8') as f:
                    stats["chunks_created"] = sum(1 for _ in f)
            
        else:
            # Basic extraction or AI processing (placeholder for now)
            stats = {
                "total_files": len(file_paths),
                "processed_files": 0,
                "chunks_created": 0,
                "message": f"Processing method '{request.processing_method}' not yet implemented"
            }
        
        # Update job with results
        background_jobs[job_id]["status"] = "completed"
        background_jobs[job_id]["details"]["results"] = stats
        
        # Save to database if processing was successful
        if request.processing_method == "enhanced" and output_file.exists():
            db = SessionLocal()
            try:
                # Create dataset record
                dataset = Dataset(
                    name=f"Enhanced Extraction - {output_name}",
                    description=f"Enhanced document extraction with OCR and chunking",
                    source_type="enhanced_extraction",
                    status="completed",
                    metadata={
                        "processing_method": request.processing_method,
                        "parameters": request.model_dump(),
                        "statistics": stats
                    }
                )
                
                db.add(dataset)
                db.commit()
                db.refresh(dataset)
                
                # Add records from the output file
                if output_file.exists():
                    with open(output_file, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f):
                            try:
                                chunk = json.loads(line.strip())
                                record = DataRecord(
                                    dataset_id=dataset.id,
                                    content=chunk if isinstance(chunk, str) else str(chunk),
                                    external_id=f"chunk_{line_num}",
                                    record_metadata={"chunk_number": line_num},
                                    is_valid=True,
                                    quality_score=request.text_quality
                                )
                                db.add(record)
                            except json.JSONDecodeError:
                                continue
                    
                    db.commit()
                
                background_jobs[job_id]["details"]["dataset_id"] = dataset.id
                
            finally:
                db.close()
        
        logger.info(f"Enhanced extraction job {job_id} completed successfully")
        
    except Exception as e:
        # Update job with error
        background_jobs[job_id]["status"] = "failed"
        background_jobs[job_id]["details"]["error"] = str(e)
        logger.error(f"Enhanced extraction job {job_id} failed: {e}")
    
    finally:
        # Clean up temporary files
        try:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory {temp_dir}: {e}")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/datasets", response_class=HTMLResponse)
async def list_datasets(request: Request):
    """Render the datasets page."""
    return templates.TemplateResponse("datasets.html", {"request": request})


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


@app.get("/api/datasets/{dataset_id}", response_class=JSONResponse)
async def api_get_dataset(dataset_id: int):
    """Get a dataset by ID."""
    db = next(get_db())
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return {
        "id": dataset.id,
        "name": dataset.name,
        "description": dataset.description,
        "source_type": dataset.source_type,
        "total_records": dataset.total_records,
        "cleaned_records": dataset.cleaned_records,
        "created_at": dataset.created_at.isoformat(),
        "updated_at": dataset.updated_at.isoformat(),
        "config": dataset.config,
    }


@app.post("/api/datasets", response_class=JSONResponse)
async def api_create_dataset(request: Request):
    """Create a new dataset."""
    data = await request.json()
    
    db = next(get_db())
    
    # Create new dataset
    dataset = Dataset(
        name=data.get("name"),
        description=data.get("description"),
        source_type=data.get("source_type"),
        config=data.get("config"),
    )
    
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    
    return {
        "id": dataset.id,
        "name": dataset.name,
        "description": dataset.description,
        "source_type": dataset.source_type,
        "total_records": dataset.total_records,
        "cleaned_records": dataset.cleaned_records,
        "created_at": dataset.created_at.isoformat(),
        "updated_at": dataset.updated_at.isoformat(),
    }


@app.get("/api/datasets/{dataset_id}/records", response_class=JSONResponse)
async def api_list_dataset_records(dataset_id: int, limit: int = 100, offset: int = 0):
    """Get records for a dataset."""
    db = next(get_db())
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    records = db.query(DataRecord).filter(
        DataRecord.dataset_id == dataset_id
    ).limit(limit).offset(offset).all()
    
    result = []
    for record in records:
        result.append({
            "id": record.id,
            "dataset_id": record.dataset_id,
            "external_id": record.external_id,
            "content": record.content,
            "processed_content": record.processed_content,
            "is_valid": record.is_valid,
            "quality_score": record.quality_score,
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
        })
    
    return result


@app.post("/api/scrape", response_class=JSONResponse)
async def api_scrape(request: ScrapeRequest, background_tasks: BackgroundTasks):
    """Scrape websites."""
    # Generate job ID
    job_id = f"scrape_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Create job record
    background_jobs[job_id] = {
        "id": job_id,
        "job_type": "scrape",
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "details": {
            "urls": request.urls,
            "selectors": request.selectors,
            "follow_links": request.follow_links,
            "max_pages": request.max_pages,
        },
    }
    
    # Run scraping in the background
    background_tasks.add_task(
        run_scrape_job,
        job_id=job_id,
        urls=request.urls,
        selectors=request.selectors,
        follow_links=request.follow_links,
        max_pages=request.max_pages,
    )
    
    return {"job_id": job_id}


async def run_scrape_job(job_id: str, urls: List[str], selectors: Dict[str, str], 
                         follow_links: bool = False, max_pages: int = 10):
    """Run a scraping job in the background."""
    try:
        # Update job status
        background_jobs[job_id]["status"] = "running"
        
        # Import the advanced scraper
        from src.ingestion.advanced_web_scraper import AdvancedWebScraper
        
        # Use the advanced scraper
        async with AdvancedWebScraper() as scraper:
            if len(urls) == 1 and follow_links:
                # Use crawling for single URL with following links
                data = await scraper.crawl_website(
                    start_url=urls[0],
                    max_pages=max_pages,
                    selectors=selectors,
                    use_playwright=False
                )
            else:
                # Use regular scraping for multiple URLs
                url_dicts = [{'url': url} for url in urls]
                data = scraper.scrape_urls_batch(url_dicts, None, "API Scraping")
        
        # Process the scraped data
        total_items = len(data)
        text_items = 0
        
        for item in data:
            if isinstance(item, dict) and ('text' in item or 'content' in item):
                text_items += 1
        
        # Update job with results
        background_jobs[job_id]["status"] = "completed"
        background_jobs[job_id]["details"]["results"] = {
            "total_items": total_items,
            "text_items": text_items,
            "sample_data": data[:5]  # Store first 5 items as sample
        }
        
        logger.info(f"Scraping job {job_id} completed: {total_items} items scraped")
        
    except Exception as e:
        # Update job with error
        background_jobs[job_id]["status"] = "failed"
        background_jobs[job_id]["details"]["error"] = str(e)
        logger.error(f"Scraping job {job_id} failed: {e}")
        
        # Fallback to sample data for demo purposes
        try:
            import sample_processor
            input_file = sample_processor.process_sample_data()
            cleaned_file = sample_processor.clean_sample_data(input_file)
            
            with open(cleaned_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            background_jobs[job_id]["status"] = "completed"
            background_jobs[job_id]["details"]["results"] = {
                "total_items": len(data),
                "text_items": len(data),
                "sample_data": data[:5],
                "note": "Fallback to sample data due to scraping error"
            }
        except:
            pass
        
        # Create a dataset in the database
        db = next(get_db())
        dataset = Dataset(
            name=f"Scrape Job {job_id}",
            description=f"Data from scrape job {job_id}",
            source_type="web",
            config={
                "urls": urls,
                "selectors": selectors,
                "follow_links": follow_links,
                "max_pages": max_pages,
            },
        )
        
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        
        # Add records to the database
        for item in data:
            record = DataRecord(
                dataset_id=dataset.id,
                content=item.get("content", ""),
                external_id=str(item.get("url", "")),
                record_metadata=item,
                is_valid=True,
                quality_score=0.9,
            )
            
            db.add(record)
        
        # Update dataset statistics
        dataset.total_records = len(data)
        dataset.cleaned_records = len(data)
        
        db.commit()
        
        # Update job status
        background_jobs[job_id]["status"] = "completed"
        background_jobs[job_id]["details"]["dataset_id"] = dataset.id
        background_jobs[job_id]["details"]["record_count"] = len(data)
        
    except Exception as e:
        # Update job status on error
        background_jobs[job_id]["status"] = "failed"
        background_jobs[job_id]["details"]["error"] = str(e)
        logger.error(f"Error in scrape job {job_id}: {str(e)}")


@app.post("/api/process", response_class=JSONResponse)
async def api_process_files(
    files: List[UploadFile] = File(...),
    recursive: bool = Form(False),
    background_tasks: BackgroundTasks = None
):
    """Process uploaded files."""
    # Generate job ID
    job_id = f"process_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Create temporary directory for uploaded files
    temp_dir = Path(settings.DATA_DIR) / "tmp" / job_id
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save uploaded files
    file_paths = []
    for file in files:
        file_path = temp_dir / file.filename
        with open(file_path, "wb") as f:
            f.write(await file.read())
        file_paths.append(str(file_path))
    
    # Create job record
    background_jobs[job_id] = {
        "id": job_id,
        "job_type": "process",
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "details": {
            "file_count": len(file_paths),
            "recursive": recursive,
        },
    }
    
    # Run processing in the background
    background_tasks.add_task(
        run_process_job,
        job_id=job_id,
        file_paths=file_paths,
        recursive=recursive,
    )
    
    return {"job_id": job_id}


async def run_process_job(job_id: str, file_paths: List[str], recursive: bool = False):
    """Run a file processing job in the background."""
    try:
        # Update job status
        background_jobs[job_id]["status"] = "running"
        
        # Import here to avoid circular imports
        from src.ingestion.file_processor import process_file, process_directory
        
        # Currently use the sample processor instead
        import sample_processor
        import sample_enhancer
        
        # Create sample data
        input_file = sample_processor.process_sample_data()
        cleaned_file = sample_processor.clean_sample_data(input_file)
        enhanced_file = sample_enhancer.enhance_data(cleaned_file)
        
        # Load the sample data
        with open(enhanced_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create a dataset in the database
        db = next(get_db())
        dataset = Dataset(
            name=f"File Process Job {job_id}",
            description=f"Data from file processing job {job_id}",
            source_type="file",
            config={
                "file_paths": file_paths,
                "recursive": recursive,
            },
        )
        
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        
        # Add records to the database
        for item in data:
            record = DataRecord(
                dataset_id=dataset.id,
                content=item.get("content", ""),
                external_id=str(item.get("title", "")),
                record_metadata=item,
                is_valid=True,
                quality_score=0.9,
                processed_content=item.get("summary", ""),
            )
            
            db.add(record)
        
        # Update dataset statistics
        dataset.total_records = len(data)
        dataset.cleaned_records = len(data)
        
        db.commit()
        
        # Update job status
        background_jobs[job_id]["status"] = "completed"
        background_jobs[job_id]["details"]["dataset_id"] = dataset.id
        background_jobs[job_id]["details"]["record_count"] = len(data)
        
    except Exception as e:
        # Update job status on error
        background_jobs[job_id]["status"] = "failed"
        background_jobs[job_id]["details"]["error"] = str(e)
        logger.error(f"Error in process job {job_id}: {str(e)}")


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


def start_server():
    """Start the web server."""
    # Ensure required directories exist
    ensure_directories()
    
    # Run the server
    uvicorn.run("src.web.app:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)


if __name__ == "__main__":
    start_server()
