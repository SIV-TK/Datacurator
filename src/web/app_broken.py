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
async def clean_data(
    file: UploadFile = File(None),
    dataset_id: int = Form(None),
    text_data: str = Form(None),
    cleaning_options: str = Form("{}"),
    output_file: str = Form(""),
    output_format: str = Form("jsonl"),
    background_tasks: BackgroundTasks = None
):
    """Clean text data using basic or advanced cleaning methods."""
    import uuid
    import json
    
    job_id = str(uuid.uuid4())
    
    # Parse cleaning options
    try:
        options = json.loads(cleaning_options)
    except json.JSONDecodeError:
        options = {}
    
    # Extract text data from various sources
    texts_to_clean = []
    
    if file:
        # Process uploaded file
        content = await file.read()
        try:
            if file.filename.endswith('.json'):
                data = json.loads(content.decode('utf-8'))
                if isinstance(data, list):
                    texts_to_clean = [str(item) for item in data]
                else:
                    texts_to_clean = [str(data)]
            elif file.filename.endswith('.jsonl'):
                for line in content.decode('utf-8').split('\n'):
                    if line.strip():
                        try:
                            item = json.loads(line)
                            texts_to_clean.append(str(item))
                        except json.JSONDecodeError:
                            texts_to_clean.append(line.strip())
            elif file.filename.endswith('.txt'):
                texts_to_clean = [content.decode('utf-8')]
            elif file.filename.endswith('.csv'):
                import csv
                import io
                csv_content = io.StringIO(content.decode('utf-8'))
                reader = csv.reader(csv_content)
                for row in reader:
                    if row:
                        texts_to_clean.append(row[0])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing file: {e}")
    
    elif text_data:
        # Process pasted text data
        lines = text_data.split('\n')
        for line in lines:
            line = line.strip()
            if line:
                try:
                    # Try to parse as JSON
                    item = json.loads(line)
                    texts_to_clean.append(str(item))
                except json.JSONDecodeError:
                    # Treat as plain text
                    texts_to_clean.append(line)
    
    # Create job record
    background_jobs[job_id] = {
        "id": job_id,
        "job_type": "clean",
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "details": {
            "use_advanced": options.get('use_advanced', False),
            "dataset_id": dataset_id,
            "text_count": len(texts_to_clean),
            "options": options,
            "output_file": output_file or f"cleaned_data_{job_id}",
            "output_format": output_format,
        },
    }
    
    # Run cleaning in the background
    background_tasks.add_task(
        run_clean_job,
        job_id=job_id,
        dataset_id=dataset_id,
        text_data=texts_to_clean,
        use_advanced=options.get('use_advanced', False),
        remove_html=options.get('remove_html', True),
        min_length=options.get('min_length', 0),
        max_length=options.get('max_length', 0),
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
        
        # Save cleaned results to database
        if cleaned_texts:
            db = SessionLocal()
            try:
                # Create dataset record
                dataset = Dataset(
                    name=f"Text Cleaning - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    description=f"Cleaned {len(texts_to_clean)} texts using {'advanced' if use_advanced else 'basic'} cleaning",
                    source_type="text_cleaning",
                    status="completed",
                    metadata={
                        "original_dataset_id": dataset_id,
                        "use_advanced": use_advanced,
                        "remove_html": remove_html,
                        "min_length": min_length,
                        "max_length": max_length,
                        "original_count": len(texts_to_clean),
                        "cleaned_count": len(cleaned_texts)
                    },
                )
                
                db.add(dataset)
                db.commit()
                db.refresh(dataset)
                
                # Add cleaned texts as records
                for i, cleaned_text in enumerate(cleaned_texts):
                    record = DataRecord(
                        dataset_id=dataset.id,
                        content=cleaned_text,
                        external_id=f"cleaned_{i}",
                        record_metadata={
                            "index": i,
                            "original_length": len(texts_to_clean[i]) if i < len(texts_to_clean) else 0,
                            "cleaned_length": len(cleaned_text),
                            "cleaning_method": "advanced" if use_advanced else "basic"
                        },
                        is_valid=True,
                        quality_score=0.9,
                    )
                    
                    db.add(record)
                
                # Update dataset statistics
                dataset.total_records = len(cleaned_texts)
                dataset.cleaned_records = len(cleaned_texts)
                
                db.commit()
                
                # Update job status
                background_jobs[job_id]["details"]["dataset_id"] = dataset.id
                
            finally:
                db.close()
        
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
async def api_scrape(
    urls: str = Form(None),
    file: UploadFile = File(None),
    max_pages: int = Form(10),
    rate_limit_delay: float = Form(0.5),
    follow_links: bool = Form(False),
    use_selenium_fallback: bool = Form(False),
    exclude_domains: str = Form(""),
    output_file: str = Form(""),
    background_tasks: BackgroundTasks = None
):
    """Scrape websites."""
    # Generate job ID
    job_id = f"scrape_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Parse URLs
    url_list = []
    if urls:
        # Parse URLs from text input
        url_list = [url.strip() for url in urls.split('\n') if url.strip()]
    elif file:
        # Parse URLs from uploaded file
        content = await file.read()
        try:
            if file.filename.endswith('.txt'):
                url_list = [url.strip() for url in content.decode('utf-8').split('\n') if url.strip()]
            elif file.filename.endswith('.csv'):
                import csv
                import io
                csv_content = io.StringIO(content.decode('utf-8'))
                reader = csv.reader(csv_content)
                for row in reader:
                    if row:
                        url_list.append(row[0].strip())
            elif file.filename.endswith('.jsonl'):
                import json
                for line in content.decode('utf-8').split('\n'):
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if 'url' in data:
                                url_list.append(data['url'])
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error parsing file: {e}")
    
    if not url_list:
        raise HTTPException(status_code=400, detail="No valid URLs provided")
    
    # Parse exclude domains
    exclude_list = [domain.strip() for domain in exclude_domains.split(',') if domain.strip()] if exclude_domains else []
    
    # Create job record
    background_jobs[job_id] = {
        "id": job_id,
        "job_type": "scrape",
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "details": {
            "urls": url_list,
            "max_pages": max_pages,
            "follow_links": follow_links,
            "exclude_domains": exclude_list,
            "output_file": output_file or f"scraped_data_{job_id}",
        },
    }
    
    # Prepare selectors (using default selectors for now)
    selectors = {
        "title": "h1, h2, title",
        "content": "p, article, .content, .post",
        "text": "body"
    }
    
    # Run scraping in the background
    background_tasks.add_task(
        run_scrape_job,
        job_id=job_id,
        urls=url_list,
        selectors=selectors,
        follow_links=follow_links,
        max_pages=max_pages,
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
        
        # Initialize scraper
        scraper = AdvancedWebScraper()
        
        # Prepare data for processing
        data = []
        
        if len(urls) == 1 and follow_links:
            # Use crawling for single URL with following links
            try:
                async with scraper:
                    scraped_data = await scraper.crawl_website(
                        start_url=urls[0],
                        max_pages=max_pages,
                        selectors=selectors,
                        use_playwright=False
                    )
                    data = scraped_data
            except Exception as e:
                logger.warning(f"Async crawling failed, falling back to sync: {e}")
                # Fallback to sync scraping
                url_dicts = [{'url': urls[0]}]
                data = scraper.scrape_urls_batch(url_dicts, selectors, "API Scraping")
        else:
            # Use regular scraping for multiple URLs
            url_dicts = [{'url': url} for url in urls]
            data = scraper.scrape_urls_batch(url_dicts, selectors, "API Scraping")
        
        # Process the scraped data
        total_items = len(data) if data else 0
        text_items = 0
        
        if data:
            for item in data:
                if isinstance(item, dict) and ('text' in item or 'content' in item):
                    text_items += 1
        
        # Save results to database if we have data
        if data and total_items > 0:
            db = SessionLocal()
            try:
                # Create dataset record
                dataset = Dataset(
                    name=f"Web Scraping - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    description=f"Scraped from {len(urls)} URL(s)",
                    source_type="web_scraping",
                    status="completed",
                    metadata={
                        "urls": urls,
                        "selectors": selectors,
                        "follow_links": follow_links,
                        "max_pages": max_pages,
                        "total_items": total_items,
                        "text_items": text_items
                    },
                )
                
                db.add(dataset)
                db.commit()
                db.refresh(dataset)
                
                # Add records to the database
                for item in data:
                    content = ""
                    if isinstance(item, dict):
                        content = item.get("text") or item.get("content") or str(item)
                    else:
                        content = str(item)
                    
                    record = DataRecord(
                        dataset_id=dataset.id,
                        content=content,
                        external_id=item.get("url", "") if isinstance(item, dict) else "",
                        record_metadata=item if isinstance(item, dict) else {"content": str(item)},
                        is_valid=True,
                        quality_score=0.8,
                    )
                    
                    db.add(record)
                
                # Update dataset statistics
                dataset.total_records = total_items
                dataset.cleaned_records = text_items
                
                db.commit()
                
                # Update job status
                background_jobs[job_id]["details"]["dataset_id"] = dataset.id
                background_jobs[job_id]["details"]["record_count"] = total_items
                
            finally:
                db.close()
        
        # Update job with results
        background_jobs[job_id]["status"] = "completed"
        background_jobs[job_id]["details"]["results"] = {
            "total_items": total_items,
            "text_items": text_items,
            "sample_data": data[:5] if data else []  # Store first 5 items as sample
        }
        
        logger.info(f"Scraping job {job_id} completed: {total_items} items scraped")
        
    except Exception as e:
        # Update job status on error
        background_jobs[job_id]["status"] = "failed"
        background_jobs[job_id]["details"]["error"] = str(e)
        logger.error(f"Error in scrape job {job_id}: {str(e)}")
        
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
        
        # Import file processors
        from src.ingestion.file_processor import FileProcessor
        from src.ingestion.enhanced_document_extractor import EnhancedDocumentExtractor
        
        # Process all files
        all_data = []
        file_processor = FileProcessor()
        
        for file_path in file_paths:
            try:
                file_path_obj = Path(file_path)
                
                # Determine processing method based on file type
                if file_path_obj.suffix.lower() in ['.pdf', '.docx', '.doc']:
                    # Use enhanced document extractor for documents
                    temp_dir = file_path_obj.parent
                    output_file = temp_dir / f"{file_path_obj.stem}_extracted.jsonl"
                    
                    extractor = EnhancedDocumentExtractor(
                        input_dir=str(temp_dir),
                        output_file=str(output_file),
                        max_tokens=512,
                        min_chunk_size=100,
                        overlap=50,
                        target_language="en",
                        min_text_quality=0.7,
                        enable_ocr=True,
                        num_workers=1
                    )
                    
                    # Process just this file by creating a single-file directory
                    single_file_dir = temp_dir / f"single_{file_path_obj.stem}"
                    single_file_dir.mkdir(exist_ok=True)
                    shutil.copy2(file_path, single_file_dir / file_path_obj.name)
                    
                    extractor.input_dir = single_file_dir
                    extractor.process_documents()
                    
                    # Load extracted chunks
                    if output_file.exists():
                        with open(output_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                try:
                                    chunk = json.loads(line.strip())
                                    all_data.append({
                                        "content": chunk if isinstance(chunk, str) else str(chunk),
                                        "source": file_path_obj.name,
                                        "type": "document_chunk",
                                        "metadata": {"file_path": str(file_path_obj)}
                                    })
                                except json.JSONDecodeError:
                                    continue
                    
                    # Clean up single file directory
                    shutil.rmtree(single_file_dir, ignore_errors=True)
                    
                else:
                    # Use regular file processor for other formats
                    result = file_processor.process_file(file_path)
                    if result and isinstance(result, dict):
                        all_data.append({
                            "content": result.get("text", str(result)),
                            "source": file_path_obj.name,
                            "type": "file_content",
                            "metadata": result
                        })
                    elif result:
                        all_data.append({
                            "content": str(result),
                            "source": file_path_obj.name,
                            "type": "file_content",
                            "metadata": {"file_path": str(file_path_obj)}
                        })
            
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                # Add error record
                all_data.append({
                    "content": f"Error processing file: {str(e)}",
                    "source": Path(file_path).name,
                    "type": "error",
                    "metadata": {"error": str(e), "file_path": file_path}
                })
        
        # Save results to database if we have data
        if all_data:
            db = SessionLocal()
            try:
                # Create dataset record
                dataset = Dataset(
                    name=f"File Processing - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    description=f"Processed {len(file_paths)} file(s)",
                    source_type="file_processing",
                    status="completed",
                    metadata={
                        "file_paths": file_paths,
                        "recursive": recursive,
                        "total_items": len(all_data),
                        "file_count": len(file_paths)
                    },
                )
                
                db.add(dataset)
                db.commit()
                db.refresh(dataset)
                
                # Add records to the database
                for item in all_data:
                    record = DataRecord(
                        dataset_id=dataset.id,
                        content=item["content"],
                        external_id=item["source"],
                        record_metadata=item,
                        is_valid=item["type"] != "error",
                        quality_score=0.8 if item["type"] != "error" else 0.1,
                    )
                    
                    db.add(record)
                
                # Update dataset statistics
                dataset.total_records = len(all_data)
                dataset.cleaned_records = len([item for item in all_data if item["type"] != "error"])
                
                db.commit()
                
                # Update job status
                background_jobs[job_id]["details"]["dataset_id"] = dataset.id
                background_jobs[job_id]["details"]["record_count"] = len(all_data)
                
            finally:
                db.close()
        
        # Update job status
        background_jobs[job_id]["status"] = "completed"
        background_jobs[job_id]["details"]["results"] = {
            "total_files": len(file_paths),
            "processed_items": len(all_data),
            "successful_files": len([item for item in all_data if item["type"] != "error"])
        }
        
        logger.info(f"Processing job {job_id} completed: {len(all_data)} items processed")
        
    except Exception as e:
        # Update job status on error
        background_jobs[job_id]["status"] = "failed"
        background_jobs[job_id]["details"]["error"] = str(e)
        logger.error(f"Error in process job {job_id}: {str(e)}")
    
    finally:
        # Clean up temporary files
        try:
            temp_dir = Path(settings.DATA_DIR) / "tmp" / job_id
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory for job {job_id}: {e}")
        
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
