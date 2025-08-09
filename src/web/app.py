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


@app.get("/api/datasets", response_class=JSONResponse)
async def api_list_datasets():
    """Get all datasets."""
    try:
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
                "created_at": dataset.created_at.isoformat() if dataset.created_at else None,
                "updated_at": dataset.updated_at.isoformat() if dataset.updated_at else None,
            })
        
        return result
    except Exception as e:
        logger.error(f"Error listing datasets: {e}")
        return []


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


@app.post("/api/scrape")
async def api_scrape(
    urls: str = Form(None),
    file: UploadFile = File(None),
    max_pages: int = Form(10),
    follow_links: bool = Form(False),
    background_tasks: BackgroundTasks = None
):
    """Scrape websites."""
    # Generate job ID
    job_id = f"scrape_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Parse URLs
    url_list = []
    if urls:
        url_list = [url.strip() for url in urls.split('\n') if url.strip()]
    elif file:
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
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error parsing file: {e}")
    
    if not url_list:
        raise HTTPException(status_code=400, detail="No valid URLs provided")
    
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
        },
    }
    
    # Run scraping in the background (placeholder for now)
    background_tasks.add_task(run_scrape_job, job_id, url_list, max_pages, follow_links)
    
    return {"job_id": job_id}


@app.post("/api/clean")
async def api_clean(
    file: UploadFile = File(None),
    dataset_id: int = Form(None),
    text_data: str = Form(None),
    use_advanced: bool = Form(False),
    background_tasks: BackgroundTasks = None
):
    """Clean text data."""
    import uuid
    
    job_id = str(uuid.uuid4())
    
    # Extract text data
    texts_to_clean = []
    if file:
        content = await file.read()
        try:
            if file.filename.endswith('.json'):
                data = json.loads(content.decode('utf-8'))
                if isinstance(data, list):
                    texts_to_clean = [str(item) for item in data]
                else:
                    texts_to_clean = [str(data)]
            elif file.filename.endswith('.txt'):
                texts_to_clean = [content.decode('utf-8')]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing file: {e}")
    elif text_data:
        texts_to_clean = text_data.split('\n')
    
    # Create job record
    background_jobs[job_id] = {
        "id": job_id,
        "job_type": "clean",
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "details": {
            "use_advanced": use_advanced,
            "text_count": len(texts_to_clean),
        },
    }
    
    # Run cleaning in the background (placeholder for now)
    background_tasks.add_task(run_clean_job, job_id, texts_to_clean, use_advanced)
    
    return {"job_id": job_id}


@app.post("/api/process")
async def api_process_files(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """Process uploaded files."""
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
        },
    }
    
    # Run processing in the background (placeholder for now)
    background_tasks.add_task(run_process_job, job_id, file_paths)
    
    return {"job_id": job_id}


@app.post("/api/enhanced-extract")
async def api_enhanced_extract(
    files: List[UploadFile] = File(None),
    request_data: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    """Enhanced document extraction."""
    try:
        request = json.loads(request_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request data: {e}")
    
    job_id = f"enhanced_extract_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Create temporary directory for uploaded files
    temp_dir = Path(settings.DATA_DIR) / "tmp" / job_id
    os.makedirs(temp_dir, exist_ok=True)
    
    file_paths = []
    if files:
        for file in files:
            if file.filename:
                file_path = temp_dir / file.filename
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                file_paths.append(str(file_path))
    
    # Create job record
    background_jobs[job_id] = {
        "id": job_id,
        "job_type": "enhanced_extract",
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "details": {
            "processing_method": request.get("processing_method", "enhanced"),
            "files_count": len(file_paths),
        },
    }
    
    # Run enhanced extraction in the background (placeholder for now)
    background_tasks.add_task(run_enhanced_extraction_job, job_id, file_paths, request)
    
    return {"job_id": job_id}


# Background job functions (actual implementations)
async def run_scrape_job(job_id: str, urls: List[str], max_pages: int, follow_links: bool):
    """Run a scraping job in the background."""
    try:
        background_jobs[job_id]["status"] = "running"
        
        # Import the advanced scraper
        from src.ingestion.advanced_web_scraper import AdvancedWebScraper
        
        # Initialize scraper
        scraper = AdvancedWebScraper()
        
        # Scrape URLs
        url_dicts = [{'url': url} for url in urls]
        data = scraper.scrape_urls_batch(url_dicts, None, "Web UI Scraping")
        
        # Save to database
        db = SessionLocal()
        try:
            dataset = Dataset(
                name=f"Web Scraping - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                description=f"Scraped from {len(urls)} URL(s)",
                source_type="web_scraping",
                config={
                    "urls": urls,
                    "max_pages": max_pages,
                    "follow_links": follow_links,
                    "total_items": len(data) if data else 0
                }
            )
            
            db.add(dataset)
            db.commit()
            db.refresh(dataset)
            
            if data:
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
                
                dataset.total_records = len(data)
                dataset.cleaned_records = len(data)
                db.commit()
                
                background_jobs[job_id]["details"]["dataset_id"] = dataset.id
            
        finally:
            db.close()
        
        background_jobs[job_id]["status"] = "completed"
        background_jobs[job_id]["details"]["results"] = {
            "total_items": len(data) if data else 0,
            "sample_data": data[:5] if data else []
        }
        
        logger.info(f"Scraping job {job_id} completed: {len(data) if data else 0} items scraped")
        
    except Exception as e:
        background_jobs[job_id]["status"] = "failed"
        background_jobs[job_id]["details"]["error"] = str(e)
        logger.error(f"Error in scrape job {job_id}: {str(e)}")


async def run_clean_job(job_id: str, texts: List[str], use_advanced: bool):
    """Run a cleaning job in the background."""
    try:
        background_jobs[job_id]["status"] = "running"
        
        cleaned_texts = []
        
        if use_advanced:
            # Use advanced cleaning
            from src.core.advanced_cleaner import AdvancedTextCleaner
            
            with AdvancedTextCleaner() as cleaner:
                cleaned_texts = cleaner.clean_texts_batch(texts, show_progress=False)
        else:
            # Use basic cleaning
            from src.core.cleaner import TextCleaner
            
            cleaner = TextCleaner()
            
            for text in texts:
                cleaned = cleaner.clean_text(text)
                if cleaned:
                    cleaned_texts.append(cleaned)
        
        # Save results to database
        if cleaned_texts:
            db = SessionLocal()
            try:
                dataset = Dataset(
                    name=f"Text Cleaning - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    description=f"Cleaned {len(texts)} texts using {'advanced' if use_advanced else 'basic'} cleaning",
                    source_type="text_cleaning",
                    config={
                        "use_advanced": use_advanced,
                        "original_count": len(texts),
                        "cleaned_count": len(cleaned_texts)
                    },
                )
                
                db.add(dataset)
                db.commit()
                db.refresh(dataset)
                
                for i, cleaned_text in enumerate(cleaned_texts):
                    record = DataRecord(
                        dataset_id=dataset.id,
                        content=cleaned_text,
                        external_id=f"cleaned_{i}",
                        record_metadata={
                            "index": i,
                            "cleaning_method": "advanced" if use_advanced else "basic"
                        },
                        is_valid=True,
                        quality_score=0.9,
                    )
                    db.add(record)
                
                dataset.total_records = len(cleaned_texts)
                dataset.cleaned_records = len(cleaned_texts)
                db.commit()
                
                background_jobs[job_id]["details"]["dataset_id"] = dataset.id
                
            finally:
                db.close()
        
        background_jobs[job_id]["status"] = "completed"
        background_jobs[job_id]["details"]["results"] = {
            "original_count": len(texts),
            "cleaned_count": len(cleaned_texts),
            "sample_cleaned": cleaned_texts[:3] if cleaned_texts else []
        }
        
        logger.info(f"Cleaning job {job_id} completed: {len(cleaned_texts)} texts cleaned")
        
    except Exception as e:
        background_jobs[job_id]["status"] = "failed"
        background_jobs[job_id]["details"]["error"] = str(e)
        logger.error(f"Cleaning job {job_id} failed: {e}")


async def run_process_job(job_id: str, file_paths: List[str]):
    """Run a processing job in the background."""
    try:
        background_jobs[job_id]["status"] = "running"
        background_jobs[job_id]["progress"] = 0
        background_jobs[job_id]["current_file"] = ""
        background_jobs[job_id]["status_message"] = "Starting processing..."
        
        # Import file processors
        from src.ingestion.file_processor import process_file
        
        # Process all files
        all_data = []
        total_files = len(file_paths)
        
        for i, file_path in enumerate(file_paths):
            # Update progress - show starting this file
            progress = int((i / total_files) * 80)  # Reserve 20% for processing steps
            background_jobs[job_id]["progress"] = progress
            background_jobs[job_id]["current_file"] = Path(file_path).name
            background_jobs[job_id]["status_message"] = f"Loading file {i+1} of {total_files}: {Path(file_path).name}"
            
            # Add delay to make progress visible
            import asyncio
            await asyncio.sleep(1.0)  # Increased delay
            
            try:
                # Update progress during processing
                progress = int((i / total_files) * 80) + 10
                background_jobs[job_id]["progress"] = progress
                background_jobs[job_id]["status_message"] = f"Processing content from {Path(file_path).name}"
                
                file_path_obj = Path(file_path)
                result = process_file(file_path)
                
                # Another progress update
                progress = int((i / total_files) * 80) + 15
                background_jobs[job_id]["progress"] = progress
                background_jobs[job_id]["status_message"] = f"Extracting data from {Path(file_path).name}"
                await asyncio.sleep(0.5)
                file_path_obj = Path(file_path)
                result = process_file(file_path)
                
                if result and isinstance(result, dict):
                    content = result.get("content", "")
                    if isinstance(content, (list, dict)):
                        content = json.dumps(content, indent=2)
                    elif not isinstance(content, str):
                        content = str(content)
                    
                    all_data.append({
                        "content": content,
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
                all_data.append({
                    "content": f"Error processing file: {str(e)}",
                    "source": Path(file_path).name,
                    "type": "error",
                    "metadata": {"error": str(e), "file_path": file_path}
                })
        
        # Save results to database
        background_jobs[job_id]["progress"] = 85
        background_jobs[job_id]["status_message"] = "Preparing database..."
        background_jobs[job_id]["current_file"] = ""
        await asyncio.sleep(0.5)
        
        if all_data:
            background_jobs[job_id]["progress"] = 90
            background_jobs[job_id]["status_message"] = "Creating dataset..."
            await asyncio.sleep(0.5)
            
            db = SessionLocal()
            try:
                dataset = Dataset(
                    name=f"File Processing - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    description=f"Processed {len(file_paths)} file(s)",
                    source_type="file_processing",
                    config={
                        "file_paths": file_paths,
                        "total_items": len(all_data),
                        "file_count": len(file_paths)
                    },
                )
                
                db.add(dataset)
                db.commit()
                db.refresh(dataset)
                
                background_jobs[job_id]["progress"] = 95
                background_jobs[job_id]["status_message"] = f"Creating database records for {len(all_data)} items..."
                
                for i, item in enumerate(all_data):
                    record = DataRecord(
                        dataset_id=dataset.id,
                        content=item["content"],
                        external_id=item["source"],
                        record_metadata=item,
                        is_valid=item["type"] != "error",
                        quality_score=0.8 if item["type"] != "error" else 0.1,
                    )
                    db.add(record)
                
                dataset.total_records = len(all_data)
                dataset.cleaned_records = len([item for item in all_data if item["type"] != "error"])
                db.commit()
                
                background_jobs[job_id]["details"]["dataset_id"] = dataset.id
                
            finally:
                db.close()
        
        background_jobs[job_id]["progress"] = 100
        background_jobs[job_id]["status_message"] = "Processing completed successfully!"
        background_jobs[job_id]["current_file"] = ""
        background_jobs[job_id]["status"] = "completed"
        background_jobs[job_id]["details"]["results"] = {
            "total_files": len(file_paths),
            "processed_items": len(all_data),
            "successful_files": len([item for item in all_data if item["type"] != "error"])
        }
        
        # Clean up temp files
        try:
            temp_dir = Path(settings.DATA_DIR) / "tmp" / job_id
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        except Exception:
            pass
            
        logger.info(f"Processing job {job_id} completed: {len(all_data)} items processed")
            
    except Exception as e:
        background_jobs[job_id]["progress"] = 0
        background_jobs[job_id]["status_message"] = f"Processing failed: {str(e)}"
        background_jobs[job_id]["current_file"] = ""
        background_jobs[job_id]["status"] = "failed"
        background_jobs[job_id]["details"]["error"] = str(e)
        logger.error(f"Error in process job {job_id}: {str(e)}")


async def run_enhanced_extraction_job(job_id: str, file_paths: List[str], request: dict):
    """Run an enhanced extraction job in the background."""
    try:
        background_jobs[job_id]["status"] = "running"
        background_jobs[job_id]["progress"] = 0
        background_jobs[job_id]["current_file"] = ""
        background_jobs[job_id]["status_message"] = "Starting enhanced extraction..."
        
        # Import the enhanced document extractor
        from src.ingestion.enhanced_document_extractor import EnhancedDocumentExtractor
        
        # Get temp directory
        temp_dir = Path(settings.DATA_DIR) / "tmp" / job_id
        
        # Prepare output file
        output_name = request.get("output_name") or f"enhanced_extraction_{job_id}"
        output_file = Path(settings.DATA_DIR) / "exports" / f"{output_name}.jsonl"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure extractor
        target_language = None if request.get("target_language") == "auto" else request.get("target_language", "en")
        num_workers = request.get("num_workers", 4)
        if num_workers == 0:
            num_workers = None
        
        extractor = EnhancedDocumentExtractor(
            input_dir=str(temp_dir),
            output_file=str(output_file),
            max_tokens=request.get("max_tokens", 512),
            min_chunk_size=request.get("min_chunk_size", 100),
            overlap=request.get("chunk_overlap", 50),
            target_language=target_language,
            min_text_quality=request.get("text_quality", 0.7),
            enable_ocr=request.get("enable_ocr", True),
            num_workers=num_workers
        )
        
        # Run the extraction
        background_jobs[job_id]["progress"] = 50
        background_jobs[job_id]["status_message"] = "Running document extraction..."
        extractor.process_documents()
        
        background_jobs[job_id]["progress"] = 80
        background_jobs[job_id]["status_message"] = "Saving results to database..."
        
        # Count chunks and save to database
        chunk_count = 0
        if output_file.exists():
            db = SessionLocal()
            try:
                # Create dataset record with unique name
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                dataset = Dataset(
                    name=f"Enhanced Extraction - {output_name}_{timestamp}",
                    description=f"Enhanced document extraction with OCR and chunking - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    source_type="enhanced_extraction",
                    config={
                        "processing_method": request.get("processing_method", "enhanced"),
                        "parameters": request,
                        "output_file": str(output_file)
                    }
                )
                
                db.add(dataset)
                db.commit()
                db.refresh(dataset)
                
                # Add records from the output file
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
                                quality_score=request.get("text_quality", 0.7)
                            )
                            db.add(record)
                            chunk_count += 1
                        except json.JSONDecodeError:
                            continue
                
                dataset.total_records = chunk_count
                dataset.cleaned_records = chunk_count
                db.commit()
                
                background_jobs[job_id]["details"]["dataset_id"] = dataset.id
                
            finally:
                db.close()
        
        background_jobs[job_id]["status"] = "completed"
        background_jobs[job_id]["details"]["results"] = {
            "total_files": len(file_paths),
            "chunks_created": chunk_count,
            "output_file": str(output_file)
        }
        
        logger.info(f"Enhanced extraction job {job_id} completed: {chunk_count} chunks created")
        
        background_jobs[job_id]["progress"] = 100
        background_jobs[job_id]["status_message"] = "Enhanced extraction completed successfully!"
        background_jobs[job_id]["current_file"] = ""
        
    except Exception as e:
        background_jobs[job_id]["progress"] = 0
        background_jobs[job_id]["status_message"] = f"Enhanced extraction failed: {str(e)}"
        background_jobs[job_id]["current_file"] = ""
        background_jobs[job_id]["status"] = "failed"
        background_jobs[job_id]["details"]["error"] = str(e)
        logger.error(f"Enhanced extraction job {job_id} failed: {e}")
    
    finally:
        # Clean up temporary files
        try:
            temp_dir = Path(settings.DATA_DIR) / "tmp" / job_id
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"Failed to clean up temp directory for job {job_id}: {e}")


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
    
    # Run the server with proper import string
    uvicorn.run("src.web.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    start_server()
