"""
REST API for Document AI Platform
FastAPI-based web service for document processing
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from pathlib import Path
import uuid
import tempfile
import shutil

from document_ai.utils.logging import get_logger, LoggerConfig
from document_ai.config import Config, ExportFormat, DeviceType
from document_ai.document.loader.document_loader import DocumentLoader
from document_ai.core.ocr.pipeline import OCRPipeline
from document_ai.document.exporter.document_exporter import DocumentExporter
from document_ai.exceptions import DocumentAIException

# Initialize logging
LoggerConfig.setup(log_level="INFO")
logger = get_logger(__name__)

# Create app
app = FastAPI(
    title="Document AI Platform API",
    description="Production-grade OCR and document intelligence API",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
config = Config()
document_loader = DocumentLoader(
    doc_config=config.document,
    security_config=config.security,
)
ocr_pipeline = OCRPipeline(config=config.ocr)
document_exporter = DocumentExporter()

# Job storage (use Redis/database in production)
jobs: Dict[str, Dict[str, Any]] = {}


# Request/Response Models
class OCRRequest(BaseModel):
    """OCR processing request"""
    export_format: str = Field(default="json", description="Export format (txt, json, csv, docx, pdf)")
    enable_preprocessing: bool = Field(default=True, description="Enable image preprocessing")
    enable_deskew: bool = Field(default=True, description="Enable deskewing")
    device: str = Field(default="cpu", description="Computation device (cpu, cuda, mps)")


class TextRegion(BaseModel):
    """Detected text region"""
    text: str
    confidence: float
    bounding_box: Dict[str, int]


class PageResult(BaseModel):
    """OCR result for a single page"""
    page_number: int
    full_text: str
    confidence: float
    num_regions: int
    text_regions: List[TextRegion]


class OCRResponse(BaseModel):
    """OCR processing response"""
    job_id: str
    status: str
    total_pages: int
    pages: List[PageResult]
    download_url: Optional[str] = None


class JobStatus(BaseModel):
    """Job status response"""
    job_id: str
    status: str
    progress: float
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# Endpoints
@app.get("/")
async def root():
    """API information"""
    return {
        "name": "Document AI Platform API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "ocr": "/ocr",
            "batch": "/batch",
            "jobs": "/jobs/{job_id}",
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "components": {
            "ocr_pipeline": "ready",
            "document_loader": "ready",
            "exporter": "ready",
        }
    }


@app.post("/ocr", response_model=OCRResponse)
async def process_document(
    file: UploadFile = File(...),
    request: OCRRequest = OCRRequest(),
    background_tasks: BackgroundTasks = None,
):
    """
    Process a document for OCR
    
    Upload a document (PDF or image) and extract text.
    Returns structured OCR results with bounding boxes.
    """
    job_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Job {job_id}: Processing {file.filename}")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Read file content
        content = await file.read()
        
        # Determine file type
        file_ext = Path(file.filename).suffix.lower()
        if file_ext == '.pdf':
            file_type = 'pdf'
        elif file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp']:
            file_type = file_ext[1:]
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file_ext}"
            )
        
        # Load document
        document = document_loader.load(content, file_type=file_type)
        logger.info(f"Job {job_id}: Loaded {document.total_pages} page(s)")
        
        # Update configuration
        pipeline_config = config.ocr
        pipeline_config.enable_preprocessing = request.enable_preprocessing
        pipeline_config.enable_deskew = request.enable_deskew
        pipeline_config.device = DeviceType(request.device)
        
        # Process pages
        ocr_results = []
        for i, page in enumerate(document.pages):
            logger.debug(f"Job {job_id}: Processing page {i+1}")
            result = ocr_pipeline.process_image(page.image)
            ocr_results.append(result)
        
        # Build response
        pages_data = []
        for i, result in enumerate(ocr_results):
            regions = []
            for box in result.text_boxes:
                x, y, w, h = box.bbox
                regions.append(TextRegion(
                    text=box.text,
                    confidence=box.confidence,
                    bounding_box={
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h),
                    }
                ))
            
            pages_data.append(PageResult(
                page_number=i + 1,
                full_text=result.full_text,
                confidence=result.confidence,
                num_regions=result.num_detected_regions,
                text_regions=regions,
            ))
        
        # Export if requested
        download_url = None
        if request.export_format != "json":
            try:
                export_format = ExportFormat(request.export_format)
                output_path = config.storage.output_path / f"{job_id}.{request.export_format}"
                
                document_exporter.export(
                    ocr_results,
                    output_path,
                    export_format,
                    document,
                )
                
                download_url = f"/download/{job_id}/{output_path.name}"
                
            except Exception as e:
                logger.error(f"Export failed: {e}")
        
        # Store job result
        jobs[job_id] = {
            "status": "completed",
            "result": {
                "job_id": job_id,
                "status": "completed",
                "total_pages": len(pages_data),
                "pages": [p.dict() for p in pages_data],
                "download_url": download_url,
            }
        }
        
        logger.info(f"Job {job_id}: Completed successfully")
        
        return OCRResponse(
            job_id=job_id,
            status="completed",
            total_pages=len(pages_data),
            pages=pages_data,
            download_url=download_url,
        )
    
    except HTTPException:
        raise
    except DocumentAIException as e:
        logger.error(f"Job {job_id}: Document AI error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Job {job_id}: Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/batch")
async def process_batch(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None,
):
    """
    Process multiple documents in batch
    
    Upload multiple files and get a batch job ID.
    Poll /jobs/{job_id} for status and results.
    """
    job_id = str(uuid.uuid4())
    
    # Store job
    jobs[job_id] = {
        "status": "processing",
        "total": len(files),
        "completed": 0,
        "results": [],
    }
    
    # Process in background (in production, use Celery/RQ)
    # For now, return job ID immediately
    
    return {
        "job_id": job_id,
        "status": "processing",
        "total_files": len(files),
        "message": f"Processing {len(files)} files. Check /jobs/{job_id} for status."
    }


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """
    Get job status and results
    
    Poll this endpoint to check processing status.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_data = jobs[job_id]
    
    return JobStatus(
        job_id=job_id,
        status=job_data["status"],
        progress=job_data.get("completed", 0) / job_data.get("total", 1) * 100,
        result=job_data.get("result"),
        error=job_data.get("error"),
    )


@app.get("/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    """
    Download processed file
    
    Download the exported document.
    """
    file_path = config.storage.output_path / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream",
    )


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """
    Delete job and associated files
    
    Cleanup job data and temporary files.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete associated files
    output_dir = config.storage.output_path
    for file_path in output_dir.glob(f"{job_id}.*"):
        file_path.unlink()
    
    # Remove job
    del jobs[job_id]
    
    return {"status": "deleted", "job_id": job_id}


# Exception handlers
@app.exception_handler(DocumentAIException)
async def document_ai_exception_handler(request, exc: DocumentAIException):
    """Handle Document AI exceptions"""
    return JSONResponse(
        status_code=400,
        content={
            "error": exc.message,
            "details": exc.details,
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
