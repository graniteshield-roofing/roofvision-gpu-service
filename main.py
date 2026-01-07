"""
RoofVision Measurement Service API

FastAPI application providing REST endpoints for:
- Job submission
- Photo upload
- Progress tracking
- Measurement results
- Report generation
"""

import os
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import structlog

from models.measurement import (
    JobStatus,
    JobProgress,
    MeasurementResult,
    PhotoValidation,
    ConfidenceLevel
)
from services.pipeline import process_measurement
from services.photo_validation import validate_photo

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

# Configuration
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "/tmp/roofvision/uploads"))
GOOGLE_API_KEY = os.environ.get("GOOGLE_SOLAR_API_KEY", "")
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB per file

# In-memory job storage (use Redis/DB in production)
jobs: dict = {}
job_results: dict = {}


# Request/Response Models

class JobSubmitRequest(BaseModel):
    """Request to submit a new measurement job."""
    address: str = Field(..., min_length=5, max_length=500)
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    lead_id: Optional[str] = None
    notes: Optional[str] = None


class JobSubmitResponse(BaseModel):
    """Response after job submission."""
    job_id: str
    status: JobStatus
    message: str
    upload_url: str


class JobStatusResponse(BaseModel):
    """Job status response."""
    job_id: str
    status: JobStatus
    progress_percent: int
    current_stage: str
    stage_progress: int
    estimated_seconds_remaining: Optional[int]
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime


class PhotoUploadResponse(BaseModel):
    """Response after photo upload."""
    photo_id: str
    filename: str
    validation: PhotoValidation
    job_id: str


# Application lifecycle

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    # Startup
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("measurement_service_started", upload_dir=str(UPLOAD_DIR))
    
    yield
    
    # Shutdown
    logger.info("measurement_service_stopped")


# Create FastAPI app
app = FastAPI(
    title="RoofVision Measurement API",
    description="Production-grade roof measurement service using photogrammetry",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Helper functions

def get_job_dir(job_id: str) -> Path:
    """Get the directory for a job's files."""
    return UPLOAD_DIR / job_id


def update_job_progress(progress: JobProgress):
    """Update job progress in storage."""
    if progress.job_id in jobs:
        jobs[progress.job_id].update({
            "status": progress.status,
            "progress_percent": progress.progress_percent,
            "current_stage": progress.current_stage,
            "stage_progress": progress.stage_progress,
            "error_message": progress.error_message,
            "updated_at": datetime.utcnow()
        })


async def run_measurement_pipeline(job_id: str):
    """Background task to run measurement pipeline."""
    job = jobs.get(job_id)
    if not job:
        return
    
    job_dir = get_job_dir(job_id)
    image_paths = list(job_dir.glob("*.jpg")) + list(job_dir.glob("*.jpeg")) + list(job_dir.glob("*.png"))
    
    if len(image_paths) < 4:
        jobs[job_id]["status"] = JobStatus.FAILED
        jobs[job_id]["error_message"] = "Need at least 4 photos"
        return
    
    try:
        result = await process_measurement(
            job_id=job_id,
            address=job["address"],
            image_paths=[str(p) for p in image_paths],
            latitude=job.get("latitude"),
            longitude=job.get("longitude"),
            google_api_key=GOOGLE_API_KEY,
            progress_callback=update_job_progress
        )
        
        job_results[job_id] = result
        jobs[job_id]["status"] = JobStatus.COMPLETED
        
    except Exception as e:
        logger.error("pipeline_failed", job_id=job_id, error=str(e))
        jobs[job_id]["status"] = JobStatus.FAILED
        jobs[job_id]["error_message"] = str(e)


# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.post("/jobs", response_model=JobSubmitResponse)
async def create_job(request: JobSubmitRequest):
    """
    Create a new measurement job.
    
    Returns a job ID and upload URL for photos.
    """
    job_id = str(uuid.uuid4())[:8]
    
    # Create job directory
    job_dir = get_job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    
    # Store job
    jobs[job_id] = {
        "job_id": job_id,
        "address": request.address,
        "latitude": request.latitude,
        "longitude": request.longitude,
        "lead_id": request.lead_id,
        "notes": request.notes,
        "status": JobStatus.QUEUED,
        "progress_percent": 0,
        "current_stage": "Waiting for photos",
        "stage_progress": 0,
        "error_message": None,
        "photo_count": 0,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    logger.info("job_created", job_id=job_id, address=request.address)
    
    return JobSubmitResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        message="Job created. Upload photos to begin processing.",
        upload_url=f"/jobs/{job_id}/photos"
    )


@app.post("/jobs/{job_id}/photos", response_model=PhotoUploadResponse)
async def upload_photo(
    job_id: str,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload a photo for a measurement job.
    
    Photos are validated on upload. After uploading all photos,
    call POST /jobs/{job_id}/start to begin processing.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] not in [JobStatus.QUEUED, JobStatus.UPLOADING]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot upload photos. Job status: {job['status']}"
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save file
    job_dir = get_job_dir(job_id)
    photo_id = str(uuid.uuid4())[:8]
    ext = Path(file.filename).suffix or ".jpg"
    file_path = job_dir / f"{photo_id}{ext}"
    
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")
    
    with open(file_path, "wb") as f:
        f.write(content)
    
    # Validate photo
    validation = validate_photo(str(file_path), photo_id)
    
    # Update job
    job["photo_count"] = job.get("photo_count", 0) + 1
    job["status"] = JobStatus.UPLOADING
    job["updated_at"] = datetime.utcnow()
    
    logger.info(
        "photo_uploaded",
        job_id=job_id,
        photo_id=photo_id,
        quality=validation.quality.value
    )
    
    return PhotoUploadResponse(
        photo_id=photo_id,
        filename=file.filename,
        validation=validation,
        job_id=job_id
    )


@app.post("/jobs/{job_id}/start")
async def start_processing(
    job_id: str,
    background_tasks: BackgroundTasks
):
    """
    Start processing a measurement job.
    
    Call this after uploading all photos.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] not in [JobStatus.QUEUED, JobStatus.UPLOADING]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot start processing. Job status: {job['status']}"
        )
    
    if job.get("photo_count", 0) < 4:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 4 photos. Current: {job.get('photo_count', 0)}"
        )
    
    # Update status
    job["status"] = JobStatus.VALIDATING
    job["current_stage"] = "Starting processing"
    job["updated_at"] = datetime.utcnow()
    
    # Start background processing
    background_tasks.add_task(run_measurement_pipeline, job_id)
    
    logger.info("processing_started", job_id=job_id)
    
    return {
        "job_id": job_id,
        "status": JobStatus.VALIDATING,
        "message": "Processing started"
    }


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the current status of a measurement job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress_percent=job.get("progress_percent", 0),
        current_stage=job.get("current_stage", ""),
        stage_progress=job.get("stage_progress", 0),
        estimated_seconds_remaining=None,  # Could calculate based on progress
        error_message=job.get("error_message"),
        created_at=job["created_at"],
        updated_at=job["updated_at"]
    )


@app.get("/jobs/{job_id}/result", response_model=MeasurementResult)
async def get_job_result(job_id: str):
    """Get the measurement result for a completed job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Status: {job['status']}"
        )
    
    if job_id not in job_results:
        raise HTTPException(status_code=404, detail="Result not found")
    
    return job_results[job_id]


@app.get("/jobs/{job_id}/report")
async def get_job_report(job_id: str, format: str = Query("pdf", enum=["pdf", "json"])):
    """
    Get the measurement report for a completed job.
    
    Supports PDF and JSON formats.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job_id not in job_results:
        raise HTTPException(status_code=404, detail="Result not found")
    
    result = job_results[job_id]
    
    if format == "json":
        return result
    
    # PDF generation would go here
    # For now, return JSON with a note
    return JSONResponse(
        content={
            "message": "PDF generation available in production",
            "result": result.model_dump()
        }
    )


@app.get("/jobs")
async def list_jobs(
    status: Optional[JobStatus] = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """List all jobs with optional status filter."""
    filtered = list(jobs.values())
    
    if status:
        filtered = [j for j in filtered if j["status"] == status]
    
    # Sort by created_at descending
    filtered.sort(key=lambda x: x["created_at"], reverse=True)
    
    return {
        "total": len(filtered),
        "jobs": filtered[offset:offset + limit]
    }


# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
