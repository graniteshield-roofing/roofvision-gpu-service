"""
Celery Worker for RoofVision Measurement Processing

Handles background job processing with:
- Job queue management
- Retry logic for failures
- Progress tracking via Redis
- GPU resource management
"""

import os
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

from celery import Celery
from celery.signals import task_prerun, task_postrun, task_failure
import structlog

from models.measurement import JobStatus, JobProgress, MeasurementResult
from services.pipeline import process_measurement

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

# Celery configuration
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
GOOGLE_API_KEY = os.environ.get("GOOGLE_SOLAR_API_KEY", "")

# Create Celery app
celery_app = Celery(
    "roofvision",
    broker=REDIS_URL,
    backend=REDIS_URL
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Task settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    
    # Retry settings
    task_default_retry_delay=60,
    task_max_retries=3,
    
    # Concurrency (1 per GPU typically)
    worker_concurrency=int(os.environ.get("WORKER_CONCURRENCY", "1")),
    
    # Prefetch (1 task at a time for GPU workloads)
    worker_prefetch_multiplier=1,
    
    # Task time limits
    task_soft_time_limit=1800,  # 30 minutes
    task_time_limit=2100,  # 35 minutes hard limit
    
    # Result expiry
    result_expires=86400,  # 24 hours
)


# Redis helpers for progress tracking

def get_redis_client():
    """Get Redis client for progress updates."""
    import redis
    return redis.from_url(REDIS_URL)


def publish_progress(job_id: str, progress: JobProgress):
    """Publish progress update to Redis."""
    try:
        client = get_redis_client()
        
        # Store current progress
        client.hset(f"job:{job_id}", mapping={
            "status": progress.status.value,
            "progress_percent": progress.progress_percent,
            "current_stage": progress.current_stage,
            "stage_progress": progress.stage_progress,
            "error_message": progress.error_message or "",
            "updated_at": datetime.utcnow().isoformat()
        })
        
        # Publish for real-time subscribers
        client.publish(f"job:{job_id}:progress", json.dumps({
            "job_id": job_id,
            "status": progress.status.value,
            "progress_percent": progress.progress_percent,
            "current_stage": progress.current_stage,
            "stage_progress": progress.stage_progress
        }))
        
    except Exception as e:
        logger.error("redis_publish_error", job_id=job_id, error=str(e))


def store_result(job_id: str, result: MeasurementResult):
    """Store measurement result in Redis."""
    try:
        client = get_redis_client()
        
        # Store full result
        client.set(
            f"job:{job_id}:result",
            result.model_dump_json(),
            ex=86400 * 30  # 30 days
        )
        
        # Update job status
        client.hset(f"job:{job_id}", mapping={
            "status": JobStatus.COMPLETED.value,
            "progress_percent": 100,
            "current_stage": "Complete",
            "completed_at": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error("redis_store_error", job_id=job_id, error=str(e))


# Celery signals

@task_prerun.connect
def task_prerun_handler(task_id, task, args, kwargs, **_):
    """Log task start."""
    job_id = kwargs.get("job_id") or (args[0] if args else None)
    logger.info("task_started", task_id=task_id, task=task.name, job_id=job_id)


@task_postrun.connect
def task_postrun_handler(task_id, task, args, kwargs, retval, state, **_):
    """Log task completion."""
    job_id = kwargs.get("job_id") or (args[0] if args else None)
    logger.info("task_completed", task_id=task_id, task=task.name, job_id=job_id, state=state)


@task_failure.connect
def task_failure_handler(task_id, exception, args, kwargs, traceback, einfo, **_):
    """Handle task failure."""
    job_id = kwargs.get("job_id") or (args[0] if args else None)
    logger.error(
        "task_failed",
        task_id=task_id,
        job_id=job_id,
        error=str(exception)
    )
    
    if job_id:
        publish_progress(job_id, JobProgress(
            job_id=job_id,
            status=JobStatus.FAILED,
            progress_percent=0,
            current_stage="Failed",
            stage_progress=0,
            error_message=str(exception)
        ))


# Celery tasks

@celery_app.task(
    bind=True,
    name="roofvision.process_measurement",
    max_retries=3,
    default_retry_delay=60
)
def process_measurement_task(
    self,
    job_id: str,
    address: str,
    image_paths: list,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None
):
    """
    Process a measurement job.
    
    This task runs the full measurement pipeline:
    1. Photo validation
    2. 3D reconstruction
    3. Roof plane extraction
    4. Measurement calculation
    5. Google validation
    6. Confidence scoring
    
    Args:
        job_id: Unique job identifier
        address: Property address
        image_paths: List of photo file paths
        latitude: Optional latitude for validation
        longitude: Optional longitude for validation
        
    Returns:
        Measurement result as dictionary
    """
    logger.info(
        "processing_job",
        job_id=job_id,
        address=address,
        photo_count=len(image_paths)
    )
    
    def progress_callback(progress: JobProgress):
        """Callback to publish progress updates."""
        publish_progress(job_id, progress)
    
    try:
        # Run async pipeline in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                process_measurement(
                    job_id=job_id,
                    address=address,
                    image_paths=image_paths,
                    latitude=latitude,
                    longitude=longitude,
                    google_api_key=GOOGLE_API_KEY,
                    progress_callback=progress_callback
                )
            )
        finally:
            loop.close()
        
        # Store result
        store_result(job_id, result)
        
        logger.info(
            "job_completed",
            job_id=job_id,
            total_area=result.total_area_sqft,
            confidence=result.confidence_score
        )
        
        return result.model_dump()
        
    except Exception as e:
        logger.error("job_failed", job_id=job_id, error=str(e))
        
        # Retry on certain errors
        if "reconstruction failed" in str(e).lower():
            raise self.retry(exc=e)
        
        # Mark as failed
        publish_progress(job_id, JobProgress(
            job_id=job_id,
            status=JobStatus.FAILED,
            progress_percent=0,
            current_stage="Failed",
            stage_progress=0,
            error_message=str(e)
        ))
        
        raise


@celery_app.task(name="roofvision.validate_photos")
def validate_photos_task(job_id: str, image_paths: list):
    """
    Validate photos before processing.
    
    Can be run as a separate task for quick feedback.
    
    Args:
        job_id: Job identifier
        image_paths: List of photo paths
        
    Returns:
        Validation results
    """
    from services.photo_validation import validate_photo_set
    
    validations, missing_angles, can_proceed = validate_photo_set(image_paths)
    
    return {
        "job_id": job_id,
        "validations": [v.model_dump() for v in validations],
        "missing_angles": missing_angles,
        "can_proceed": can_proceed,
        "usable_count": sum(1 for v in validations if v.is_usable)
    }


@celery_app.task(name="roofvision.generate_report")
def generate_report_task(job_id: str, format: str = "pdf"):
    """
    Generate measurement report.
    
    Args:
        job_id: Job identifier
        format: Report format (pdf, json)
        
    Returns:
        Report file path or data
    """
    # Get result from Redis
    client = get_redis_client()
    result_json = client.get(f"job:{job_id}:result")
    
    if not result_json:
        raise ValueError(f"No result found for job {job_id}")
    
    result = MeasurementResult.model_validate_json(result_json)
    
    if format == "json":
        return result.model_dump()
    
    # PDF generation would go here
    # For now, return placeholder
    return {
        "job_id": job_id,
        "format": format,
        "status": "PDF generation available in production"
    }


# Health check task
@celery_app.task(name="roofvision.health_check")
def health_check_task():
    """Worker health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "worker": os.environ.get("HOSTNAME", "unknown")
    }


# Run with: celery -A worker worker --loglevel=info
if __name__ == "__main__":
    celery_app.start()
