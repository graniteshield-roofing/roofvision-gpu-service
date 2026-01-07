"""
Measurement Pipeline Orchestrator

Coordinates the complete measurement workflow:
1. Photo validation
2. 3D reconstruction (COLMAP)
3. Roof plane extraction (Open3D RANSAC)
4. Measurement calculation
5. Google Solar validation
6. Confidence scoring
7. Report generation
"""

import asyncio
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Callable
import structlog

from models.measurement import (
    JobStatus,
    JobProgress,
    MeasurementResult,
    PhotoValidation,
    RoofFacet,
    ConfidenceLevel,
    ProcessingMetrics
)

from services.photo_validation import (
    validate_photo_set,
    get_photo_quality_score
)
from services.reconstruction import run_reconstruction
from services.plane_extraction import (
    extract_roof_planes,
    calculate_aggregate_measurements,
    estimate_ground_footprint
)
from services.google_solar import (
    fetch_google_solar_data,
    validate_measurements
)
from services.confidence_engine import (
    calculate_confidence,
    get_confidence_explanation
)

logger = structlog.get_logger()


class MeasurementPipeline:
    """
    Orchestrates the complete roof measurement pipeline.
    """
    
    def __init__(
        self,
        workspace_base: str = "/tmp/roofvision",
        google_api_key: Optional[str] = None
    ):
        """
        Initialize the measurement pipeline.
        
        Args:
            workspace_base: Base directory for processing workspaces
            google_api_key: Google Solar API key for validation
        """
        self.workspace_base = Path(workspace_base)
        self.workspace_base.mkdir(parents=True, exist_ok=True)
        self.google_api_key = google_api_key
        
        self._progress_callback: Optional[Callable] = None
        self._current_job_id: Optional[str] = None
    
    def set_progress_callback(self, callback: Callable[[JobProgress], None]):
        """Set callback for progress updates."""
        self._progress_callback = callback
    
    def _update_progress(
        self,
        status: JobStatus,
        progress: int,
        stage: str,
        stage_progress: int = 0,
        error: Optional[str] = None
    ):
        """Send progress update."""
        if self._progress_callback and self._current_job_id:
            update = JobProgress(
                job_id=self._current_job_id,
                status=status,
                progress_percent=progress,
                current_stage=stage,
                stage_progress=stage_progress,
                error_message=error
            )
            self._progress_callback(update)
    
    async def process_job(
        self,
        job_id: str,
        address: str,
        image_paths: List[str],
        latitude: Optional[float] = None,
        longitude: Optional[float] = None
    ) -> MeasurementResult:
        """
        Process a complete measurement job.
        
        Args:
            job_id: Unique job identifier
            address: Property address
            image_paths: List of photo file paths
            latitude: Optional latitude for Google validation
            longitude: Optional longitude for Google validation
            
        Returns:
            MeasurementResult with all measurements
        """
        self._current_job_id = job_id
        start_time = time.time()
        
        logger.info(
            "pipeline_started",
            job_id=job_id,
            address=address,
            photo_count=len(image_paths)
        )
        
        try:
            # Stage 1: Photo Validation
            self._update_progress(
                JobStatus.VALIDATING, 5, "Validating photos", 0
            )
            
            photo_validations, missing_angles, can_proceed = validate_photo_set(image_paths)
            
            if not can_proceed:
                raise ValueError(
                    f"Insufficient usable photos. Missing angles: {missing_angles}"
                )
            
            usable_paths = [
                path for path, v in zip(image_paths, photo_validations)
                if v.is_usable
            ]
            
            self._update_progress(
                JobStatus.VALIDATING, 10, "Photo validation complete", 100
            )
            
            # Stage 2: 3D Reconstruction
            self._update_progress(
                JobStatus.RECONSTRUCTING, 15, "Starting 3D reconstruction", 0
            )
            
            def recon_progress(status, percent, stage):
                overall = 15 + int(percent * 0.35)  # 15-50%
                self._update_progress(status, overall, stage, percent)
            
            success, metrics, output = run_reconstruction(
                job_id=job_id,
                image_paths=usable_paths,
                workspace_base=str(self.workspace_base),
                progress_callback=recon_progress
            )
            
            if not success:
                raise ValueError(f"3D reconstruction failed: {output}")
            
            point_cloud_path = output
            
            # Stage 3: Roof Plane Extraction
            self._update_progress(
                JobStatus.EXTRACTING, 55, "Extracting roof planes", 0
            )
            
            def extraction_progress(status, percent, stage):
                overall = 55 + int(percent * 0.20)  # 55-75%
                self._update_progress(status, overall, stage, percent)
            
            facets = extract_roof_planes(
                point_cloud_path,
                progress_callback=extraction_progress
            )
            
            if not facets:
                raise ValueError("No roof planes could be extracted")
            
            # Stage 4: Calculate Measurements
            self._update_progress(
                JobStatus.MEASURING, 75, "Calculating measurements", 0
            )
            
            aggregate = calculate_aggregate_measurements(facets)
            ground_footprint = estimate_ground_footprint(facets)
            
            self._update_progress(
                JobStatus.MEASURING, 80, "Measurements calculated", 100
            )
            
            # Stage 5: Google Solar Validation
            google_validation = None
            google_validated = False
            google_variance = None
            
            if latitude and longitude:
                self._update_progress(
                    JobStatus.VALIDATING_RESULTS, 82, "Validating with Google Solar", 0
                )
                
                google_data = await fetch_google_solar_data(
                    latitude, longitude, self.google_api_key
                )
                
                if google_data.available:
                    google_validation = validate_measurements(
                        aggregate["total_area_m2"],
                        google_data
                    )
                    google_validated = google_validation.validated
                    google_variance = google_validation.variance_percent
                
                self._update_progress(
                    JobStatus.VALIDATING_RESULTS, 88, "Validation complete", 100
                )
            
            # Stage 6: Confidence Scoring
            self._update_progress(
                JobStatus.GENERATING_REPORT, 90, "Calculating confidence", 0
            )
            
            confidence_score, confidence_level, confidence_reasons, factors = calculate_confidence(
                photo_validations=photo_validations,
                reconstruction_metrics=metrics,
                facets=facets,
                google_validation=google_validation
            )
            
            # Stage 7: Build Result
            self._update_progress(
                JobStatus.GENERATING_REPORT, 95, "Generating report", 50
            )
            
            processing_time = time.time() - start_time
            
            result = MeasurementResult(
                job_id=job_id,
                address=address,
                
                # Aggregate measurements
                total_area_sqft=aggregate["total_area_sqft"],
                total_area_m2=aggregate["total_area_m2"],
                roofing_squares=aggregate["roofing_squares"],
                ground_footprint_sqft=ground_footprint,
                
                # Pitch
                dominant_pitch_12=aggregate["dominant_pitch_12"],
                dominant_pitch_degrees=aggregate["dominant_pitch_degrees"],
                
                # Facets
                facets=facets,
                facet_count=aggregate["facet_count"],
                
                # Photos
                photos_used=len(usable_paths),
                photos_rejected=len(image_paths) - len(usable_paths),
                photo_validations=photo_validations,
                
                # Confidence
                confidence_score=round(confidence_score, 1),
                confidence_level=confidence_level,
                confidence_reasons=confidence_reasons,
                
                # Validation
                google_validated=google_validated,
                google_variance_percent=google_variance,
                
                # Metadata
                processing_time_seconds=round(processing_time, 1),
                pipeline_version="1.0.0",
                created_at=datetime.utcnow()
            )
            
            self._update_progress(
                JobStatus.COMPLETED, 100, "Measurement complete", 100
            )
            
            logger.info(
                "pipeline_completed",
                job_id=job_id,
                total_area=result.total_area_sqft,
                squares=result.roofing_squares,
                confidence=result.confidence_score,
                time=processing_time
            )
            
            return result
            
        except Exception as e:
            logger.error("pipeline_error", job_id=job_id, error=str(e))
            
            self._update_progress(
                JobStatus.FAILED, 0, "Processing failed", 0, str(e)
            )
            
            raise


async def process_measurement(
    job_id: str,
    address: str,
    image_paths: List[str],
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    google_api_key: Optional[str] = None,
    progress_callback: Optional[Callable] = None
) -> MeasurementResult:
    """
    Convenience function to process a measurement job.
    
    Args:
        job_id: Unique job identifier
        address: Property address
        image_paths: List of photo file paths
        latitude: Optional latitude for validation
        longitude: Optional longitude for validation
        google_api_key: Google Solar API key
        progress_callback: Optional progress callback
        
    Returns:
        MeasurementResult with all measurements
    """
    pipeline = MeasurementPipeline(google_api_key=google_api_key)
    
    if progress_callback:
        pipeline.set_progress_callback(progress_callback)
    
    return await pipeline.process_job(
        job_id=job_id,
        address=address,
        image_paths=image_paths,
        latitude=latitude,
        longitude=longitude
    )


# Example usage and testing
if __name__ == "__main__":
    async def test_pipeline():
        """Test the measurement pipeline with sample data."""
        
        # This would be replaced with actual image paths
        test_images = [
            "/path/to/photo1.jpg",
            "/path/to/photo2.jpg",
            # ... 8 photos
        ]
        
        def progress_handler(progress: JobProgress):
            print(f"[{progress.status.value}] {progress.progress_percent}% - {progress.current_stage}")
        
        try:
            result = await process_measurement(
                job_id="test_001",
                address="123 Main St, Austin, TX 78701",
                image_paths=test_images,
                latitude=30.2672,
                longitude=-97.7431,
                progress_callback=progress_handler
            )
            
            print(f"\n=== MEASUREMENT RESULT ===")
            print(f"Total Area: {result.total_area_sqft} sq ft")
            print(f"Roofing Squares: {result.roofing_squares}")
            print(f"Dominant Pitch: {result.dominant_pitch_12}:12")
            print(f"Facets: {result.facet_count}")
            print(f"Confidence: {result.confidence_score} ({result.confidence_level.value})")
            
        except Exception as e:
            print(f"Pipeline failed: {e}")
    
    asyncio.run(test_pipeline())
