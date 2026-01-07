"""
RoofVision Measurement Data Models

Defines the core data structures for roof measurements, facets, and job processing.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Job processing status"""
    QUEUED = "queued"
    UPLOADING = "uploading"
    VALIDATING = "validating"
    RECONSTRUCTING = "reconstructing"
    EXTRACTING = "extracting"
    MEASURING = "measuring"
    VALIDATING_RESULTS = "validating_results"
    GENERATING_REPORT = "generating_report"
    COMPLETED = "completed"
    FAILED = "failed"


class ConfidenceLevel(str, Enum):
    """Confidence level classification"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PhotoQuality(str, Enum):
    """Photo quality assessment"""
    GOOD = "good"
    WARNING = "warning"
    POOR = "poor"


# Pydantic models for API requests/responses

class PhotoValidation(BaseModel):
    """Photo quality validation result"""
    photo_id: str
    filename: str
    quality: PhotoQuality
    blur_score: float = Field(ge=0, description="Laplacian variance, higher is sharper")
    resolution_mp: float = Field(ge=0, description="Resolution in megapixels")
    angle_detected: Optional[str] = None
    issues: List[str] = Field(default_factory=list)
    is_usable: bool = True


class RoofFacet(BaseModel):
    """Individual roof facet/plane measurement"""
    facet_id: int
    pitch_degrees: float = Field(ge=0, le=90, description="Pitch angle in degrees")
    pitch_12: float = Field(ge=0, description="Pitch in X:12 format")
    azimuth_degrees: float = Field(ge=0, lt=360, description="Compass direction roof faces")
    area_sqft: float = Field(ge=0, description="Area in square feet")
    area_m2: float = Field(ge=0, description="Area in square meters")
    point_count: int = Field(ge=0, description="Number of points in facet")
    center: Tuple[float, float, float] = Field(description="Center point (x, y, z)")
    confidence: float = Field(ge=0, le=100, description="Facet confidence score")
    
    class Config:
        json_schema_extra = {
            "example": {
                "facet_id": 1,
                "pitch_degrees": 26.57,
                "pitch_12": 6.0,
                "azimuth_degrees": 180.0,
                "area_sqft": 450.5,
                "area_m2": 41.85,
                "point_count": 15000,
                "center": (10.5, 5.2, 8.3),
                "confidence": 92.5
            }
        }


class MeasurementResult(BaseModel):
    """Complete roof measurement result"""
    job_id: str
    address: str
    
    # Aggregate measurements
    total_area_sqft: float = Field(ge=0, description="Total roof area in square feet")
    total_area_m2: float = Field(ge=0, description="Total roof area in square meters")
    roofing_squares: float = Field(ge=0, description="Roofing squares (area/100)")
    ground_footprint_sqft: float = Field(ge=0, description="Ground footprint area")
    
    # Pitch information
    dominant_pitch_12: float = Field(ge=0, description="Dominant pitch in X:12 format")
    dominant_pitch_degrees: float = Field(ge=0, le=90, description="Dominant pitch in degrees")
    
    # Per-facet breakdown
    facets: List[RoofFacet] = Field(default_factory=list)
    facet_count: int = Field(ge=0)
    
    # Photo analysis
    photos_used: int = Field(ge=0)
    photos_rejected: int = Field(ge=0)
    photo_validations: List[PhotoValidation] = Field(default_factory=list)
    
    # Confidence
    confidence_score: float = Field(ge=0, le=100)
    confidence_level: ConfidenceLevel
    confidence_reasons: List[str] = Field(default_factory=list)
    
    # Validation
    google_validated: bool = False
    google_variance_percent: Optional[float] = None
    
    # Metadata
    processing_time_seconds: float = Field(ge=0)
    pipeline_version: str = "1.0.0"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_abc123",
                "address": "123 Main St, Austin, TX 78701",
                "total_area_sqft": 2450.75,
                "total_area_m2": 227.67,
                "roofing_squares": 24.51,
                "ground_footprint_sqft": 1850.0,
                "dominant_pitch_12": 6.0,
                "dominant_pitch_degrees": 26.57,
                "facets": [],
                "facet_count": 4,
                "photos_used": 8,
                "photos_rejected": 0,
                "photo_validations": [],
                "confidence_score": 87.5,
                "confidence_level": "high",
                "confidence_reasons": ["Full 360° coverage", "High quality photos"],
                "google_validated": True,
                "google_variance_percent": 2.3,
                "processing_time_seconds": 245.5,
                "pipeline_version": "1.0.0"
            }
        }


class JobProgress(BaseModel):
    """Job processing progress update"""
    job_id: str
    status: JobStatus
    progress_percent: int = Field(ge=0, le=100)
    current_stage: str
    stage_progress: int = Field(ge=0, le=100)
    estimated_seconds_remaining: Optional[int] = None
    error_message: Optional[str] = None


class GoogleSolarData(BaseModel):
    """Google Solar API response data"""
    whole_roof_area_m2: Optional[float] = None
    ground_area_m2: Optional[float] = None
    max_sunshine_hours: Optional[float] = None
    imagery_quality: Optional[str] = None
    segments: List[dict] = Field(default_factory=list)
    available: bool = True


class ValidationResult(BaseModel):
    """Validation against Google Solar API"""
    validated: bool
    our_area_m2: float
    google_area_m2: Optional[float]
    variance_percent: Optional[float]
    confidence_adjustment: int = 0
    reason: str


# Dataclasses for internal processing

@dataclass
class PlaneEquation:
    """3D plane equation: ax + by + cz + d = 0"""
    a: float
    b: float
    c: float
    d: float
    
    def normal_vector(self) -> Tuple[float, float, float]:
        """Return normalized normal vector"""
        import numpy as np
        norm = np.sqrt(self.a**2 + self.b**2 + self.c**2)
        return (self.a/norm, self.b/norm, self.c/norm)


@dataclass
class ProcessingMetrics:
    """Metrics from COLMAP reconstruction"""
    registered_images: int = 0
    total_images: int = 0
    sparse_points: int = 0
    dense_points: int = 0
    mean_reprojection_error: float = 0.0
    reconstruction_success: bool = False
    processing_time_seconds: float = 0.0


@dataclass
class ConfidenceFactors:
    """Factors contributing to confidence score"""
    photo_quality_score: float = 0.0  # 0-100
    angle_coverage_score: float = 0.0  # 0-100
    reconstruction_stability: float = 0.0  # 0-100
    facet_consistency: float = 0.0  # 0-100
    google_validation_score: float = 0.0  # 0-100
    
    def calculate_overall(self) -> float:
        """Calculate weighted overall confidence"""
        weights = {
            'photo_quality': 0.20,
            'angle_coverage': 0.25,
            'reconstruction': 0.25,
            'facet_consistency': 0.15,
            'google_validation': 0.15
        }
        
        score = (
            self.photo_quality_score * weights['photo_quality'] +
            self.angle_coverage_score * weights['angle_coverage'] +
            self.reconstruction_stability * weights['reconstruction'] +
            self.facet_consistency * weights['facet_consistency'] +
            self.google_validation_score * weights['google_validation']
        )
        
        return min(100, max(0, score))
