"""
Production Confidence Engine

Calculates measurement confidence scores based on multiple factors:
- Photo quality assessment
- Angle coverage completeness
- 3D reconstruction stability
- Facet consistency analysis
- Google Solar API validation
"""

import numpy as np
from typing import List, Optional, Tuple
import structlog

from models.measurement import (
    ConfidenceLevel,
    ConfidenceFactors,
    PhotoValidation,
    ProcessingMetrics,
    RoofFacet,
    ValidationResult,
    PhotoQuality
)

logger = structlog.get_logger()

# Scoring weights
WEIGHTS = {
    "photo_quality": 0.20,
    "angle_coverage": 0.25,
    "reconstruction": 0.25,
    "facet_consistency": 0.15,
    "google_validation": 0.15
}

# Thresholds
REQUIRED_ANGLES = 8
MIN_REGISTERED_IMAGES = 6
MAX_REPROJECTION_ERROR = 1.0
MAX_PITCH_VARIANCE = 5.0


def calculate_photo_quality_score(
    validations: List[PhotoValidation]
) -> Tuple[float, List[str]]:
    """
    Calculate photo quality contribution to confidence.
    
    Args:
        validations: List of photo validation results
        
    Returns:
        Tuple of (score 0-100, list of reasons)
    """
    if not validations:
        return 0.0, ["No photos validated"]
    
    reasons = []
    
    # Count quality levels
    good_count = sum(1 for v in validations if v.quality == PhotoQuality.GOOD)
    warning_count = sum(1 for v in validations if v.quality == PhotoQuality.WARNING)
    poor_count = sum(1 for v in validations if v.quality == PhotoQuality.POOR)
    total = len(validations)
    
    # Calculate weighted score
    score = (good_count * 100 + warning_count * 70 + poor_count * 30) / total
    
    # Generate reasons
    if good_count == total:
        reasons.append("All photos are high quality")
    elif good_count >= total * 0.75:
        reasons.append("Most photos are high quality")
    elif poor_count > total * 0.25:
        reasons.append(f"{poor_count} photos have quality issues")
    
    # Check for specific issues
    blur_issues = sum(1 for v in validations if "blurry" in str(v.issues).lower())
    if blur_issues > 0:
        reasons.append(f"{blur_issues} photos are blurry")
    
    return score, reasons


def calculate_angle_coverage_score(
    validations: List[PhotoValidation],
    registered_images: int
) -> Tuple[float, List[str]]:
    """
    Calculate angle coverage contribution to confidence.
    
    Args:
        validations: Photo validations with detected angles
        registered_images: Number of images used in reconstruction
        
    Returns:
        Tuple of (score 0-100, list of reasons)
    """
    reasons = []
    
    # Check detected angles
    detected_angles = set()
    for v in validations:
        if v.angle_detected:
            detected_angles.add(v.angle_detected)
    
    angle_coverage = len(detected_angles) / REQUIRED_ANGLES
    
    # Check registered images
    registration_ratio = min(1.0, registered_images / MIN_REGISTERED_IMAGES)
    
    # Combined score
    score = (angle_coverage * 0.6 + registration_ratio * 0.4) * 100
    
    # Generate reasons
    if len(detected_angles) >= REQUIRED_ANGLES:
        reasons.append("Full 360° coverage achieved")
    elif len(detected_angles) >= 6:
        reasons.append(f"{len(detected_angles)}/8 angles captured")
    else:
        missing = REQUIRED_ANGLES - len(detected_angles)
        reasons.append(f"Missing {missing} capture angles")
    
    if registered_images >= MIN_REGISTERED_IMAGES:
        reasons.append(f"All {registered_images} images registered")
    elif registered_images < 4:
        reasons.append(f"Only {registered_images} images registered (need 6+)")
    
    return score, reasons


def calculate_reconstruction_score(
    metrics: ProcessingMetrics
) -> Tuple[float, List[str]]:
    """
    Calculate reconstruction stability contribution to confidence.
    
    Args:
        metrics: Processing metrics from COLMAP
        
    Returns:
        Tuple of (score 0-100, list of reasons)
    """
    reasons = []
    
    if not metrics.reconstruction_success:
        return 0.0, ["3D reconstruction failed"]
    
    score = 50.0  # Base score for successful reconstruction
    
    # Registration ratio
    if metrics.total_images > 0:
        reg_ratio = metrics.registered_images / metrics.total_images
        score += reg_ratio * 20
        
        if reg_ratio >= 0.9:
            reasons.append("Excellent image registration")
        elif reg_ratio < 0.6:
            reasons.append("Some images failed to register")
    
    # Reprojection error
    if metrics.mean_reprojection_error > 0:
        if metrics.mean_reprojection_error < 0.5:
            score += 20
            reasons.append("Low reprojection error")
        elif metrics.mean_reprojection_error < MAX_REPROJECTION_ERROR:
            score += 10
        else:
            score -= 10
            reasons.append("High reprojection error detected")
    
    # Point cloud density
    if metrics.dense_points > 500000:
        score += 10
        reasons.append("Dense point cloud generated")
    elif metrics.dense_points < 100000:
        reasons.append("Sparse point cloud may affect accuracy")
    
    return min(100, score), reasons


def calculate_facet_consistency_score(
    facets: List[RoofFacet]
) -> Tuple[float, List[str]]:
    """
    Calculate facet consistency contribution to confidence.
    
    Checks for anomalies in extracted facets.
    
    Args:
        facets: List of extracted roof facets
        
    Returns:
        Tuple of (score 0-100, list of reasons)
    """
    reasons = []
    
    if not facets:
        return 0.0, ["No roof facets extracted"]
    
    score = 70.0  # Base score for having facets
    
    # Check pitch consistency
    pitches = [f.pitch_12 for f in facets]
    pitch_variance = np.var(pitches) if len(pitches) > 1 else 0
    
    if pitch_variance < 2:
        score += 15
        reasons.append("Consistent pitch across facets")
    elif pitch_variance > MAX_PITCH_VARIANCE:
        score -= 10
        reasons.append(f"High pitch variance ({pitch_variance:.1f})")
    
    # Check for anomalous pitches
    anomalous = [f for f in facets if f.pitch_12 < 1 or f.pitch_12 > 18]
    if anomalous:
        score -= len(anomalous) * 5
        reasons.append(f"{len(anomalous)} facets with unusual pitch")
    
    # Check facet count reasonableness
    if 2 <= len(facets) <= 12:
        score += 10
        reasons.append(f"{len(facets)} facets detected (typical)")
    elif len(facets) > 15:
        score -= 5
        reasons.append(f"Unusually high facet count ({len(facets)})")
    elif len(facets) == 1:
        reasons.append("Single facet detected (simple roof)")
    
    # Check point density per facet
    avg_points = np.mean([f.point_count for f in facets])
    if avg_points > 5000:
        score += 5
    elif avg_points < 1000:
        score -= 5
        reasons.append("Low point density in facets")
    
    return min(100, max(0, score)), reasons


def calculate_google_validation_score(
    validation: Optional[ValidationResult]
) -> Tuple[float, List[str]]:
    """
    Calculate Google validation contribution to confidence.
    
    Args:
        validation: Validation result against Google Solar API
        
    Returns:
        Tuple of (score 0-100, list of reasons)
    """
    reasons = []
    
    if validation is None:
        return 50.0, ["Google validation not performed"]
    
    if not validation.validated and validation.google_area_m2 is None:
        return 50.0, ["Google Solar data not available for location"]
    
    if validation.validated:
        # Score based on variance
        variance = validation.variance_percent or 0
        
        if variance < 3:
            score = 100
            reasons.append(f"Excellent match with Google ({variance:.1f}% variance)")
        elif variance < 5:
            score = 90
            reasons.append(f"Very good match with Google ({variance:.1f}% variance)")
        elif variance < 10:
            score = 75
            reasons.append(f"Good match with Google ({variance:.1f}% variance)")
        else:
            score = 60
            reasons.append(f"Acceptable match with Google ({variance:.1f}% variance)")
    else:
        variance = validation.variance_percent or 0
        score = max(20, 60 - variance)
        reasons.append(f"Deviation from Google data ({variance:.1f}% variance)")
    
    return score, reasons


def calculate_confidence(
    photo_validations: List[PhotoValidation],
    reconstruction_metrics: ProcessingMetrics,
    facets: List[RoofFacet],
    google_validation: Optional[ValidationResult] = None
) -> Tuple[float, ConfidenceLevel, List[str], ConfidenceFactors]:
    """
    Calculate overall confidence score with detailed breakdown.
    
    Args:
        photo_validations: Photo quality validations
        reconstruction_metrics: COLMAP processing metrics
        facets: Extracted roof facets
        google_validation: Optional Google API validation
        
    Returns:
        Tuple of (score, level, reasons, factors)
    """
    all_reasons = []
    
    # Calculate individual scores
    photo_score, photo_reasons = calculate_photo_quality_score(photo_validations)
    all_reasons.extend(photo_reasons)
    
    angle_score, angle_reasons = calculate_angle_coverage_score(
        photo_validations,
        reconstruction_metrics.registered_images
    )
    all_reasons.extend(angle_reasons)
    
    recon_score, recon_reasons = calculate_reconstruction_score(reconstruction_metrics)
    all_reasons.extend(recon_reasons)
    
    facet_score, facet_reasons = calculate_facet_consistency_score(facets)
    all_reasons.extend(facet_reasons)
    
    google_score, google_reasons = calculate_google_validation_score(google_validation)
    all_reasons.extend(google_reasons)
    
    # Store factors
    factors = ConfidenceFactors(
        photo_quality_score=photo_score,
        angle_coverage_score=angle_score,
        reconstruction_stability=recon_score,
        facet_consistency=facet_score,
        google_validation_score=google_score
    )
    
    # Calculate weighted overall score
    overall_score = (
        photo_score * WEIGHTS["photo_quality"] +
        angle_score * WEIGHTS["angle_coverage"] +
        recon_score * WEIGHTS["reconstruction"] +
        facet_score * WEIGHTS["facet_consistency"] +
        google_score * WEIGHTS["google_validation"]
    )
    
    # Apply Google validation adjustment
    if google_validation:
        overall_score += google_validation.confidence_adjustment
    
    # Clamp score
    overall_score = max(0, min(100, overall_score))
    
    # Determine level
    if overall_score >= 85:
        level = ConfidenceLevel.HIGH
    elif overall_score >= 65:
        level = ConfidenceLevel.MEDIUM
    else:
        level = ConfidenceLevel.LOW
    
    # Filter to most important reasons (max 5)
    priority_reasons = all_reasons[:5]
    
    logger.info(
        "confidence_calculated",
        score=overall_score,
        level=level.value,
        factors={
            "photo": photo_score,
            "angle": angle_score,
            "recon": recon_score,
            "facet": facet_score,
            "google": google_score
        }
    )
    
    return overall_score, level, priority_reasons, factors


def get_confidence_explanation(
    score: float,
    level: ConfidenceLevel,
    factors: ConfidenceFactors
) -> str:
    """
    Generate human-readable confidence explanation.
    
    Args:
        score: Overall confidence score
        level: Confidence level
        factors: Individual factor scores
        
    Returns:
        Explanation string
    """
    explanations = []
    
    if level == ConfidenceLevel.HIGH:
        explanations.append(
            f"This measurement has HIGH confidence ({score:.0f}/100). "
            "The photos were high quality, coverage was complete, and "
            "the 3D reconstruction was stable."
        )
    elif level == ConfidenceLevel.MEDIUM:
        explanations.append(
            f"This measurement has MEDIUM confidence ({score:.0f}/100). "
            "Results are reliable but some factors could be improved."
        )
    else:
        explanations.append(
            f"This measurement has LOW confidence ({score:.0f}/100). "
            "Consider recapturing photos for better accuracy."
        )
    
    # Add specific factor insights
    if factors.photo_quality_score < 60:
        explanations.append("Photo quality issues detected.")
    
    if factors.angle_coverage_score < 70:
        explanations.append("Some capture angles may be missing.")
    
    if factors.reconstruction_stability < 60:
        explanations.append("3D reconstruction had stability issues.")
    
    if factors.google_validation_score >= 80:
        explanations.append("Validated against Google satellite data.")
    
    return " ".join(explanations)
