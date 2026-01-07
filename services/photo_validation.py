"""
Photo Validation Service

Validates photo quality before processing:
- Blur detection using Laplacian variance
- Resolution verification
- Angle classification
- Obstruction detection
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import structlog

from models.measurement import PhotoValidation, PhotoQuality

logger = structlog.get_logger()

# Minimum requirements
MIN_RESOLUTION_MP = 12.0
MIN_BLUR_SCORE = 100.0
REQUIRED_ANGLES = [
    "front", "front_right", "right", "back_right",
    "back", "back_left", "left", "front_left"
]


def detect_blur(image: np.ndarray) -> Tuple[float, bool]:
    """
    Detect blur using Laplacian variance.
    
    Higher variance = sharper image.
    Threshold of 100 works well for 12MP+ images.
    
    Args:
        image: BGR image array
        
    Returns:
        Tuple of (blur_score, is_blurry)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return float(laplacian_var), laplacian_var < MIN_BLUR_SCORE


def check_resolution(image: np.ndarray) -> Tuple[float, bool]:
    """
    Check if image meets minimum resolution requirement.
    
    Args:
        image: BGR image array
        
    Returns:
        Tuple of (megapixels, meets_requirement)
    """
    height, width = image.shape[:2]
    megapixels = (height * width) / 1_000_000
    
    return megapixels, megapixels >= MIN_RESOLUTION_MP


def analyze_exposure(image: np.ndarray) -> Tuple[str, List[str]]:
    """
    Analyze image exposure and lighting.
    
    Args:
        image: BGR image array
        
    Returns:
        Tuple of (quality_assessment, issues_list)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    
    issues = []
    
    # Check for underexposure (too dark)
    dark_pixels = hist[:50].sum()
    if dark_pixels > 0.5:
        issues.append("Image appears underexposed (too dark)")
    
    # Check for overexposure (too bright)
    bright_pixels = hist[200:].sum()
    if bright_pixels > 0.5:
        issues.append("Image appears overexposed (too bright)")
    
    # Check for low contrast
    std_dev = np.std(gray)
    if std_dev < 40:
        issues.append("Low contrast detected")
    
    if len(issues) == 0:
        return "good", []
    elif len(issues) == 1:
        return "warning", issues
    else:
        return "poor", issues


def detect_roof_coverage(image: np.ndarray) -> Tuple[float, bool]:
    """
    Estimate how much of the image contains roof/building.
    Uses edge detection and color analysis.
    
    Args:
        image: BGR image array
        
    Returns:
        Tuple of (coverage_percent, has_sufficient_coverage)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Focus on upper 2/3 of image (where roof should be)
    height = edges.shape[0]
    roof_region = edges[:int(height * 0.67), :]
    
    # Calculate edge density
    edge_density = np.sum(roof_region > 0) / roof_region.size
    
    # Estimate coverage (edge density correlates with structure presence)
    coverage = min(100, edge_density * 500)  # Scale to percentage
    
    return coverage, coverage > 20


def classify_angle(image: np.ndarray, filename: str) -> Optional[str]:
    """
    Attempt to classify which angle the photo represents.
    
    This is a simplified version - production would use ML model.
    For now, extracts from filename if available.
    
    Args:
        image: BGR image array
        filename: Original filename
        
    Returns:
        Detected angle or None
    """
    filename_lower = filename.lower()
    
    for angle in REQUIRED_ANGLES:
        if angle.replace("_", "") in filename_lower or angle in filename_lower:
            return angle
    
    # Could add ML-based classification here
    return None


def validate_photo(
    image_path: str,
    photo_id: str
) -> PhotoValidation:
    """
    Perform comprehensive photo validation.
    
    Args:
        image_path: Path to image file
        photo_id: Unique identifier for the photo
        
    Returns:
        PhotoValidation result
    """
    path = Path(image_path)
    filename = path.name
    issues = []
    
    try:
        # Load image
        image = cv2.imread(str(path))
        if image is None:
            return PhotoValidation(
                photo_id=photo_id,
                filename=filename,
                quality=PhotoQuality.POOR,
                blur_score=0,
                resolution_mp=0,
                issues=["Failed to load image"],
                is_usable=False
            )
        
        # Check resolution
        resolution_mp, resolution_ok = check_resolution(image)
        if not resolution_ok:
            issues.append(f"Resolution too low: {resolution_mp:.1f}MP (need {MIN_RESOLUTION_MP}MP)")
        
        # Check blur
        blur_score, is_blurry = detect_blur(image)
        if is_blurry:
            issues.append(f"Image is blurry (score: {blur_score:.0f}, need >{MIN_BLUR_SCORE})")
        
        # Check exposure
        exposure_quality, exposure_issues = analyze_exposure(image)
        issues.extend(exposure_issues)
        
        # Check roof coverage
        coverage, has_coverage = detect_roof_coverage(image)
        if not has_coverage:
            issues.append(f"Insufficient building/roof coverage ({coverage:.0f}%)")
        
        # Classify angle
        angle = classify_angle(image, filename)
        
        # Determine overall quality
        if len(issues) == 0:
            quality = PhotoQuality.GOOD
            is_usable = True
        elif len(issues) <= 2 and resolution_ok and not is_blurry:
            quality = PhotoQuality.WARNING
            is_usable = True
        else:
            quality = PhotoQuality.POOR
            is_usable = resolution_ok and not is_blurry
        
        return PhotoValidation(
            photo_id=photo_id,
            filename=filename,
            quality=quality,
            blur_score=blur_score,
            resolution_mp=resolution_mp,
            angle_detected=angle,
            issues=issues,
            is_usable=is_usable
        )
        
    except Exception as e:
        logger.error("photo_validation_error", photo_id=photo_id, error=str(e))
        return PhotoValidation(
            photo_id=photo_id,
            filename=filename,
            quality=PhotoQuality.POOR,
            blur_score=0,
            resolution_mp=0,
            issues=[f"Validation error: {str(e)}"],
            is_usable=False
        )


def validate_photo_set(
    image_paths: List[str]
) -> Tuple[List[PhotoValidation], List[str], bool]:
    """
    Validate a complete set of photos for a job.
    
    Args:
        image_paths: List of paths to image files
        
    Returns:
        Tuple of (validations, missing_angles, can_proceed)
    """
    validations = []
    detected_angles = set()
    usable_count = 0
    
    for i, path in enumerate(image_paths):
        photo_id = f"photo_{i+1}"
        validation = validate_photo(path, photo_id)
        validations.append(validation)
        
        if validation.is_usable:
            usable_count += 1
        
        if validation.angle_detected:
            detected_angles.add(validation.angle_detected)
    
    # Check for missing angles
    missing_angles = [a for a in REQUIRED_ANGLES if a not in detected_angles]
    
    # Determine if we can proceed
    # Need at least 6 usable photos for reasonable reconstruction
    can_proceed = usable_count >= 6
    
    logger.info(
        "photo_set_validation_complete",
        total=len(image_paths),
        usable=usable_count,
        missing_angles=missing_angles,
        can_proceed=can_proceed
    )
    
    return validations, missing_angles, can_proceed


def get_photo_quality_score(validations: List[PhotoValidation]) -> float:
    """
    Calculate overall photo quality score (0-100).
    
    Args:
        validations: List of photo validations
        
    Returns:
        Quality score from 0-100
    """
    if not validations:
        return 0.0
    
    scores = []
    for v in validations:
        if v.quality == PhotoQuality.GOOD:
            scores.append(100)
        elif v.quality == PhotoQuality.WARNING:
            scores.append(70)
        else:
            scores.append(30)
    
    return sum(scores) / len(scores)
