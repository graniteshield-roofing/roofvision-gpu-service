"""
Google Solar API Integration Service

Provides ground truth validation for roof measurements using
Google's high-accuracy satellite + LiDAR data.

Coverage: 472M+ buildings in 40+ countries
"""

import os
import httpx
from typing import Optional, Dict, List
import structlog

from models.measurement import GoogleSolarData, ValidationResult

logger = structlog.get_logger()

# Google Solar API configuration
GOOGLE_SOLAR_API_URL = "https://solar.googleapis.com/v1/buildingInsights:findClosest"
GOOGLE_API_KEY = os.environ.get("GOOGLE_SOLAR_API_KEY", "")

# Validation thresholds
VARIANCE_THRESHOLD_PERCENT = 10.0  # Flag if >10% variance
HIGH_CONFIDENCE_THRESHOLD = 5.0  # High confidence if <5% variance


async def fetch_google_solar_data(
    latitude: float,
    longitude: float,
    api_key: Optional[str] = None
) -> GoogleSolarData:
    """
    Fetch roof data from Google Solar API.
    
    Args:
        latitude: Building latitude
        longitude: Building longitude
        api_key: Google API key (uses env var if not provided)
        
    Returns:
        GoogleSolarData with roof measurements
    """
    key = api_key or GOOGLE_API_KEY
    
    if not key:
        logger.warning("google_api_key_not_set")
        return GoogleSolarData(available=False)
    
    params = {
        "location.latitude": f"{latitude:.6f}",
        "location.longitude": f"{longitude:.6f}",
        "requiredQuality": "HIGH",
        "key": key
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(GOOGLE_SOLAR_API_URL, params=params)
            
            if response.status_code == 404:
                logger.info("google_solar_no_data", lat=latitude, lng=longitude)
                return GoogleSolarData(available=False)
            
            if response.status_code != 200:
                logger.error(
                    "google_solar_api_error",
                    status=response.status_code,
                    body=response.text[:200]
                )
                return GoogleSolarData(available=False)
            
            data = response.json()
            
            # Extract solar potential data
            solar_potential = data.get("solarPotential", {})
            whole_roof = solar_potential.get("wholeRoofStats", {})
            
            # Extract roof segments
            segments = []
            for seg in solar_potential.get("roofSegmentStats", []):
                segments.append({
                    "pitch_degrees": seg.get("pitchDegrees"),
                    "azimuth_degrees": seg.get("azimuthDegrees"),
                    "area_m2": seg.get("stats", {}).get("areaMeters2"),
                    "ground_area_m2": seg.get("stats", {}).get("groundAreaMeters2"),
                    "center": seg.get("center")
                })
            
            result = GoogleSolarData(
                whole_roof_area_m2=whole_roof.get("areaMeters2"),
                ground_area_m2=whole_roof.get("groundAreaMeters2"),
                max_sunshine_hours=solar_potential.get("maxSunshineHoursPerYear"),
                imagery_quality=data.get("imageryQuality"),
                segments=segments,
                available=True
            )
            
            logger.info(
                "google_solar_data_fetched",
                area_m2=result.whole_roof_area_m2,
                segments=len(segments),
                quality=result.imagery_quality
            )
            
            return result
            
    except httpx.TimeoutException:
        logger.error("google_solar_timeout")
        return GoogleSolarData(available=False)
    except Exception as e:
        logger.error("google_solar_error", error=str(e))
        return GoogleSolarData(available=False)


def validate_measurements(
    our_area_m2: float,
    google_data: GoogleSolarData,
    tolerance_percent: float = VARIANCE_THRESHOLD_PERCENT
) -> ValidationResult:
    """
    Validate our measurements against Google Solar API data.
    
    Args:
        our_area_m2: Our calculated roof area in square meters
        google_data: Data from Google Solar API
        tolerance_percent: Acceptable variance threshold
        
    Returns:
        ValidationResult with comparison details
    """
    if not google_data.available or google_data.whole_roof_area_m2 is None:
        return ValidationResult(
            validated=False,
            our_area_m2=our_area_m2,
            google_area_m2=None,
            variance_percent=None,
            confidence_adjustment=0,
            reason="Google Solar data not available for this location"
        )
    
    google_area = google_data.whole_roof_area_m2
    
    if google_area <= 0:
        return ValidationResult(
            validated=False,
            our_area_m2=our_area_m2,
            google_area_m2=google_area,
            variance_percent=None,
            confidence_adjustment=0,
            reason="Invalid Google Solar area data"
        )
    
    # Calculate variance
    variance_percent = abs(our_area_m2 - google_area) / google_area * 100
    
    # Determine validation status and confidence adjustment
    if variance_percent <= HIGH_CONFIDENCE_THRESHOLD:
        validated = True
        confidence_adjustment = 10  # Boost confidence
        reason = f"Excellent match with Google data ({variance_percent:.1f}% variance)"
    elif variance_percent <= tolerance_percent:
        validated = True
        confidence_adjustment = 5
        reason = f"Good match with Google data ({variance_percent:.1f}% variance)"
    else:
        validated = False
        # Penalize confidence based on variance
        confidence_adjustment = -min(20, int(variance_percent - tolerance_percent))
        reason = f"Significant deviation from Google data ({variance_percent:.1f}% variance)"
    
    logger.info(
        "measurement_validation",
        our_area=our_area_m2,
        google_area=google_area,
        variance=variance_percent,
        validated=validated
    )
    
    return ValidationResult(
        validated=validated,
        our_area_m2=round(our_area_m2, 2),
        google_area_m2=round(google_area, 2),
        variance_percent=round(variance_percent, 2),
        confidence_adjustment=confidence_adjustment,
        reason=reason
    )


def compare_facets(
    our_facets: List[Dict],
    google_segments: List[Dict]
) -> Dict:
    """
    Compare our facets with Google's roof segments.
    
    Args:
        our_facets: Our extracted facets
        google_segments: Google's roof segments
        
    Returns:
        Comparison results with per-facet analysis
    """
    if not google_segments:
        return {
            "comparable": False,
            "reason": "No Google segments available",
            "matches": []
        }
    
    matches = []
    
    # Simple matching by pitch similarity
    for our_facet in our_facets:
        our_pitch = our_facet.get("pitch_degrees", 0)
        our_azimuth = our_facet.get("azimuth_degrees", 0)
        our_area = our_facet.get("area_m2", 0)
        
        best_match = None
        best_score = float('inf')
        
        for google_seg in google_segments:
            google_pitch = google_seg.get("pitch_degrees", 0) or 0
            google_azimuth = google_seg.get("azimuth_degrees", 0) or 0
            google_area = google_seg.get("area_m2", 0) or 0
            
            # Score based on pitch and azimuth similarity
            pitch_diff = abs(our_pitch - google_pitch)
            azimuth_diff = min(
                abs(our_azimuth - google_azimuth),
                360 - abs(our_azimuth - google_azimuth)
            )
            
            score = pitch_diff + azimuth_diff * 0.1
            
            if score < best_score:
                best_score = score
                best_match = {
                    "google_pitch": google_pitch,
                    "google_azimuth": google_azimuth,
                    "google_area_m2": google_area,
                    "pitch_diff": pitch_diff,
                    "azimuth_diff": azimuth_diff
                }
        
        if best_match and best_score < 15:  # Reasonable match threshold
            area_variance = 0
            if best_match["google_area_m2"] > 0:
                area_variance = abs(our_area - best_match["google_area_m2"]) / best_match["google_area_m2"] * 100
            
            matches.append({
                "our_facet": our_facet,
                "google_match": best_match,
                "area_variance_percent": round(area_variance, 1),
                "match_quality": "good" if best_score < 5 else "fair"
            })
    
    return {
        "comparable": True,
        "our_facet_count": len(our_facets),
        "google_segment_count": len(google_segments),
        "matched_count": len(matches),
        "matches": matches
    }


async def get_calibration_data(
    latitude: float,
    longitude: float,
    api_key: Optional[str] = None
) -> Optional[Dict]:
    """
    Get calibration data for a location.
    
    Used during system calibration to tune RANSAC parameters.
    
    Args:
        latitude: Building latitude
        longitude: Building longitude
        api_key: Google API key
        
    Returns:
        Calibration data or None
    """
    google_data = await fetch_google_solar_data(latitude, longitude, api_key)
    
    if not google_data.available:
        return None
    
    return {
        "location": {"lat": latitude, "lng": longitude},
        "total_area_m2": google_data.whole_roof_area_m2,
        "ground_area_m2": google_data.ground_area_m2,
        "segment_count": len(google_data.segments),
        "segments": google_data.segments,
        "imagery_quality": google_data.imagery_quality
    }
