"""
Roof Plane Extraction Service using Open3D

Extracts individual roof facets from dense point cloud using:
- RANSAC for plane detection
- DBSCAN for cluster refinement
- Convex hull for area calculation
"""

import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import structlog

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    o3d = None

from sklearn.cluster import DBSCAN

from models.measurement import RoofFacet, JobStatus

logger = structlog.get_logger()

# RANSAC parameters optimized for photogrammetric point clouds
RANSAC_DISTANCE_THRESHOLD = 0.02  # 2cm tolerance
RANSAC_NUM_ITERATIONS = 2000
RANSAC_MIN_POINTS = 3
MIN_FACET_POINTS = 500  # Minimum points per facet
MAX_FACETS = 20  # Maximum facets to extract
ROOF_HEIGHT_PERCENTILE = 60  # Filter to top 40% by height

# DBSCAN parameters for cluster refinement
DBSCAN_EPS = 0.1  # 10cm neighborhood
DBSCAN_MIN_SAMPLES = 10


def pitch_from_normal(normal: Tuple[float, float, float]) -> Tuple[float, float]:
    """
    Calculate roof pitch from plane normal vector.
    
    Args:
        normal: (nx, ny, nz) normal vector
        
    Returns:
        Tuple of (pitch_degrees, pitch_12_format)
    """
    nx, ny, nz = normal
    
    # Pitch is angle from horizontal
    horizontal_component = np.sqrt(nx**2 + ny**2)
    vertical_component = abs(nz)
    
    pitch_rad = np.arctan2(horizontal_component, vertical_component)
    pitch_deg = np.degrees(pitch_rad)
    pitch_12 = np.tan(pitch_rad) * 12
    
    return pitch_deg, pitch_12


def azimuth_from_normal(normal: Tuple[float, float, float]) -> float:
    """
    Calculate azimuth (compass direction) from plane normal.
    
    Args:
        normal: (nx, ny, nz) normal vector
        
    Returns:
        Azimuth in degrees (0-360, 0=North, 90=East)
    """
    nx, ny, nz = normal
    
    # Azimuth from horizontal components
    azimuth_rad = np.arctan2(ny, nx)
    azimuth_deg = np.degrees(azimuth_rad)
    
    # Convert to compass bearing (0=North)
    azimuth_deg = (90 - azimuth_deg + 360) % 360
    
    return azimuth_deg


def calculate_plane_area(
    points: np.ndarray,
    use_convex_hull: bool = True
) -> float:
    """
    Calculate area of a roof plane from its points.
    
    Args:
        points: Nx3 array of points
        use_convex_hull: If True, use convex hull; else use bounding box
        
    Returns:
        Area in square meters
    """
    if not OPEN3D_AVAILABLE:
        # Fallback: bounding box estimation
        min_pt = points.min(axis=0)
        max_pt = points.max(axis=0)
        extent = max_pt - min_pt
        return extent[0] * extent[1]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if use_convex_hull and len(points) >= 4:
        try:
            hull, _ = pcd.compute_convex_hull()
            return hull.get_surface_area()
        except Exception:
            pass
    
    # Fallback to bounding box
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    return extent[0] * extent[1]


def filter_roof_region(
    pcd: "o3d.geometry.PointCloud",
    height_percentile: float = ROOF_HEIGHT_PERCENTILE
) -> "o3d.geometry.PointCloud":
    """
    Filter point cloud to roof region (upper portion by height).
    
    Args:
        pcd: Input point cloud
        height_percentile: Keep points above this percentile
        
    Returns:
        Filtered point cloud
    """
    points = np.asarray(pcd.points)
    z_threshold = np.percentile(points[:, 2], height_percentile)
    
    roof_mask = points[:, 2] > z_threshold
    roof_indices = np.where(roof_mask)[0]
    
    return pcd.select_by_index(roof_indices)


def refine_with_dbscan(
    points: np.ndarray,
    eps: float = DBSCAN_EPS,
    min_samples: int = DBSCAN_MIN_SAMPLES
) -> List[np.ndarray]:
    """
    Refine plane points using DBSCAN clustering.
    
    Separates disconnected regions within a plane.
    
    Args:
        points: Nx3 array of plane points
        eps: DBSCAN neighborhood radius
        min_samples: Minimum points per cluster
        
    Returns:
        List of point arrays, one per cluster
    """
    if len(points) < min_samples:
        return [points]
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    
    clusters = []
    unique_labels = set(labels) - {-1}  # Exclude noise
    
    for label in unique_labels:
        mask = labels == label
        cluster_points = points[mask]
        if len(cluster_points) >= MIN_FACET_POINTS:
            clusters.append(cluster_points)
    
    return clusters if clusters else [points]


def extract_roof_planes(
    point_cloud_path: str,
    distance_threshold: float = RANSAC_DISTANCE_THRESHOLD,
    min_points: int = MIN_FACET_POINTS,
    max_planes: int = MAX_FACETS,
    progress_callback=None
) -> List[RoofFacet]:
    """
    Extract roof planes from dense point cloud using iterative RANSAC.
    
    Args:
        point_cloud_path: Path to PLY file
        distance_threshold: RANSAC inlier distance (meters)
        min_points: Minimum points per plane
        max_planes: Maximum planes to extract
        progress_callback: Optional progress callback
        
    Returns:
        List of RoofFacet objects with measurements
    """
    if not OPEN3D_AVAILABLE:
        logger.error("open3d_not_available")
        raise ImportError("Open3D is required for plane extraction")
    
    logger.info("loading_point_cloud", path=point_cloud_path)
    
    # Load point cloud
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    total_points = len(pcd.points)
    
    if total_points < 1000:
        logger.warning("insufficient_points", count=total_points)
        return []
    
    # Filter to roof region
    roof_pcd = filter_roof_region(pcd)
    logger.info("roof_region_filtered", original=total_points, filtered=len(roof_pcd.points))
    
    facets = []
    remaining = roof_pcd
    facet_id = 0
    
    for iteration in range(max_planes):
        if len(remaining.points) < min_points:
            break
        
        if progress_callback:
            progress = int((iteration / max_planes) * 100)
            progress_callback(JobStatus.EXTRACTING, progress, f"Extracting facet {iteration + 1}")
        
        # RANSAC plane fitting
        try:
            plane_model, inliers = remaining.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=RANSAC_MIN_POINTS,
                num_iterations=RANSAC_NUM_ITERATIONS
            )
        except Exception as e:
            logger.warning("ransac_failed", iteration=iteration, error=str(e))
            break
        
        if len(inliers) < min_points:
            logger.info("stopping_extraction", reason="insufficient_inliers", count=len(inliers))
            break
        
        # Extract plane points
        plane_cloud = remaining.select_by_index(inliers)
        remaining = remaining.select_by_index(inliers, invert=True)
        
        plane_points = np.asarray(plane_cloud.points)
        
        # Refine with DBSCAN to separate disconnected regions
        clusters = refine_with_dbscan(plane_points)
        
        for cluster_points in clusters:
            if len(cluster_points) < min_points:
                continue
            
            facet_id += 1
            a, b, c, d = plane_model
            
            # Normalize normal vector
            norm = np.sqrt(a**2 + b**2 + c**2)
            normal = (a/norm, b/norm, c/norm)
            
            # Calculate pitch
            pitch_deg, pitch_12 = pitch_from_normal(normal)
            
            # Skip near-vertical surfaces (walls, not roof)
            if pitch_deg > 75:
                continue
            
            # Calculate azimuth
            azimuth = azimuth_from_normal(normal)
            
            # Calculate area
            area_m2 = calculate_plane_area(cluster_points)
            area_sqft = area_m2 * 10.7639
            
            # Center point
            center = tuple(cluster_points.mean(axis=0))
            
            # Confidence based on point density
            confidence = min(100, len(cluster_points) / 1000 * 100)
            
            facet = RoofFacet(
                facet_id=facet_id,
                pitch_degrees=round(pitch_deg, 1),
                pitch_12=round(pitch_12, 1),
                azimuth_degrees=round(azimuth, 1),
                area_sqft=round(area_sqft, 2),
                area_m2=round(area_m2, 2),
                point_count=len(cluster_points),
                center=center,
                confidence=round(confidence, 1)
            )
            
            facets.append(facet)
            
            logger.info(
                "facet_extracted",
                facet_id=facet_id,
                pitch=pitch_12,
                area_sqft=area_sqft,
                points=len(cluster_points)
            )
    
    logger.info("extraction_complete", facet_count=len(facets))
    
    return facets


def calculate_aggregate_measurements(facets: List[RoofFacet]) -> dict:
    """
    Calculate aggregate roof measurements from facets.
    
    Args:
        facets: List of extracted roof facets
        
    Returns:
        Dictionary with aggregate measurements
    """
    if not facets:
        return {
            "total_area_sqft": 0,
            "total_area_m2": 0,
            "roofing_squares": 0,
            "dominant_pitch_12": 0,
            "dominant_pitch_degrees": 0,
            "facet_count": 0
        }
    
    total_area_sqft = sum(f.area_sqft for f in facets)
    total_area_m2 = sum(f.area_m2 for f in facets)
    
    # Dominant pitch (area-weighted average)
    if total_area_sqft > 0:
        weighted_pitch = sum(f.pitch_12 * f.area_sqft for f in facets)
        dominant_pitch_12 = weighted_pitch / total_area_sqft
        
        weighted_pitch_deg = sum(f.pitch_degrees * f.area_sqft for f in facets)
        dominant_pitch_degrees = weighted_pitch_deg / total_area_sqft
    else:
        dominant_pitch_12 = 0
        dominant_pitch_degrees = 0
    
    # Roofing squares (100 sqft = 1 square)
    roofing_squares = total_area_sqft / 100
    
    return {
        "total_area_sqft": round(total_area_sqft, 2),
        "total_area_m2": round(total_area_m2, 2),
        "roofing_squares": round(roofing_squares, 2),
        "dominant_pitch_12": round(dominant_pitch_12, 1),
        "dominant_pitch_degrees": round(dominant_pitch_degrees, 1),
        "facet_count": len(facets)
    }


def estimate_ground_footprint(facets: List[RoofFacet]) -> float:
    """
    Estimate ground footprint from roof facets.
    
    Uses pitch to project roof area to ground plane.
    
    Args:
        facets: List of roof facets
        
    Returns:
        Ground footprint in square feet
    """
    footprint = 0
    
    for facet in facets:
        # Project roof area to horizontal
        pitch_rad = np.radians(facet.pitch_degrees)
        horizontal_area = facet.area_sqft * np.cos(pitch_rad)
        footprint += horizontal_area
    
    return round(footprint, 2)
