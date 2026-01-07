"""
3D Reconstruction Service using COLMAP

Handles Structure-from-Motion (SfM) and Multi-View Stereo (MVS)
to generate dense point clouds from photos.
"""

import os
import subprocess
import shutil
import time
from pathlib import Path
from typing import Optional, Tuple, Callable
import structlog

from models.measurement import ProcessingMetrics, JobStatus

logger = structlog.get_logger()

# COLMAP configuration
COLMAP_BIN = os.environ.get("COLMAP_BIN", "colmap")
GPU_INDEX = os.environ.get("COLMAP_GPU_INDEX", "0")


class ReconstructionService:
    """
    Service for 3D reconstruction using COLMAP.
    
    Pipeline:
    1. Feature extraction (SIFT)
    2. Feature matching (exhaustive for small sets)
    3. Sparse reconstruction (SfM)
    4. Image undistortion
    5. Dense reconstruction (MVS)
    6. Point cloud fusion
    """
    
    def __init__(
        self,
        workspace_dir: str,
        progress_callback: Optional[Callable[[JobStatus, int, str], None]] = None
    ):
        """
        Initialize reconstruction service.
        
        Args:
            workspace_dir: Directory for COLMAP workspace
            progress_callback: Optional callback for progress updates
        """
        self.workspace = Path(workspace_dir)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.progress_callback = progress_callback
        
        # Create subdirectories
        self.images_dir = self.workspace / "images"
        self.database_path = self.workspace / "database.db"
        self.sparse_dir = self.workspace / "sparse"
        self.dense_dir = self.workspace / "dense"
        
        for d in [self.images_dir, self.sparse_dir, self.dense_dir]:
            d.mkdir(exist_ok=True)
    
    def _update_progress(self, status: JobStatus, percent: int, stage: str):
        """Send progress update if callback is set."""
        if self.progress_callback:
            self.progress_callback(status, percent, stage)
    
    def _run_colmap(
        self,
        command: str,
        args: dict,
        timeout: int = 600
    ) -> Tuple[bool, str]:
        """
        Run a COLMAP command with arguments.
        
        Args:
            command: COLMAP command (e.g., "feature_extractor")
            args: Dictionary of arguments
            timeout: Command timeout in seconds
            
        Returns:
            Tuple of (success, output/error message)
        """
        cmd = [COLMAP_BIN, command]
        
        for key, value in args.items():
            cmd.append(f"--{key}")
            cmd.append(str(value))
        
        logger.info("running_colmap", command=command, args=args)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.workspace)
            )
            
            if result.returncode != 0:
                logger.error(
                    "colmap_error",
                    command=command,
                    stderr=result.stderr[:500]
                )
                return False, result.stderr
            
            return True, result.stdout
            
        except subprocess.TimeoutExpired:
            return False, f"Command timed out after {timeout}s"
        except Exception as e:
            return False, str(e)
    
    def copy_images(self, image_paths: list) -> int:
        """
        Copy images to workspace directory.
        
        Args:
            image_paths: List of source image paths
            
        Returns:
            Number of images copied
        """
        count = 0
        for path in image_paths:
            src = Path(path)
            if src.exists():
                dst = self.images_dir / src.name
                shutil.copy2(src, dst)
                count += 1
        
        logger.info("images_copied", count=count)
        return count
    
    def extract_features(self) -> bool:
        """
        Extract SIFT features from images.
        
        Returns:
            Success status
        """
        self._update_progress(
            JobStatus.RECONSTRUCTING, 10, "Extracting features"
        )
        
        args = {
            "database_path": str(self.database_path),
            "image_path": str(self.images_dir),
            "ImageReader.single_camera": "1",
            "ImageReader.camera_model": "OPENCV",
            "SiftExtraction.max_image_size": "3200",
            "SiftExtraction.max_num_features": "8192",
            "SiftExtraction.gpu_index": GPU_INDEX
        }
        
        success, msg = self._run_colmap("feature_extractor", args)
        return success
    
    def match_features(self) -> bool:
        """
        Match features between images (exhaustive for small sets).
        
        Returns:
            Success status
        """
        self._update_progress(
            JobStatus.RECONSTRUCTING, 25, "Matching features"
        )
        
        args = {
            "database_path": str(self.database_path),
            "SiftMatching.guided_matching": "1",
            "SiftMatching.gpu_index": GPU_INDEX
        }
        
        success, msg = self._run_colmap("exhaustive_matcher", args)
        return success
    
    def sparse_reconstruction(self) -> bool:
        """
        Run Structure-from-Motion to get camera poses.
        
        Returns:
            Success status
        """
        self._update_progress(
            JobStatus.RECONSTRUCTING, 40, "Building sparse model"
        )
        
        args = {
            "database_path": str(self.database_path),
            "image_path": str(self.images_dir),
            "output_path": str(self.sparse_dir)
        }
        
        success, msg = self._run_colmap("mapper", args, timeout=900)
        
        # Check if reconstruction was created
        if success:
            model_dir = self.sparse_dir / "0"
            if not model_dir.exists():
                return False
        
        return success
    
    def undistort_images(self) -> bool:
        """
        Undistort images for dense reconstruction.
        
        Returns:
            Success status
        """
        self._update_progress(
            JobStatus.RECONSTRUCTING, 55, "Preparing for dense reconstruction"
        )
        
        args = {
            "image_path": str(self.images_dir),
            "input_path": str(self.sparse_dir / "0"),
            "output_path": str(self.dense_dir)
        }
        
        success, msg = self._run_colmap("image_undistorter", args)
        return success
    
    def dense_reconstruction(self) -> bool:
        """
        Run Multi-View Stereo for dense point cloud.
        
        Returns:
            Success status
        """
        self._update_progress(
            JobStatus.RECONSTRUCTING, 70, "Dense reconstruction"
        )
        
        # Patch match stereo
        args = {
            "workspace_path": str(self.dense_dir),
            "PatchMatchStereo.geom_consistency": "true",
            "PatchMatchStereo.gpu_index": GPU_INDEX
        }
        
        success, msg = self._run_colmap("patch_match_stereo", args, timeout=1200)
        if not success:
            return False
        
        self._update_progress(
            JobStatus.RECONSTRUCTING, 85, "Fusing point cloud"
        )
        
        # Stereo fusion
        args = {
            "workspace_path": str(self.dense_dir),
            "output_path": str(self.dense_dir / "fused.ply"),
            "StereoFusion.min_num_pixels": "3"
        }
        
        success, msg = self._run_colmap("stereo_fusion", args)
        return success
    
    def get_metrics(self) -> ProcessingMetrics:
        """
        Extract reconstruction metrics from COLMAP output.
        
        Returns:
            ProcessingMetrics with reconstruction statistics
        """
        metrics = ProcessingMetrics()
        
        # Count input images
        metrics.total_images = len(list(self.images_dir.glob("*")))
        
        # Parse sparse reconstruction stats
        cameras_file = self.sparse_dir / "0" / "cameras.bin"
        images_file = self.sparse_dir / "0" / "images.bin"
        points_file = self.sparse_dir / "0" / "points3D.bin"
        
        if images_file.exists():
            # Estimate registered images from file size
            # (proper parsing would use pycolmap)
            metrics.registered_images = min(
                metrics.total_images,
                max(1, images_file.stat().st_size // 1000)
            )
        
        if points_file.exists():
            metrics.sparse_points = max(1, points_file.stat().st_size // 50)
        
        # Check dense point cloud
        dense_ply = self.dense_dir / "fused.ply"
        if dense_ply.exists():
            # Estimate point count from file size
            metrics.dense_points = max(1, dense_ply.stat().st_size // 30)
            metrics.reconstruction_success = True
        
        return metrics
    
    def run_full_pipeline(self, image_paths: list) -> Tuple[bool, ProcessingMetrics, str]:
        """
        Run complete reconstruction pipeline.
        
        Args:
            image_paths: List of input image paths
            
        Returns:
            Tuple of (success, metrics, output_path or error)
        """
        start_time = time.time()
        
        try:
            # Copy images
            copied = self.copy_images(image_paths)
            if copied < 4:
                return False, ProcessingMetrics(), "Need at least 4 images"
            
            # Feature extraction
            if not self.extract_features():
                return False, ProcessingMetrics(), "Feature extraction failed"
            
            # Feature matching
            if not self.match_features():
                return False, ProcessingMetrics(), "Feature matching failed"
            
            # Sparse reconstruction
            if not self.sparse_reconstruction():
                return False, ProcessingMetrics(), "Sparse reconstruction failed"
            
            # Undistort
            if not self.undistort_images():
                return False, ProcessingMetrics(), "Image undistortion failed"
            
            # Dense reconstruction
            if not self.dense_reconstruction():
                return False, ProcessingMetrics(), "Dense reconstruction failed"
            
            # Get metrics
            metrics = self.get_metrics()
            metrics.processing_time_seconds = time.time() - start_time
            
            output_path = str(self.dense_dir / "fused.ply")
            
            self._update_progress(
                JobStatus.RECONSTRUCTING, 100, "Reconstruction complete"
            )
            
            logger.info(
                "reconstruction_complete",
                metrics=metrics.__dict__,
                output=output_path
            )
            
            return True, metrics, output_path
            
        except Exception as e:
            logger.error("reconstruction_error", error=str(e))
            return False, ProcessingMetrics(), str(e)
    
    def cleanup(self):
        """Remove workspace directory."""
        if self.workspace.exists():
            shutil.rmtree(self.workspace)


def run_reconstruction(
    job_id: str,
    image_paths: list,
    workspace_base: str = "/tmp/roofvision",
    progress_callback: Optional[Callable] = None
) -> Tuple[bool, ProcessingMetrics, str]:
    """
    Convenience function to run reconstruction for a job.
    
    Args:
        job_id: Unique job identifier
        image_paths: List of input image paths
        workspace_base: Base directory for workspaces
        progress_callback: Optional progress callback
        
    Returns:
        Tuple of (success, metrics, output_path or error)
    """
    workspace = os.path.join(workspace_base, job_id)
    service = ReconstructionService(workspace, progress_callback)
    
    try:
        return service.run_full_pipeline(image_paths)
    finally:
        # Optionally cleanup (keep for debugging)
        pass
