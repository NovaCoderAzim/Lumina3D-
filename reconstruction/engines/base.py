"""
Abstract base classes and value types for Lumina3D reconstruction engines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from ..schemas import CameraPose, ReconstructionConfig


@dataclass
class SfmResult:
    reconstruction: Any
    sparse_cloud_path: Path
    camera_poses: List[CameraPose]
    total_images: int
    registered_images: int
    sparse_points: int
    reprojection_error: Optional[float] = None
    mean_track_length: Optional[float] = None


@dataclass
class DenseResult:
    is_available: bool
    dense_cloud_path: Optional[Path] = None
    dense_points: Optional[int] = None
    engine_name: str = "none"
    depth_maps_count: int = 0
    surface_mesh_available: bool = False
    mesh_path: Optional[Path] = None
    texture_available: bool = False
    texture_atlas_path: Optional[Path] = None
    textured_glb_path: Optional[Path] = None
    warnings: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    error_message: Optional[str] = None


class BaseReconstructionEngine(ABC):
    """Abstract interface separating SfM, dense MVS, and surface meshing."""

    def __init__(self, config: ReconstructionConfig):
        self.config = config

    @abstractmethod
    def run_sparse_sfm(
        self,
        images_dir: Path,
        database_path: Path,
        sparse_dir: Path,
        mask_path: Optional[Path] = None,
    ) -> SfmResult:
        """Runs feature extraction, matching, and incremental bundle adjustment."""
        raise NotImplementedError

    @abstractmethod
    def run_dense_mvs(
        self,
        sparse_result: SfmResult,
        images_dir: Path,
        dense_dir: Path,
    ) -> DenseResult:
        """Runs patch-match stereo and depth fusion if genuine MVS is available."""
        raise NotImplementedError
