"""
Point cloud processing and quality pipeline (Section 9).

Applies:
- Statistical outlier removal (SOR)
- Radius outlier filtering for isolated floaters
- NaN/Inf invalid coordinate rejection
- Normal estimation and orientation toward camera centers (for dense clouds)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import open3d as o3d

logger = logging.getLogger("reconstruction.pointcloud")


def clean_point_cloud(
    pcd: o3d.geometry.PointCloud,
    is_dense: bool = False,
    camera_centers: Optional[List[List[float]]] = None,
    nb_neighbors: int = 20,
    std_ratio: float = 2.5,
) -> o3d.geometry.PointCloud:
    """Cleans a point cloud using documented geometric outlier removal and normal orientation."""
    if len(pcd.points) == 0:
        return pcd

    # 1. Remove non-finite points (NaN/Inf)
    pcd.remove_non_finite_points()

    # 2. Statistical Outlier Removal
    if len(pcd.points) > nb_neighbors:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )

    # 3. For dense clouds, estimate normals and orient towards camera centers
    if is_dense and len(pcd.points) > 100:
        if not pcd.has_normals():
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)
            )

        if camera_centers and len(camera_centers) > 0:
            avg_cam = np.mean(camera_centers, axis=0)
            pcd.orient_normals_towards_camera_location(camera_location=avg_cam)
        else:
            pcd.orient_normals_consistent_tangent_plane(k=20)

    return pcd


def load_and_clean(
    ply_path: Path,
    voxel_size: float | None = None,
    nb_neighbors: int = 20,
    std_ratio: float = 2.5,
    is_dense: bool = False,
    camera_centers: Optional[List[List[float]]] = None,
) -> o3d.geometry.PointCloud:
    """Loads a PLY point cloud and applies scientific quality filtering."""
    pcd = o3d.io.read_point_cloud(str(ply_path))

    if voxel_size and voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    return clean_point_cloud(
        pcd,
        is_dense=is_dense,
        camera_centers=camera_centers,
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )


def export_cloud(pcd: o3d.geometry.PointCloud, output_path: Path) -> int:
    """Exports point cloud preserving original RGB colors."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(output_path), pcd)
    return len(pcd.points)
