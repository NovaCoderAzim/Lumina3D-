"""
association.py

Geometric 2D -> 3D reconstruction association (Section 17).

Uses actual reconstruction artifacts:
- camera poses & calibrated intrinsics (cameras.json)
- triangulated 3D point cloud (cloud.ply / cloud_sparse.ply)

Progression:
2D detection box -> camera pose + intrinsics -> camera ray & frustum cone
-> geometric intersection with observed 3D points -> 3D position & confidence.

Records truthful reasons when geometric association cannot be made.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ai.association")


def _quat_to_rot_matrix(q: List[float]) -> np.ndarray:
    """Converts a [w, x, y, z] quaternion to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
    ], dtype=float)


class ReconstructionAssociator:
    """Associates a 2D bounding box with genuine 3D scene geometry."""

    def __init__(self, reconstruction_metadata: Optional[Dict[str, Any]] = None):
        self.metadata = reconstruction_metadata or {}
        self.cameras: List[Dict[str, Any]] = []
        self.points: Optional[np.ndarray] = None
        self.last_failure_reason: Optional[str] = None

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads cameras.json and cloud.ply from paths provided in metadata."""
        # 1. Load cameras
        cams_path = self.metadata.get("camera_poses") or self.metadata.get("cameras_path")
        if cams_path and Path(cams_path).exists():
            try:
                self.cameras = json.loads(Path(cams_path).read_text())
            except Exception as exc:
                logger.warning("Failed to load cameras.json: %s", exc)

        # 2. Load points
        pcd_path = (
            self.metadata.get("point_cloud")
            or self.metadata.get("cloud_path")
            or self.metadata.get("model")
        )
        if pcd_path and Path(pcd_path).exists():
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(str(pcd_path))
                raw_pts = np.asarray(pcd.points)
                if len(raw_pts) > 0:
                    # Point cloud in cloud.ply is in COLMAP coordinates; convert to glTF frame (Y-up)
                    # matching the camera poses in cameras.json: (x, -y, -z)
                    self.points = raw_pts * np.array([1.0, -1.0, -1.0])
            except Exception as exc:
                logger.warning("Failed to load point cloud for association: %s", exc)

        self.available = bool(self.cameras) and (self.points is not None and len(self.points) > 0)
        if not self.available:
            logger.info(
                "Geometric 3D association unavailable: cameras=%s, points=%s",
                len(self.cameras),
                len(self.points) if self.points is not None else 0,
            )

    def _camera_for_frame(self, frame_id: int) -> Optional[Dict[str, Any]]:
        """Finds camera matching the frame index or image filename."""
        if not self.cameras:
            return None

        patterns = [
            f"frame_{frame_id:06d}",
            f"frame_{frame_id:04d}",
            f"_{frame_id:06d}",
            f"_{frame_id}.",
            str(frame_id),
        ]

        for cam in self.cameras:
            img_name = str(cam.get("image", ""))
            for pat in patterns:
                if pat in img_name:
                    return cam

        # Fallback: if frame_id is in range of sorted cameras
        if 0 <= frame_id < len(self.cameras):
            return self.cameras[frame_id]

        return None

    def associate(
        self, bbox: List[float], frame_id: int
    ) -> Tuple[Optional[List[float]], Optional[str], Optional[float]]:
        """Returns (estimated_3d_position, semantic_region, association_confidence)."""
        if not self.available:
            self.last_failure_reason = "reconstruction_geometry_unavailable"
            return None, None, None

        camera = self._camera_for_frame(frame_id)
        if camera is None:
            self.last_failure_reason = f"camera_pose_unregistered_for_frame_{frame_id}"
            return None, None, None

        cam_pos = np.array(camera.get("position", []), dtype=float)
        cam_rot = camera.get("rotation", [])
        if len(cam_pos) != 3 or len(cam_rot) != 4:
            self.last_failure_reason = "invalid_camera_pose_parameters"
            return None, None, None

        # Camera rotation matrix R (world-from-camera in glTF frame)
        R = _quat_to_rot_matrix(cam_rot)

        # In glTF frame, camera looks down -Z, +X is right, +Y is up
        # Default frame resolution
        img_w, img_h = 1920.0, 1080.0
        intrinsics = camera.get("intrinsics") or {}
        if "width" in intrinsics and "height" in intrinsics:
            img_w = float(intrinsics["width"])
            img_h = float(intrinsics["height"])

        focal_length = img_w * 1.2
        cx, cy = img_w / 2.0, img_h / 2.0
        if "params" in intrinsics and len(intrinsics["params"]) >= 3:
            params = intrinsics["params"]
            focal_length = float(params[0])
            cx = float(params[1])
            cy = float(params[2])

        # Bounding box center in pixel coordinates
        x1, y1, x2, y2 = bbox
        u = (x1 + x2) / 2.0
        v = (y1 + y2) / 2.0
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)

        # Unproject pixel (u, v) into camera-frame ray direction (glTF: X-right, Y-up, -Z forward)
        # Note: image pixels are Y-down, so Y_cam = -(v - cy)
        ray_cam = np.array([(u - cx) / focal_length, -(v - cy) / focal_length, -1.0])
        ray_cam = ray_cam / np.linalg.norm(ray_cam)

        # Transform ray into world space
        ray_world = R @ ray_cam
        ray_world = ray_world / np.linalg.norm(ray_world)

        # Compute angular radius of the bounding box
        angular_radius = np.arctan2(max(box_w, box_h) / 2.0, focal_length)
        # Expand tolerance slightly for perspective foreshortening
        max_angle = max(angular_radius * 1.5, 0.05)

        # Vector from camera to all points: (N, 3)
        vecs = self.points - cam_pos
        depths = np.dot(vecs, ray_world)

        # Filter points in front of camera
        valid_mask = depths > 0.1
        if not np.any(valid_mask):
            self.last_failure_reason = "all_reconstructed_points_behind_camera"
            return None, None, None

        valid_vecs = vecs[valid_mask]
        valid_depths = depths[valid_mask]
        valid_pts = self.points[valid_mask]

        norms = np.linalg.norm(valid_vecs, axis=1)
        cos_angles = valid_depths / (norms + 1e-8)
        angles = np.arccos(np.clip(cos_angles, -1.0, 1.0))

        # Points within the cone of the bounding box ray
        cone_mask = angles <= max_angle
        if not np.any(cone_mask):
            self.last_failure_reason = "no_3d_points_in_bounding_ray_cone"
            return None, None, None

        candidate_points = valid_pts[cone_mask]
        candidate_depths = valid_depths[cone_mask]

        # Use median depth to reject background floaters
        med_depth = np.median(candidate_depths)
        depth_inlier_mask = np.abs(candidate_depths - med_depth) <= (med_depth * 0.35)
        inlier_points = candidate_points[depth_inlier_mask]

        if len(inlier_points) == 0:
            inlier_points = candidate_points

        # Calculate estimated 3D position
        est_3d = np.median(inlier_points, axis=0)
        est_pos = [round(float(c), 3) for c in est_3d]

        # Association confidence based on point support and ray alignment
        n_pts = len(inlier_points)
        assoc_conf = round(float(min(0.90, max(0.35, 0.35 + (n_pts / 50.0)))), 3)

        self.last_failure_reason = None
        return est_pos, None, assoc_conf
