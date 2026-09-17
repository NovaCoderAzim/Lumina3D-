"""
USE_MOCK_RECONSTRUCTION=true

Enables testing of the 3D viewer and API layer without COLMAP, a GPU,
or a real drone dataset installed.

The schema returned here is identical to pipeline.py's real ReconstructionResult.
This module has no dependency on colmap_runner.py, pointcloud.py, or mesh.py —
only trimesh/numpy, so it runs anywhere.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import trimesh

from .schemas import (
    CameraPose,
    GpuInfo,
    ReconstructionError,
    ReconstructionResult,
    ReconstructionStatus,
)


def _make_demo_mesh() -> trimesh.Trimesh:
    """A small textured-looking icosphere stands in for a real scan -
    good enough to prove the viewer, camera path, and download pipeline
    all work end to end."""

    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)

    # Fake per-vertex color so the viewer has something to render besides
    # flat gray - mimics what the real pipeline bakes from the dense cloud.
    normals = mesh.vertex_normals
    colors = ((normals + 1.0) / 2.0 * 255).astype(np.uint8)
    alpha = np.full((colors.shape[0], 1), 255, dtype=np.uint8)
    mesh.visual.vertex_colors = np.hstack([colors, alpha])

    return mesh


def _make_demo_cameras(n: int = 12) -> list[CameraPose]:
    """n cameras on a ring around the demo mesh, already in glTF Y-up
    convention - same contract the real pipeline provides."""

    cameras = []
    radius = 3.0
    for i in range(n):
        angle = 2 * np.pi * i / n
        position = [radius * np.cos(angle), 0.5, radius * np.sin(angle)]
        # Identity-ish rotation is fine for a demo; a real implementation
        # would look-at the origin.
        rotation = [1.0, 0.0, 0.0, 0.0]
        cameras.append(
            CameraPose(
                image=f"frame_{i:06d}.jpg",
                camera_id=i,
                position=position,
                rotation=rotation,
            )
        )
    return cameras


def run_mock_reconstruction(
    frames_directory: str, project_id: str, output_dir: str | Path
) -> ReconstructionResult:
    start = time.perf_counter()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames_dir = Path(frames_directory)
    input_images = (
        len([p for p in frames_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if frames_dir.is_dir()
        else 176  # plausible fallback if the directory doesn't exist yet
    )
    registered_images = max(1, int(input_images * 0.96))

    mesh = _make_demo_mesh()
    glb_path = out / "model.glb"
    mesh.export(glb_path, file_type="glb")

    cloud_path = out / "cloud.ply"
    trimesh.PointCloud(mesh.vertices).export(cloud_path)

    cameras = _make_demo_cameras()
    cameras_path = out / "cameras.json"
    cameras_path.write_text(
        json.dumps([c.model_dump() for c in cameras], indent=2)
    )

    elapsed = time.perf_counter() - start

    result = ReconstructionResult(
        project_id=project_id,
        status=ReconstructionStatus.COMPLETED,
        model=str(glb_path),
        point_cloud=str(cloud_path),
        mesh=str(glb_path),
        camera_poses=str(cameras_path),
        input_images=input_images,
        registered_images=registered_images,
        registration_rate=round(100 * registered_images / input_images, 2),
        sparse_points=len(mesh.vertices) * 2,
        dense_points=len(mesh.vertices) * 20,
        mesh_vertices=len(mesh.vertices),
        mesh_faces=len(mesh.faces),
        processing_time_seconds=round(elapsed, 2),
        gpu=GpuInfo(available=False, name="mock-mode-no-gpu"),
        error=None,
    )

    metadata_path = out / "reconstruction_metadata.json"
    metadata_path.write_text(result.model_dump_json(indent=2))

    return result
