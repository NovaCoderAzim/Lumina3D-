"""
3D Gaussian Splatting (3DGS) Exporter for DRISHTI-3D
Converts dense MVS point clouds, normals, and camera visibility into standard
anisotropic 3D Gaussian Splats (.ply) for photorealistic real-time WebGL radiance rendering.
Compatible with @mkkellogg/gaussian-splats-3d, SuperSplat, LichtFeld, and gsplat.
"""

import os
import struct
import logging
from pathlib import Path
from typing import Optional, Union, Tuple
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

logger = logging.getLogger("reconstruction.splat")

# Spherical Harmonics constant for degree 0
SH_C0 = 0.28209479177387814


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Vectorized conversion of (N, 3, 3) rotation matrices to (N, 4) quaternions [w, x, y, z]."""
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    q = np.zeros((len(R), 4), dtype=np.float32)
    
    # Case 1: tr > 0
    mask1 = tr > 0
    if np.any(mask1):
        s = np.sqrt(tr[mask1] + 1.0) * 2.0  # s = 4 * qw
        q[mask1, 0] = 0.25 * s
        q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s
        q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s
        q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s

    # Case 2: R[0,0] > R[1,1] and R[0,0] > R[2,2]
    mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    if np.any(mask2):
        s = np.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2.0
        q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s
        q[mask2, 1] = 0.25 * s
        q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s
        q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s

    # Case 3: R[1,1] > R[2,2]
    mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    if np.any(mask3):
        s = np.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2.0
        q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s
        q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s
        q[mask3, 2] = 0.25 * s
        q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s

    # Case 4: default
    mask4 = (~mask1) & (~mask2) & (~mask3)
    if np.any(mask4):
        s = np.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2.0
        q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s
        q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s
        q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s
        q[mask4, 3] = 0.25 * s

    norms = np.linalg.norm(q, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return q / norms


def normals_to_rotations(normals: np.ndarray) -> np.ndarray:
    """
    Constructs an orthonormal frame [t1, t2, normal] for each surfel and converts
    to a rotation quaternion [w, x, y, z] aligning the z-axis with the normal.
    """
    N = len(normals)
    z_axis = np.asarray(normals, dtype=np.float32)
    norms = np.linalg.norm(z_axis, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    z_axis = z_axis / norms

    # Pick an arbitrary reference vector not parallel to z_axis
    ref = np.zeros_like(z_axis)
    ref[:, 0] = 1.0
    parallel_mask = np.abs(z_axis[:, 0]) > 0.9
    ref[parallel_mask, 0] = 0.0
    ref[parallel_mask, 1] = 1.0

    # Tangent vector 1: t1 = ref x z_axis
    t1 = np.cross(ref, z_axis)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)

    # Tangent vector 2: t2 = z_axis x t1
    t2 = np.cross(z_axis, t1)
    t2 /= np.linalg.norm(t2, axis=1, keepdims=True)

    # Rotation matrix R with columns [t1, t2, z_axis]
    R = np.stack([t1, t2, z_axis], axis=-1)
    return rotation_matrix_to_quaternion(R)


def generate_gaussian_splats_from_point_cloud(
    pcd: o3d.geometry.PointCloud,
    output_ply_path: Union[str, Path],
    k_neighbors: int = 4,
    scale_multiplier: float = 1.35,
    min_scale: float = 0.005,
    max_scale: float = 0.15,
    opacity_value: float = 0.96,
) -> int:
    """
    Constructs a standard 3D Gaussian Splat PLY from an Open3D PointCloud.
    
    Each 3D Gaussian comprises:
    - Position (x, y, z)
    - Normal (nx, ny, nz)
    - Diffuse Color (f_dc_0, f_dc_1, f_dc_2) in spherical harmonics space
    - Opacity logit
    - Log-scale (scale_0, scale_1, scale_2) where scale_0 and scale_1 form a tangent disk
      and scale_2 represents the surface thickness
    - Rotation quaternion (rot_0, rot_1, rot_2, rot_3) [w, x, y, z]
    """
    output_ply_path = Path(output_ply_path)
    output_ply_path.parent.mkdir(parents=True, exist_ok=True)

    pts = np.asarray(pcd.points, dtype=np.float32)
    N = len(pts)
    if N == 0:
        logger.warning("Empty point cloud passed to Gaussian Splat generation.")
        return 0

    logger.info("Generating 3D Gaussian Splats for %d points...", N)

    # 1. Colors to Spherical Harmonics Degree 0
    if pcd.has_colors():
        colors = np.asarray(pcd.colors, dtype=np.float32)
    else:
        colors = np.ones((N, 3), dtype=np.float32) * 0.8
    colors = np.clip(colors, 0.0, 1.0)
    # SH_0 = (RGB - 0.5) / C0
    f_dc = (colors - 0.5) / SH_C0

    # 2. Normals
    if not pcd.has_normals() or len(pcd.normals) == 0:
        logger.info("Computing vertex normals for 3D Gaussian Splats...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
    normals = np.asarray(pcd.normals, dtype=np.float32)
    # Ensure unit normals
    n_norms = np.linalg.norm(normals, axis=1, keepdims=True)
    n_norms[n_norms == 0] = 1.0
    normals = normals / n_norms

    # 3. Anisotropic Scales via local k-NN spacing
    logger.info("Computing adaptive spatial footprint via k-NN spatial graph...")
    tree = cKDTree(pts)
    distances, _ = tree.query(pts, k=min(k_neighbors, N), workers=-1)
    # Mean distance to nearest neighbors (excluding self at index 0)
    if distances.shape[1] > 1:
        mean_dist = np.mean(distances[:, 1:], axis=1)
    else:
        mean_dist = np.full(N, 0.03, dtype=np.float32)

    # Radii along tangent axes
    r_tangent = np.clip(mean_dist * scale_multiplier, min_scale, max_scale)
    # Thickness along normal axis (thin surfel disk)
    r_normal = np.clip(r_tangent * 0.15, min_scale * 0.2, max_scale * 0.2)

    # 3DGS scale properties are stored in log space: log(scale)
    log_scale_0 = np.log(r_tangent).astype(np.float32)
    log_scale_1 = np.log(r_tangent).astype(np.float32)
    log_scale_2 = np.log(r_normal).astype(np.float32)

    # 4. Rotation Quaternions [w, x, y, z]
    logger.info("Computing surfel orientation quaternions...")
    quats = normals_to_rotations(normals)  # (N, 4) in order [w, x, y, z]

    # 5. Opacity in logit space: logit(p) = log(p / (1 - p))
    op = np.clip(opacity_value, 0.01, 0.99)
    logit_opacity = np.full(N, np.log(op / (1.0 - op)), dtype=np.float32)

    # 6. Binary Little-Endian PLY packing
    # Total properties per vertex: 3(xyz) + 3(normal) + 3(sh) + 1(opacity) + 3(scale) + 4(rot) = 17 floats = 68 bytes
    logger.info("Writing standard 3DGS binary PLY to %s...", output_ply_path)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {N}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property float f_dc_0\n"
        "property float f_dc_1\n"
        "property float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\n"
        "property float scale_1\n"
        "property float scale_2\n"
        "property float rot_0\n"
        "property float rot_1\n"
        "property float rot_2\n"
        "property float rot_3\n"
        "end_header\n"
    )

    # Pack into structured contiguous array for high-speed I/O
    vertex_dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
        ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
        ("opacity", "<f4"),
        ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
        ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
    ])

    structured_data = np.empty(N, dtype=vertex_dtype)
    structured_data["x"] = pts[:, 0]
    structured_data["y"] = pts[:, 1]
    structured_data["z"] = pts[:, 2]
    structured_data["nx"] = normals[:, 0]
    structured_data["ny"] = normals[:, 1]
    structured_data["nz"] = normals[:, 2]
    structured_data["f_dc_0"] = f_dc[:, 0]
    structured_data["f_dc_1"] = f_dc[:, 1]
    structured_data["f_dc_2"] = f_dc[:, 2]
    structured_data["opacity"] = logit_opacity
    structured_data["scale_0"] = log_scale_0
    structured_data["scale_1"] = log_scale_1
    structured_data["scale_2"] = log_scale_2
    structured_data["rot_0"] = quats[:, 0]  # w
    structured_data["rot_1"] = quats[:, 1]  # x
    structured_data["rot_2"] = quats[:, 2]  # y
    structured_data["rot_3"] = quats[:, 3]  # z

    with open(output_ply_path, "wb") as f:
        f.write(header.encode("ascii"))
        structured_data.tofile(f)

    file_size_mb = output_ply_path.stat().st_size / (1024 * 1024)
    logger.info("Successfully exported 3D Gaussian Splats: %d splats (%.2f MB) -> %s",
                N, file_size_mb, output_ply_path)
    return N
