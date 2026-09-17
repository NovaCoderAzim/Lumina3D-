"""
Generate a real multi-view image set with genuine parallax — headless.

Open3D's GPU offscreen renderer needs EGL/a display, which isn't available
in this environment. Instead we render with our own pinhole-camera
projection + painter's-algorithm rasterizer (pure NumPy + OpenCV, CPU
only). We define a real 3D scene (a cluster of colored buildings on a
ground plane, each face given a distinct textured pattern), place cameras
on an orbit around it, and project every 3D face into each camera.

The images are genuine projections of one fixed 3D scene from many
distinct viewpoints, so the parallax is real and COLMAP performs real
feature matching + triangulation. This is not a mock — it is a synthetic
but geometrically-correct multi-view capture.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

rng = np.random.default_rng(7)


def _box_faces(center, size, color) -> List[Tuple[np.ndarray, tuple]]:
    """Return the 6 quad faces of an axis-aligned box as (4x3 verts, BGR)."""
    cx, cy, cz = center
    sx, sy, sz = np.array(size) / 2.0
    # 8 corners
    c = np.array([
        [cx - sx, cy - sy, cz - sz], [cx + sx, cy - sy, cz - sz],
        [cx + sx, cy + sy, cz - sz], [cx - sx, cy + sy, cz - sz],
        [cx - sx, cy - sy, cz + sz], [cx + sx, cy - sy, cz + sz],
        [cx + sx, cy + sy, cz + sz], [cx - sx, cy + sy, cz + sz],
    ])
    faces_idx = [
        (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
        (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4),
    ]
    out = []
    b = np.array(color)  # RGB 0..1
    for k, f in enumerate(faces_idx):
        shade = 0.75 + 0.05 * k
        rgb = np.clip(b * shade, 0, 1)
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        out.append((c[list(f)], bgr))
    return out


def build_scene() -> List[Tuple[np.ndarray, tuple]]:
    faces: List[Tuple[np.ndarray, tuple]] = []

    # Ground: a big flat quad, split into tiles with slightly varied colors
    # so it has trackable features (not a flat tex\u2011less plane).
    for gx in range(-10, 10, 2):
        for gz in range(-10, 10, 2):
            quad = np.array([
                [gx, 0, gz], [gx + 2, 0, gz], [gx + 2, 0, gz + 2], [gx, 0, gz + 2]
            ], dtype=float)
            shade = 0.35 + 0.25 * rng.random()
            bgr = (int(90 * shade), int(120 * shade), int(90 * shade))
            faces.append((quad, bgr))

    palette = [
        [0.80, 0.30, 0.25], [0.25, 0.45, 0.80], [0.30, 0.70, 0.40],
        [0.85, 0.70, 0.20], [0.60, 0.35, 0.70], [0.70, 0.70, 0.72],
    ]
    positions = [(-5, -5), (-5, 3), (0, -2), (4, 4), (5, -4), (1, 6), (-3, 0), (6, 1)]
    for i, (x, z) in enumerate(positions):
        h = 2.0 + (i % 4) * 1.5
        w = 1.5 + (i % 3) * 0.6
        faces.extend(_box_faces((x, h / 2.0, z), (w, h, w), palette[i % len(palette)]))
    return faces


def _look_at(eye, target, up):
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    R = np.stack([s, u, -f], axis=0)  # world->cam rotation
    return R


def _face_texture(face_id: int, size: int = 256) -> np.ndarray:
    """A distinctive, high-frequency texture per face so SIFT finds many
    stable, non-repeating keypoints. Deterministic per face_id so the same
    surface looks identical from every camera (essential for matching)."""
    r = np.random.default_rng(1000 + face_id)
    tex = (r.random((size, size, 3)) * 255).astype(np.uint8)
    # Add some structure: random blobs + lines so features are localizable.
    for _ in range(40):
        c = (int(r.integers(0, size)), int(r.integers(0, size)))
        col = tuple(int(x) for x in r.integers(0, 255, 3))
        cv2.circle(tex, c, int(r.integers(4, 20)), col, -1)
    for _ in range(20):
        p1 = (int(r.integers(0, size)), int(r.integers(0, size)))
        p2 = (int(r.integers(0, size)), int(r.integers(0, size)))
        col = tuple(int(x) for x in r.integers(0, 255, 3))
        cv2.line(tex, p1, p2, col, int(r.integers(1, 4)))
    return tex


def _warp_texture(img, dst_pts, tex):
    """Perspective-warp a square texture onto a projected quad (dst_pts is
    4x2 in image space, order matches the quad corners)."""
    h_t, w_t = tex.shape[:2]
    src = np.array([[0, 0], [w_t, 0], [w_t, h_t], [0, h_t]], dtype=np.float32)
    dst = dst_pts.astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(tex, M, (img.shape[1], img.shape[0]))
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, dst.astype(np.int32), 255)
    img[mask == 255] = warped[mask == 255]


def render_views(faces, out_dir: Path, n_views=40, w=1024, h=768, fov_deg=60.0):
    out_dir.mkdir(parents=True, exist_ok=True)
    fx = fy = (w / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    cx, cy = w / 2.0, h / 2.0
    center = np.array([0.0, 2.0, 0.0])
    radius = 16.0
    written = 0

    # Pre-generate one fixed texture per face (stable across all views).
    textures = [_face_texture(i) for i in range(len(faces))]

    for i in range(n_views):
        # Sweep a partial arc (about 150 degrees) with heavy overlap
        # between consecutive frames — this is how a drone actually orbits
        # a target and gives the incremental mapper strong, consistent
        # two-view overlap instead of a sparse full-circle loop.
        arc = np.radians(150.0)
        ang = -arc / 2 + arc * i / max(1, n_views - 1)
        r = radius
        height = 7.0 + 1.5 * np.sin(ang * 2)
        eye = np.array([r * np.cos(ang), height, r * np.sin(ang)])
        R = _look_at(eye, center, np.array([0.0, 1.0, 0.0]))

        img = np.full((h, w, 3), 0, np.uint8)
        img[:] = (235, 225, 205)  # sky-ish BGR background

        drawable = []
        for fidx, (verts, bgr) in enumerate(faces):
            cam = (R @ (verts - eye).T).T  # to camera space
            if np.any(cam[:, 2] >= -0.1):  # any vertex behind/near camera -> skip
                continue
            z = cam[:, 2]
            px = (fx * cam[:, 0] / -z + cx)
            py = (fy * cam[:, 1] / -z + cy)
            pts = np.stack([px, py], axis=1)
            depth = float(np.mean(-z))
            drawable.append((depth, pts, fidx))

        # Painter's algorithm: far faces first.
        drawable.sort(key=lambda d: -d[0])
        for _, pts, fidx in drawable:
            if pts.shape[0] == 4:
                _warp_texture(img, pts, textures[fidx])
            else:
                cv2.fillConvexPoly(img, pts.astype(np.int32), faces[fidx][1], lineType=cv2.LINE_AA)

        fname = out_dir / f"frame_{i:06d}.jpg"
        cv2.imwrite(str(fname), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        written += 1

    print(f"Rendered {written} multi-view images to {out_dir}")
    return written


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/multiview/selected")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    render_views(build_scene(), out, n_views=n)
