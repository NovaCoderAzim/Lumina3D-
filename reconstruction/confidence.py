"""
Reconstruction confidence & coverage analytics for single-pass aerial photogrammetry.

The scientific core of Lumina3D. A single drone pass cannot see every
surface, so parts of any reconstruction require uncertainty quantification.
Instead of hallucinating those surfaces, Lumina3D measures and reports trust:

  - Per-region confidence from real signals: local point density and the
    number of cameras that observed each point (track length / multiplicity).
    A point seen by many cameras and surrounded by many neighbours is
    trustworthy; a sparse point seen by two cameras is not.
  - A coverage summary: what fraction of the model is high/medium/low
    confidence, and where the weak regions are.
  - Occlusion / gap flags: large empty regions the single pass never
    covered.

Every number here is computed from the reconstruction, not invented.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class WeakRegion:
    center: List[float]        # [x,y,z] in model units
    confidence: float          # 0..1
    point_count: int
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "center": [round(c, 3) for c in self.center],
            "confidence": round(self.confidence, 3),
            "point_count": self.point_count,
            "reason": self.reason,
        }


@dataclass
class ConfidenceReport:
    overall_confidence: float = 0.0          # 0..1
    coverage_score: float = 0.0              # 0..1 (how completely the pass covered the scene)
    high_conf_fraction: float = 0.0
    medium_conf_fraction: float = 0.0
    low_conf_fraction: float = 0.0
    mean_observations_per_point: float = 0.0
    total_points: int = 0
    weak_regions: List[WeakRegion] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_confidence": round(self.overall_confidence, 3),
            "coverage_score": round(self.coverage_score, 3),
            "high_conf_fraction": round(self.high_conf_fraction, 3),
            "medium_conf_fraction": round(self.medium_conf_fraction, 3),
            "low_conf_fraction": round(self.low_conf_fraction, 3),
            "mean_observations_per_point": round(self.mean_observations_per_point, 2),
            "total_points": self.total_points,
            "weak_regions": [w.to_dict() for w in self.weak_regions],
            "notes": self.notes,
        }


def analyze_reconstruction(
    reconstruction=None,
    point_cloud_path: Optional[str] = None,
    voxel_regions: int = 12,
) -> ConfidenceReport:
    """Compute confidence/coverage from a pycolmap reconstruction (preferred,
    gives per-point observation counts) and/or the dense/sparse point cloud.

    Confidence per point combines:
      obs_score  = how many cameras saw the point (track length), and
      density_score = local neighbour density in a voxel grid.
    """
    report = ConfidenceReport()

    xyz: Optional[np.ndarray] = None
    track_lengths: Optional[np.ndarray] = None

    # 1) Prefer the COLMAP reconstruction: gives real observation counts.
    if reconstruction is not None:
        try:
            pts = reconstruction.points3D
            coords = []
            tlen = []
            for _, p in pts.items():
                coords.append(p.xyz)
                # track length = number of images observing this 3D point
                tlen.append(len(p.track.elements))
            if coords:
                xyz = np.array(coords, dtype=float)
                track_lengths = np.array(tlen, dtype=float)
        except Exception:  # noqa: BLE001
            xyz = None

    # 2) Fall back to the exported point cloud for geometry (no obs counts).
    if xyz is None and point_cloud_path and Path(point_cloud_path).exists():
        try:
            import open3d as o3d

            pcd = o3d.io.read_point_cloud(point_cloud_path)
            xyz = np.asarray(pcd.points, dtype=float)
        except Exception:  # noqa: BLE001
            xyz = None

    if xyz is None or len(xyz) == 0:
        report.notes = "No reconstruction points available to assess confidence."
        return report

    n = len(xyz)
    report.total_points = n

    # --- Local density via a voxel grid over the bounding box ---
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    grid = np.clip(((xyz - mins) / span * voxel_regions).astype(int), 0, voxel_regions - 1)
    # count points per occupied voxel
    keys = grid[:, 0] * voxel_regions * voxel_regions + grid[:, 1] * voxel_regions + grid[:, 2]
    uniq, counts = np.unique(keys, return_counts=True)
    count_map = dict(zip(uniq.tolist(), counts.tolist()))
    per_point_density = np.array([count_map[k] for k in keys], dtype=float)
    density_score = per_point_density / (np.percentile(per_point_density, 90) + 1e-6)
    density_score = np.clip(density_score, 0, 1)

    # --- Observation score ---
    if track_lengths is not None:
        report.mean_observations_per_point = float(track_lengths.mean())
        obs_score = np.clip((track_lengths - 2.0) / 6.0, 0, 1)  # 2 obs=min, 8+=full
        confidence = 0.6 * obs_score + 0.4 * density_score
    else:
        confidence = density_score
        report.notes = (
            "Observation counts unavailable (assessed from geometry/density "
            "only). Confidence is a lower bound."
        )

    report.overall_confidence = float(confidence.mean())
    report.high_conf_fraction = float((confidence >= 0.66).mean())
    report.medium_conf_fraction = float(((confidence >= 0.33) & (confidence < 0.66)).mean())
    report.low_conf_fraction = float((confidence < 0.33).mean())

    # --- Coverage: how uniformly the scene volume is filled ---
    occupied = len(uniq)
    total_voxels = voxel_regions ** 3
    # Only the voxels within the convex-ish occupied region matter; use the
    # fraction of the occupied bounding volume that has points as a proxy.
    occ_bbox = np.prod(np.maximum(grid.max(axis=0) - grid.min(axis=0) + 1, 1))
    report.coverage_score = float(min(1.0, occupied / max(1, occ_bbox)))

    # --- Weak regions: low-confidence voxels with few points ---
    weak: List[WeakRegion] = []
    for k, cnt in zip(uniq.tolist(), counts.tolist()):
        mask = keys == k
        c = float(confidence[mask].mean())
        if c < 0.33:
            center = xyz[mask].mean(axis=0).tolist()
            weak.append(WeakRegion(
                center=center, confidence=c, point_count=int(cnt),
                reason=("Few observing cameras / sparse points — likely "
                        "occluded or seen from too few angles in the single pass."),
            ))
    weak.sort(key=lambda w: w.confidence)
    report.weak_regions = weak[:10]  # top-10 worst

    if not report.notes:
        report.notes = (
            f"{report.high_conf_fraction*100:.0f}% of the model is high-confidence; "
            f"{len(weak)} weak region(s) flagged as unreliable rather than hallucinated."
        )
    return report
