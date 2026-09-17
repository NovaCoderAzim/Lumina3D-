"""
Metric scaling & georeferencing (metric accuracy without GCPs).

COLMAP reconstructs up to an unknown similarity (arbitrary scale, rotation,
translation). We recover real-world metric scale — and a geo-anchor — by
aligning the estimated camera positions to the GPS flight track (in local
ENU metres) using a least-squares similarity (Umeyama) fit.

Output:
  - scale factor  (metres per reconstruction unit)
  - RMS alignment error (metres) -> a real, reportable metric-accuracy number
  - the similarity transform (so points/measurements convert to metres)
  - a geo-anchor (lat/lon/alt) so the model is georeferenced

This is genuine: with real GPS it yields real metric scale; with a
synthetic track it yields an illustrative scale, clearly flagged upstream.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class GeoreferenceResult:
    ok: bool
    scale_metres_per_unit: Optional[float] = None
    rms_error_m: Optional[float] = None
    num_correspondences: int = 0
    anchor_lat: Optional[float] = None
    anchor_lon: Optional[float] = None
    anchor_alt: Optional[float] = None
    is_synthetic_gps: bool = False
    # 4x4 similarity transform (reconstruction -> ENU metres), row-major.
    transform: List[List[float]] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "scale_metres_per_unit": round(self.scale_metres_per_unit, 6) if self.scale_metres_per_unit is not None else None,
            "rms_error_m": round(self.rms_error_m, 3) if self.rms_error_m is not None else None,
            "num_correspondences": self.num_correspondences,
            "anchor": {"lat": self.anchor_lat, "lon": self.anchor_lon, "alt": self.anchor_alt},
            "is_synthetic_gps": self.is_synthetic_gps,
            "notes": self.notes,
        }


def _umeyama(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity (scale s, rotation R, translation t) mapping
    src -> dst (both Nx3). Returns (s, R, t)."""
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_src = (src_c ** 2).sum() / n
    s = (D * np.diag(S)).sum() / var_src if var_src > 1e-12 else 1.0
    t = mu_dst - s * R @ mu_src
    return s, R, t


def georeference_from_cameras(
    cameras_json_path: str | Path,
    flight_track,  # telemetry.schemas.FlightTrack
    allow_synthetic: bool = False,
) -> GeoreferenceResult:
    """Align reconstruction camera positions to the GPS ENU track."""
    cams_path = Path(cameras_json_path)
    if not cams_path.exists():
        return GeoreferenceResult(ok=False, notes="No cameras.json to align.")

    cams = json.loads(cams_path.read_text())
    # cameras.json is a list of {image, position:[x,y,z], ...} in recon frame.
    # Sort by image name so ordering matches the frame index order.
    cams_sorted = sorted(cams, key=lambda c: str(c.get("image", "")))
    cam_positions = np.array([c["position"] for c in cams_sorted], dtype=float)

    if not flight_track.has_data():
        return GeoreferenceResult(
            ok=False,
            scale_metres_per_unit=None,
            rms_error_m=None,
            notes="No GPS telemetry available. Model coordinates are unscaled / relative.",
        )

    is_synth = getattr(flight_track, "is_synthetic", False)
    if is_synth and not allow_synthetic:
        return GeoreferenceResult(
            ok=False,
            scale_metres_per_unit=None,
            rms_error_m=None,
            is_synthetic_gps=True,
            notes="Telemetry is synthetic. Metric scaling and georeferencing disabled for scientific integrity.",
        )

    enu = np.array(flight_track.enu, dtype=float)

    # Correspond cameras to GPS fixes by uniform index resampling (both are
    # ordered along the single flight path).
    n_cams = len(cam_positions)
    n_fix = len(enu)
    if n_cams < 3 or n_fix < 3:
        return GeoreferenceResult(
            ok=False,
            scale_metres_per_unit=None,
            rms_error_m=None,
            num_correspondences=min(n_cams, n_fix),
            notes="Too few correspondences for a reliable similarity fit.",
        )

    idx = np.round(np.linspace(0, n_fix - 1, n_cams)).astype(int)
    gps_pts = enu[idx]

    s, R, t = _umeyama(cam_positions, gps_pts)

    # RMS alignment error in metres — the honest metric-accuracy figure.
    mapped = (s * (R @ cam_positions.T).T) + t
    rms = float(np.sqrt(((mapped - gps_pts) ** 2).sum(axis=1).mean()))

    transform = np.eye(4)
    transform[:3, :3] = s * R
    transform[:3, 3] = t

    return GeoreferenceResult(
        ok=True,
        scale_metres_per_unit=float(s),
        rms_error_m=rms,
        num_correspondences=n_cams,
        anchor_lat=flight_track.anchor_lat,
        anchor_lon=flight_track.anchor_lon,
        anchor_alt=flight_track.anchor_alt,
        is_synthetic_gps=bool(is_synth),
        transform=transform.tolist(),
        notes=(
            "Metric scale recovered by similarity-aligning camera track to real GPS telemetry."
            if not is_synth
            else "Metric scale recovered from synthetic track (DEV/TEST ONLY)."
        ),
    )
