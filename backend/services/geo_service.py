"""
Geo/metric + confidence post-processing service.

After reconstruction, this:
  1. Loads GPS/telemetry for the video (real sidecar or synthetic).
  2. Georeferences + recovers metric scale by aligning cameras to GPS.
  3. Computes per-region reconstruction confidence + coverage.

Kept separate from reconstruction_service so it degrades independently:
if georef/confidence fail, reconstruction still succeeds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ..config import get_settings
from ..utils import get_logger, log_module_event

logger = get_logger("backend.services.geo")


def compute_metric_and_confidence(
    project_id: str,
    reconstruction: Dict[str, Any],
    video_path: Optional[str],
    num_frames: int,
) -> Dict[str, Any]:
    """Returns {metric_scale, confidence} dicts (or None on failure)."""
    out: Dict[str, Any] = {"metric_scale": None, "confidence": None}
    settings = get_settings()

    cameras_json = reconstruction.get("camera_poses")
    point_cloud = reconstruction.get("point_cloud")

    # ---- Metric scale + georeference via GPS ----
    try:
        from telemetry import TelemetryLoader
        from reconstruction.georef import georeference_from_cameras

        track = TelemetryLoader().load_for_video(video_path, num_selected_frames=num_frames)
        if cameras_json and Path(cameras_json).exists():
            geo = georeference_from_cameras(cameras_json, track)
            if geo.ok and not geo.is_synthetic_gps:
                out["metric_scale"] = {
                    "metres_per_unit": geo.scale_metres_per_unit,
                    "rms_error_m": geo.rms_error_m,
                    "is_synthetic_gps": False,
                    "anchor_lat": geo.anchor_lat,
                    "anchor_lon": geo.anchor_lon,
                    "anchor_alt": geo.anchor_alt,
                }
                log_module_event(
                    logger, project_id=project_id, module="georef",
                    status="COMPLETED",
                )
            else:
                out["metric_scale"] = None
    except Exception as exc:  # noqa: BLE001
        logger.warning("georef skipped for %s: %s", project_id, exc)

    # ---- Confidence + coverage ----
    try:
        from reconstruction.confidence import analyze_reconstruction

        rep = analyze_reconstruction(point_cloud_path=point_cloud)
        out["confidence"] = {
            "overall_confidence": rep.overall_confidence,
            "coverage_score": rep.coverage_score,
            "high_conf_fraction": rep.high_conf_fraction,
            "medium_conf_fraction": rep.medium_conf_fraction,
            "low_conf_fraction": rep.low_conf_fraction,
            "weak_region_count": len(rep.weak_regions),
            "notes": rep.notes,
        }
        log_module_event(
            logger, project_id=project_id, module="confidence", status="COMPLETED",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("confidence analysis skipped for %s: %s", project_id, exc)

    return out


def analyze_capture(video_path: str) -> Optional[Dict[str, Any]]:
    """Fast pre-reconstruction capture-quality analysis."""
    try:
        from vision.capture_quality import analyze_capture as _analyze

        rep = _analyze(video_path)
        return rep.to_dict()
    except Exception as exc:  # noqa: BLE001
        logger.warning("capture analysis failed: %s", exc)
        return None
