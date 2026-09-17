"""
Analytics module adapter.

Aggregates analytics deterministically from vision, reconstruction, and AI
module outputs into unified photogrammetric & semantic statistics.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from ..models import (
    AnalyticsResponse,
    ConfidenceSummary,
    DynamicMaskingSummary,
    MetricScale,
    SemanticCategory,
    SemanticObject,
)
from ..utils import get_logger, log_module_event

logger = get_logger("backend.services.analytics")


def process(
    project_id: str,
    frame_result: Any,
    reconstruction: Dict[str, Any],
    ai_result: Dict[str, Any],
    metric_scale: Optional[Dict[str, Any]] = None,
    confidence: Optional[Dict[str, Any]] = None,
    dynamic_masking: Optional[Dict[str, Any]] = None,
) -> AnalyticsResponse:
    start = time.perf_counter()

    objects: List[SemanticObject] = ai_result.get("objects", [])

    vehicles = sum(1 for o in objects if o.category == SemanticCategory.VEHICLE)
    buildings = sum(1 for o in objects if o.category == SemanticCategory.BUILDING)
    people = sum(1 for o in objects if o.category == SemanticCategory.PERSON)
    vegetation = sum(1 for o in objects if o.category == SemanticCategory.VEGETATION)

    confidences = [o.confidence for o in objects]
    avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

    input_images = int(reconstruction.get("input_images", 0)) or getattr(
        frame_result, "selected_frames", 0
    )
    registered = int(reconstruction.get("registered_images", 0))
    reg_rate = reconstruction.get("registration_rate")
    if reg_rate is None:
        reg_rate = round(100 * registered / input_images, 2) if input_images else 0.0

    # Sum of module processing times we know about.
    proc_time = round(
        float(getattr(frame_result, "processing_time_seconds", 0.0))
        + float(reconstruction.get("adapter_time_seconds", 0.0))
        + float(ai_result.get("adapter_time_seconds", 0.0)),
        2,
    )

    response = AnalyticsResponse(
        input_images=input_images,
        registered_images=registered,
        registration_rate=float(reg_rate),
        sparse_points=int(reconstruction.get("sparse_points") or 0),
        dense_points=int(reconstruction.get("dense_points") or 0),
        objects=len(objects),
        vehicles=vehicles,
        buildings=buildings,
        people=people,
        vegetation=vegetation,
        average_detection_confidence=avg_conf,
        processing_time_seconds=proc_time,
        metric_scale=MetricScale(**metric_scale) if metric_scale else None,
        confidence=ConfidenceSummary(**confidence) if confidence else None,
        dynamic_masking=(
            DynamicMaskingSummary(
                dynamic_detections=dynamic_masking.get("dynamic_detections", 0),
                images_masked=dynamic_masking.get("images_masked", 0),
                total_images=dynamic_masking.get("total_images", 0),
                mean_masked_fraction=dynamic_masking.get("mean_masked_fraction", 0.0),
            )
            if dynamic_masking
            else None
        ),
        geometry_source=str(reconstruction.get("geometry_source", "SPARSE_SFM")),
        dense_engine=reconstruction.get("dense_engine"),
        surface_mesh_available=bool(reconstruction.get("surface_mesh_available", False)),
        dense_mvs_available=bool(reconstruction.get("dense_mvs_available", False)),
        texture_available=bool(reconstruction.get("texture_available", False)),
        splat_available=bool(reconstruction.get("splat_available", False)),
        dense_cloud_url=f"/api/projects/{project_id}/pointcloud/file" if (reconstruction.get("dense_mvs_available") or (reconstruction.get("dense_points") or 0) > 0) else None,
        textured_model_url=f"/api/projects/{project_id}/model/textured" if reconstruction.get("texture_available") else None,
        splat_url=f"/api/projects/{project_id}/splat/file" if reconstruction.get("splat_available") else None,
        gps_available=bool(metric_scale is not None and not metric_scale.get("is_synthetic_gps", True)),
        telemetry_source=str(reconstruction.get("telemetry_source", "none")) if reconstruction.get("telemetry_source") else None,
        metric_scale_available=bool(metric_scale is not None),
        georeferenced=bool(metric_scale is not None and metric_scale.get("anchor_lat") is not None),
        reprojection_error=reconstruction.get("reprojection_error"),
        warnings=list(reconstruction.get("warnings", [])),
        limitations=list(reconstruction.get("limitations", [])),
    )

    elapsed = round(time.perf_counter() - start, 3)
    log_module_event(
        logger,
        project_id=project_id,
        module="analytics",
        status="COMPLETED",
        processing_time=elapsed,
    )
    return response
