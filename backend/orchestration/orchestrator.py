"""
Pipeline orchestrator (guide section 8).

process_project(project_id):
    PROCESSING
        vision.process(video)            -> FRAME_SELECTION
        reconstruction.process(frames)   -> RECONSTRUCTION
        ai.process(frames)               -> SEMANTIC_ANALYSIS
        analytics.process(...)           -> FINALIZATION
    COMPLETED

Any module failure moves the project to FAILED and stores a
user-friendly error. The orchestrator never lets an exception escape to
the API layer as a 500 with a raw traceback (guide section 11).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from ..config import get_settings
from ..models import ProcessingError, ProcessingStage, ProjectStatus
from ..services import (
    ai_service,
    analytics_service,
    reconstruction_service,
    vision_service,
)
from ..utils import get_logger, log_module_event
from .state import STORE, Project

logger = get_logger("backend.orchestrator")


def _set_stage(project: Project, stage: ProcessingStage, progress: float, message: str) -> None:
    project.stage = stage
    project.progress = progress
    project.message = message
    STORE.save(project)
    logger.info("project=%s stage=%s progress=%s %s", project.project_id, stage.value, progress, message)


def _load_recon_metadata(project_id: str) -> Optional[dict]:
    settings = get_settings()
    meta = settings.project_reconstruction_dir(project_id) / "reconstruction_metadata.json"
    if meta.exists():
        try:
            return json.loads(meta.read_text())
        except Exception:  # noqa: BLE001
            return None
    return None


def process_project(project_id: str) -> None:
    """Run the full pipeline for a project. Safe to call in a background
    thread — all failures are captured onto the project state."""
    settings = get_settings()
    project = STORE.get(project_id)
    if project is None:
        logger.error("process_project called for unknown project %s", project_id)
        return

    overall_start = time.perf_counter()
    try:
        project.transition(ProjectStatus.PROCESSING)
    except ValueError:
        # Already terminal / not uploaded — nothing to do.
        logger.warning("project %s not in an uploadable state (%s)", project_id, project.status)
        return
    STORE.save(project)

    video_path = ""
    if project.video_filename:
        video_path = str(settings.project_input_dir(project_id) / project.video_filename)

    try:
        # ---- 0. Capture-quality analysis (fast, honest go/no-go) ----
        if video_path:
            from ..services import geo_service
            cq = geo_service.analyze_capture(video_path)
            if cq:
                project.capture_quality = cq
                STORE.save(project)
                logger.info(
                    "project=%s capture verdict=%s score=%.0f motion=%s",
                    project_id, cq.get("verdict"), cq.get("score", 0), cq.get("motion_pattern"),
                )

        # ---- 1. Vision / frame selection ----
        _set_stage(project, ProcessingStage.VIDEO_ANALYSIS, 5, "Analysing uploaded video")
        frame_result = vision_service.process(video_path, project_id)
        _set_stage(
            project,
            ProcessingStage.FRAME_SELECTION,
            25,
            f"Selected {frame_result.selected_frames} frames",
        )

        # ---- 2. Reconstruction ----
        def on_reconstruction_progress(pct: float, msg: str) -> None:
            _set_stage(project, ProcessingStage.RECONSTRUCTION, round(pct, 1), msg)

        _set_stage(project, ProcessingStage.RECONSTRUCTION, 45, "Initializing 3D reconstruction...")
        reconstruction = reconstruction_service.process(
            frame_result.frames_directory, project_id, progress_callback=on_reconstruction_progress
        )

        model_path = reconstruction.get("model")
        if model_path:
            project.model_path = str(model_path)

        # ---- 2b. Metric scale (GPS) + confidence/coverage (challenges viii, i, vii) ----
        try:
            from ..services import geo_service
            geo = geo_service.compute_metric_and_confidence(
                project_id, reconstruction, video_path, frame_result.selected_frames
            )
            project.metric_scale = geo.get("metric_scale")
            project.confidence = geo.get("confidence")
        except Exception as exc:  # noqa: BLE001
            logger.warning("geo/confidence step skipped: %s", exc)
        # Dynamic-object masking stats, if the reconstruction reported them.
        dm = reconstruction.get("dynamic_masking")
        if dm:
            project.dynamic_masking = dm

        # ---- 3. AI / semantic ----
        _set_stage(project, ProcessingStage.SEMANTIC_ANALYSIS, 70, "Detecting semantic objects")
        recon_metadata = _load_recon_metadata(project_id)
        ai_result = ai_service.process(
            frame_result.frames_directory, project_id, recon_metadata
        )
        project.objects = ai_result.get("objects", [])

        # ---- 4. Analytics ----
        _set_stage(project, ProcessingStage.FINALIZATION, 90, "Computing analytics and measurements")
        analytics = analytics_service.process(
            project_id, frame_result, reconstruction, ai_result,
            metric_scale=project.metric_scale,
            confidence=project.confidence,
            dynamic_masking=project.dynamic_masking,
        )
        project.analytics = analytics

        # ---- Done ----
        project.transition(ProjectStatus.COMPLETED)
        project.stage = None
        project.message = "Digital twin ready (Observed 3D reconstruction)"
        project.error = None
        STORE.save(project)

        total = round(time.perf_counter() - overall_start, 2)
        log_module_event(
            logger,
            project_id=project_id,
            module="pipeline",
            status="COMPLETED",
            processing_time=total,
        )

    except reconstruction_service.ReconstructionFailure as exc:
        _fail(project, exc.error, "reconstruction", overall_start)
    except ai_service.AIFailure as exc:
        _fail(project, exc.error, "ai", overall_start)
    except Exception as exc:  # noqa: BLE001
        _fail(
            project,
            ProcessingError(
                title="Processing failed",
                detail=str(exc),
                cause="pipeline_exception",
                suggestion="Check the backend logs, or run in full mock mode "
                "(USE_MOCK_*=true) to demonstrate the application.",
            ),
            "pipeline",
            overall_start,
        )


def _fail(project: Project, error: ProcessingError, module: str, start: float) -> None:
    project.fail(error)
    STORE.save(project)
    log_module_event(
        logger,
        project_id=project.project_id,
        module=module,
        status="FAILED",
        processing_time=round(time.perf_counter() - start, 2),
        error=error.detail,
    )
