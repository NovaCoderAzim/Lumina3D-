"""
interface.py

Public entry point for the Lumina3D AI & Semantic Intelligence module.
Everything else in this package (detector, tracker, semantic, association, mock)
is an implementation detail hidden behind AIProcessor.process().

Usage:

    from interface import AIProcessor

    processor = AIProcessor()
    result = processor.process(
        frames_directory="data/projects/demo_001/frames/selected",
        project_id="demo_001",
        reconstruction_metadata=None,
    )

Configuration (env vars, all optional, all overridable via constructor args):

    DETECTION_MODEL       default "yolov8n.pt"
    DETECTION_CONFIDENCE  default 0.40
    USE_GPU               default "true"
    USE_MOCK_AI           default "false"
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import mock as mock_module
from association import ReconstructionAssociator
from schemas import AIResult, AIStatistics, FrameDetections, TrackedObject
from semantic import build_summary, classify
from tracker import SimpleTracker

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ai.interface")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class AIProcessor:
    def __init__(
        self,
        model_name: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        use_gpu: Optional[bool] = None,
        use_mock: Optional[bool] = None,
    ):
        self.model_name = model_name or os.environ.get("DETECTION_MODEL", "yolov8n.pt")
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else float(os.environ.get("DETECTION_CONFIDENCE", 0.40))
        )
        self.use_gpu = (
            use_gpu
            if use_gpu is not None
            else os.environ.get("USE_GPU", "true").lower() == "true"
        )
        self.use_mock = (
            use_mock
            if use_mock is not None
            else os.environ.get("USE_MOCK_AI", "false").lower() == "true"
        )

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def process(
        self,
        frames_directory: str,
        project_id: str,
        reconstruction_metadata: Optional[Dict[str, Any]] = None,
    ) -> AIResult:
        start = time.time()

        if self.use_mock:
            logger.info("USE_MOCK_AI=true ΓÇö returning mock result for %s", project_id)
            result = mock_module.load_mock_result(project_id)
            self._write_outputs(project_id, detections_by_frame=[], result=result)
            return result

        try:
            return self._process_real(
                frames_directory, project_id, reconstruction_metadata, start
            )
        except Exception as exc:
            # Catch-all so a bug anywhere in the pipeline degrades to a
            # reported failure, never an unhandled crash that takes down
            # the orchestrator.
            logger.exception("AI processing failed for project %s", project_id)
            return AIResult(
                status="FAILED",
                project_id=project_id,
                objects=[],
                summary={},
                statistics=AIStatistics(
                    processing_time_seconds=round(time.time() - start, 2)
                ),
                error=str(exc),
            )

    # ---------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------

    def _process_real(
        self,
        frames_directory: str,
        project_id: str,
        reconstruction_metadata: Optional[Dict[str, Any]],
        start: float,
    ) -> AIResult:
        from detector import DetectorError, YOLODetector

        frames_dir = Path(frames_directory)
        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_directory}")

        frame_paths = sorted(
            p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not frame_paths:
            logger.warning("No frames found in %s ΓÇö returning empty result", frames_directory)
            return self._empty_result(project_id, start)

        try:
            detector = YOLODetector(
                model_name=self.model_name,
                confidence_threshold=self.confidence_threshold,
                device="cuda" if self.use_gpu else "cpu",
            )
        except DetectorError as exc:
            # Model / GPU unavailable is not a hard failure of the whole
            # system (spec section 18) ΓÇö report it and return an empty,
            # valid, COMPLETED result rather than crashing.
            logger.error("Model unavailable, returning empty result: %s", exc)
            result = self._empty_result(project_id, start)
            result.error = f"model_unavailable: {exc}"
            return result

        tracker = SimpleTracker()
        associator = ReconstructionAssociator(reconstruction_metadata)

        detections_by_frame: List[FrameDetections] = []
        total_detections = 0
        confidence_sum = 0.0

        for idx, frame_path in enumerate(frame_paths):
            frame_id = self._frame_id_from_path(frame_path, fallback=idx)
            dets = detector.detect(str(frame_path), frame_id)
            detections_by_frame.append(FrameDetections(frame_id=frame_id, detections=dets))
            tracker.update(dets)
            total_detections += len(dets)
            confidence_sum += sum(d.confidence for d in dets)

        tracked_objects = self._build_tracked_objects(tracker.finalize(), associator)
        summary = build_summary(tracked_objects)
        avg_conf = (confidence_sum / total_detections) if total_detections else 0.0

        statistics = AIStatistics(
            frames_processed=len(frame_paths),
            detections=total_detections,
            unique_objects=len(tracked_objects),
            average_detection_confidence=avg_conf,
            processing_time_seconds=round(time.time() - start, 2),
        )

        result = AIResult(
            status="COMPLETED",
            project_id=project_id,
            objects=tracked_objects,
            summary=summary,
            statistics=statistics,
        )

        self._write_outputs(project_id, detections_by_frame, result)
        return result

    def _build_tracked_objects(
        self, raw_tracks, associator: ReconstructionAssociator
    ) -> List[TrackedObject]:
        objects: List[TrackedObject] = []
        counters: Dict[str, int] = {}

        for track in raw_tracks:
            category = classify(track.class_name)
            counters[category] = counters.get(category, 0) + 1
            obj_id = f"{category}_{counters[category]:03d}"

            avg_conf = sum(track.confidences) / len(track.confidences)
            last_frame = track.frames_seen[-1]
            position, region, assoc_conf = associator.associate(track.last_bbox, last_frame)

            objects.append(
                TrackedObject(
                    id=obj_id,
                    class_name=track.class_name,
                    category=category,
                    confidence=avg_conf,
                    observations=len(track.frames_seen),
                    frames_seen=track.frames_seen,
                    bbox_history=track.bbox_history,
                    estimated_3d_position=position,
                    semantic_region=region,
                    association_confidence=assoc_conf,
                )
            )
        return objects

    @staticmethod
    def _empty_result(project_id: str, start: float) -> AIResult:
        return AIResult(
            status="COMPLETED",
            project_id=project_id,
            objects=[],
            summary={},
            statistics=AIStatistics(
                frames_processed=0,
                processing_time_seconds=round(time.time() - start, 2),
            ),
        )

    @staticmethod
    def _frame_id_from_path(path: Path, fallback: int) -> int:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return int(digits) if digits else fallback

    @staticmethod
    def _write_outputs(
        project_id: str,
        detections_by_frame: List[FrameDetections],
        result: AIResult,
    ) -> None:
        out_dir = Path(f"data/projects/{project_id}/ai")
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "detections.json", "w") as f:
            json.dump(
                {
                    "project_id": project_id,
                    "frames": [fd.to_dict() for fd in detections_by_frame],
                },
                f,
                indent=2,
            )

        with open(out_dir / "objects.json", "w") as f:
            json.dump(
                {
                    "project_id": project_id,
                    "objects": [o.to_dict() for o in result.objects],
                },
                f,
                indent=2,
            )

        with open(out_dir / "semantic.json", "w") as f:
            json.dump({"project_id": project_id, "summary": result.summary}, f, indent=2)

        logger.info("Wrote AI outputs to %s", out_dir)
