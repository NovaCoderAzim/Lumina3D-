"""
detector.py

Thin wrapper around a pretrained Ultralytics YOLO model. Responsible ONLY
for running inference on a single image and returning raw detections above
a configurable confidence threshold.

Does not know about tracking, semantics, or reconstruction association ΓÇö
those live in tracker.py / semantic.py / association.py, kept behind
interface.py.

IMPORTANT: detection confidence is NOT accuracy. This module stores and
reports YOLO's raw confidence score only; nothing here should ever be
labeled "accuracy".
"""

import logging
from pathlib import Path
from typing import List, Optional

from schemas import Detection

logger = logging.getLogger("ai.detector")


class DetectorError(Exception):
    """Raised when the detection model cannot be loaded. Caught by
    interface.py so a missing model degrades to an empty-but-valid result
    instead of crashing the whole pipeline."""


class YOLODetector:
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.40,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device or self._select_device()
        self.model = self._load_model()

    def _select_device(self) -> str:
        try:
            import torch

            if torch.cuda.is_available():
                logger.info("CUDA available ΓÇö using GPU")
                return "cuda"
        except ImportError:
            logger.warning("PyTorch not importable ΓÇö falling back to CPU")
        logger.info("Using CPU for inference")
        return "cpu"

    def _load_model(self):
        try:
            from ultralytics import YOLO

            return YOLO(self.model_name)
        except Exception as exc:
            # Covers: ultralytics not installed, weights missing/corrupt,
            # no internet to download pretrained weights, etc.
            raise DetectorError(
                f"Failed to load model '{self.model_name}': {exc}"
            ) from exc

    def detect(self, image_path: str, frame_id: int) -> List[Detection]:
        """Run inference on a single image and return filtered detections.

        Never raises for an unreadable/invalid image ΓÇö logs a warning and
        returns an empty list. "No detections" is a valid, non-error
        outcome (see spec section 18).
        """
        if not Path(image_path).exists():
            logger.warning("Image not found, skipping: %s", image_path)
            return []

        try:
            results = self.model.predict(
                source=image_path,
                device=self.device,
                conf=self.confidence_threshold,
                verbose=False,
            )
        except Exception as exc:
            logger.error("Inference failed on %s: %s", image_path, exc)
            return []

        detections: List[Detection] = []
        for result in results:
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue
                cls_id = int(box.cls[0])
                class_name = names.get(cls_id, str(cls_id))
                xyxy = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        frame_id=frame_id,
                        class_name=class_name,
                        confidence=conf,
                        bbox=xyxy,
                    )
                )
        return detections
