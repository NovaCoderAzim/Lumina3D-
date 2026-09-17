"""
Dynamic object masking (dynamic objects — vehicles, humans, animals).

Moving objects break Structure-from-Motion: a car that drives through the
scene produces features that don't obey the static-scene epipolar geometry,
injecting outliers that corrupt camera poses and the point cloud.

This module runs YOLO on every frame and writes a COLMAP-compatible mask
per image: an 8-bit PNG the SAME SIZE as the image, where 0 (black) marks
pixels COLMAP must ignore (the detected dynamic objects) and 255 (white)
marks pixels to keep. COLMAP reads these via
ImageReaderOptions.mask_path / --ImageReader.mask_path, expecting a file
named "<image_name>.png" in the mask directory.

The result: features are extracted only from the static scene, so moving
vehicles/people/animals no longer corrupt the reconstruction. This is a
real, standard technique — not a cosmetic filter.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("reconstruction.masking")

# COCO classes that are dynamic and should be masked out of a static-scene
# reconstruction. Names match Ultralytics' COCO labels.
DYNAMIC_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "train", "truck",
    "boat", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "airplane",
}


class MaskingResult:
    def __init__(self) -> None:
        self.mask_dir: str = ""
        self.images_masked: int = 0
        self.total_images: int = 0
        self.dynamic_detections: int = 0
        self.mean_masked_fraction: float = 0.0
        self.enabled: bool = True
        self.notes: str = ""

    def to_dict(self) -> Dict:
        return {
            "mask_dir": self.mask_dir,
            "images_masked": self.images_masked,
            "total_images": self.total_images,
            "dynamic_detections": self.dynamic_detections,
            "mean_masked_fraction": round(self.mean_masked_fraction, 4),
            "enabled": self.enabled,
            "notes": self.notes,
        }


def generate_masks(
    frames_directory: str | Path,
    mask_directory: str | Path,
    model_name: str = "yolov8n.pt",
    device: Optional[str] = None,
    confidence: float = 0.25,
    dilate_px: int = 12,
) -> MaskingResult:
    """Run YOLO on each frame and write COLMAP masks. Dilation expands each
    detection box slightly so feature points on an object's blurry edge are
    also excluded."""
    import cv2
    import numpy as np

    result = MaskingResult()
    frames_dir = Path(frames_directory)
    mask_dir = Path(mask_directory)
    mask_dir.mkdir(parents=True, exist_ok=True)
    result.mask_dir = str(mask_dir)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    frames = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in exts)
    result.total_images = len(frames)
    if not frames:
        result.notes = "No frames to mask."
        return result

    try:
        from ultralytics import YOLO
        import torch

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = YOLO(model_name)
    except Exception as exc:  # noqa: BLE001
        result.enabled = False
        result.notes = f"Masking disabled (model unavailable): {exc}"
        logger.warning(result.notes)
        return result

    fractions: List[float] = []
    for fp in frames:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        h, w = img.shape[:2]
        mask = np.full((h, w), 255, np.uint8)  # keep everything by default

        preds = model.predict(source=str(fp), device=dev, conf=confidence, verbose=False)
        masked_here = 0
        for res in preds:
            names = getattr(res, "names", {}) or {}
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            for b in boxes:
                cls = names.get(int(b.cls[0]), "")
                if cls not in DYNAMIC_CLASSES:
                    continue
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                x1 = max(0, int(x1) - dilate_px)
                y1 = max(0, int(y1) - dilate_px)
                x2 = min(w, int(x2) + dilate_px)
                y2 = min(h, int(y2) + dilate_px)
                mask[y1:y2, x1:x2] = 0  # ignore this region
                result.dynamic_detections += 1
                masked_here += 1

        # COLMAP wants "<image name>.png" (e.g. frame_000001.jpg.png).
        out = mask_dir / (fp.name + ".png")
        cv2.imwrite(str(out), mask)
        fractions.append(1.0 - float((mask > 0).sum()) / (h * w))
        if masked_here:
            result.images_masked += 1

    result.mean_masked_fraction = sum(fractions) / len(fractions) if fractions else 0.0
    result.notes = (
        f"Masked {result.dynamic_detections} dynamic-object detections across "
        f"{result.images_masked}/{result.total_images} frames; these regions "
        f"are excluded from SfM feature extraction."
    )
    logger.info(result.notes)
    return result
