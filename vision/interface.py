"""
interface.py

Public entry point for the Lumina3D vision module. Everything
about OpenCV, sharpness scoring, and frame selection is hidden behind
VisionProcessor.process().

Contract:
    vision.process(video_path) -> FrameSelectionResult

Configuration (constructor args or env vars):
    VISION_TARGET_FRAMES     default 120
    VISION_CANDIDATE_STRIDE  default 3
    VISION_MIN_SHARPNESS     default 20.0
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from .frame_extractor import extract_frames
from .schemas import FrameSelectionResult


class VisionProcessor:
    def __init__(
        self,
        target_frames: Optional[int] = None,
        candidate_stride: Optional[int] = None,
        min_sharpness: Optional[float] = None,
    ):
        self.target_frames = (
            target_frames
            if target_frames is not None
            else int(os.environ.get("VISION_TARGET_FRAMES", "120"))
        )
        self.candidate_stride = (
            candidate_stride
            if candidate_stride is not None
            else int(os.environ.get("VISION_CANDIDATE_STRIDE", "3"))
        )
        self.min_sharpness = (
            min_sharpness
            if min_sharpness is not None
            else float(os.environ.get("VISION_MIN_SHARPNESS", "20.0"))
        )

    def process(
        self,
        video_path: str,
        project_id: str,
        output_dir: str | Path,
    ) -> FrameSelectionResult:
        return extract_frames(
            video_path=video_path,
            project_id=project_id,
            output_dir=str(output_dir),
            target_frames=self.target_frames,
            candidate_stride=self.candidate_stride,
            min_sharpness=self.min_sharpness,
        )
