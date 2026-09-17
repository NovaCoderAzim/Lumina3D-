"""
Data schemas for the Lumina3D vision module.

Kept dependency-free (stdlib only) so any caller can import the contract
without pulling in OpenCV/NumPy. This is the shape the backend and
reconstruction engine rely on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class FrameSelectionResult:
    """Return value of vision.process(). `frames_directory` contains the
    selected, quality-filtered frames written to disk as JPGs."""

    status: str  # "COMPLETED" | "FAILED"
    project_id: str
    frames_directory: str
    total_frames: int
    selected_frames: int
    average_quality: float  # mean sharpness score of selected frames (0..100)
    processing_time_seconds: float
    frame_files: List[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "project_id": self.project_id,
            "frames_directory": self.frames_directory,
            "total_frames": self.total_frames,
            "selected_frames": self.selected_frames,
            "average_quality": round(float(self.average_quality), 2),
            "processing_time_seconds": round(float(self.processing_time_seconds), 2),
            "error": self.error,
        }
