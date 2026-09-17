"""DRISHTI-3D vision module (Person 2): video -> quality-selected frames."""

from .interface import VisionProcessor
from .schemas import FrameSelectionResult

__all__ = ["VisionProcessor", "FrameSelectionResult"]
