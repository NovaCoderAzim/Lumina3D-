"""Lumina3D vision module: video -> quality-selected frames."""

from .interface import VisionProcessor
from .schemas import FrameSelectionResult

__all__ = ["VisionProcessor", "FrameSelectionResult"]
