"""
schemas.py

Data schemas for the AI & Semantic Intelligence module (Person 4,
DRISHTI-3D). These are the exact JSON-serializable contracts shared with:

- Person 3 (reconstruction / camera poses)  -> consumed, not defined here
- Person 5 (3D visualization)               -> consumes objects.json / semantic.json
- Person 6 (analytics)                      -> consumes AIStatistics

Keep this file dependency-free (stdlib only) so every other module in this
package, and any external caller, can import it without pulling in
PyTorch/YOLO/OpenCV.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Detection:
    """A single raw YOLO detection in one frame."""

    frame_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] in pixel coordinates

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(float(v), 2) for v in self.bbox],
        }


@dataclass
class FrameDetections:
    """All detections found in one frame ΓÇö matches detections.json rows."""

    frame_id: int
    detections: List[Detection] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "detections": [d.to_dict() for d in self.detections],
        }


@dataclass
class TrackedObject:
    """A deduplicated object tracked across multiple frames.

    Optional fields default to None and MUST stay None whenever the
    underlying information isn't reliably known ΓÇö never fabricate a
    3D position, region, or association confidence.
    """

    id: str
    class_name: str
    category: str
    confidence: float  # representative detection confidence (mean across observations)
    observations: int
    frames_seen: List[int] = field(default_factory=list)

    bbox_history: Optional[List[List[float]]] = None
    estimated_3d_position: Optional[List[float]] = None
    semantic_region: Optional[str] = None
    association_confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "class": self.class_name,
            "category": self.category,
            "confidence": round(float(self.confidence), 4),
            "observations": self.observations,
            "frames_seen": self.frames_seen,
            "bbox_history": self.bbox_history,
            "estimated_3d_position": self.estimated_3d_position,
            "semantic_region": self.semantic_region,
            "association_confidence": (
                round(float(self.association_confidence), 4)
                if self.association_confidence is not None
                else None
            ),
        }


@dataclass
class AIStatistics:
    """Processing statistics ΓÇö consumed by Person 6's analytics dashboard."""

    frames_processed: int = 0
    detections: int = 0
    unique_objects: int = 0
    average_detection_confidence: float = 0.0
    processing_time_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frames_processed": self.frames_processed,
            "detections": self.detections,
            "unique_objects": self.unique_objects,
            "average_detection_confidence": round(
                float(self.average_detection_confidence), 4
            ),
            "processing_time_seconds": round(float(self.processing_time_seconds), 2),
        }


@dataclass
class AIResult:
    """Top-level return value of AIProcessor.process()."""

    status: str  # "COMPLETED" | "FAILED"
    project_id: str
    objects: List[TrackedObject] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    statistics: Optional[AIStatistics] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "project_id": self.project_id,
            "objects": [o.to_dict() for o in self.objects],
            "summary": self.summary,
            "statistics": self.statistics.to_dict() if self.statistics else None,
            "error": self.error,
        }
