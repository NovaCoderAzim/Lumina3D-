"""
tracker.py

Lightweight multi-frame association / deduplication so that the same
physical object seen in many frames collapses into one tracked object
with N observations, instead of N separate detections.

Default strategy: greedy IoU + centroid-distance matching against each
track's most recent bounding box. No external dependency ΓÇö works
everywhere, including CPU-only environments, and needs no training.

If ByteTrack/BoT-SORT is available through the installed Ultralytics
version, interface.py may prefer `model.track(...)` instead and skip this
module entirely ΓÇö this class exists as the reliable fallback so tracking
never blocks the demo.
"""

import itertools
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from schemas import Detection

logger = logging.getLogger("ai.tracker")


def _centroid(bbox: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    class_name: str
    last_bbox: List[float]
    confidences: List[float] = field(default_factory=list)
    frames_seen: List[int] = field(default_factory=list)
    bbox_history: List[List[float]] = field(default_factory=list)


class SimpleTracker:
    """Greedy IoU + centroid-distance tracker.

    Matching rule: same class_name AND (IoU above threshold OR centroid
    distance below threshold) is treated as the same physical object.
    This is intentionally simple ΓÇö good enough for a single-pass drone
    video where the same object reappears across nearby frames.
    """

    def __init__(self, iou_threshold: float = 0.3, centroid_max_dist: float = 80.0):
        self.iou_threshold = iou_threshold
        self.centroid_max_dist = centroid_max_dist
        self._tracks: Dict[int, Track] = {}
        self._next_id = itertools.count(1)

    def _match(self, det: Detection) -> Optional[Track]:
        best_track, best_score = None, 0.0
        for track in self._tracks.values():
            if track.class_name != det.class_name:
                continue
            iou = _iou(track.last_bbox, det.bbox)
            cx1, cy1 = _centroid(track.last_bbox)
            cx2, cy2 = _centroid(det.bbox)
            dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
            matches = iou >= self.iou_threshold or dist <= self.centroid_max_dist
            if matches and iou >= best_score:
                best_track, best_score = track, iou
        return best_track

    def update(self, detections: List[Detection]) -> None:
        """Feed one frame's detections into the tracker, in frame order."""
        for det in detections:
            track = self._match(det)
            if track is None:
                track_id = next(self._next_id)
                track = Track(
                    track_id=track_id,
                    class_name=det.class_name,
                    last_bbox=det.bbox,
                )
                self._tracks[track_id] = track
            track.last_bbox = det.bbox
            track.confidences.append(det.confidence)
            track.frames_seen.append(det.frame_id)
            track.bbox_history.append(det.bbox)

    def finalize(self) -> List[Track]:
        """Call once all frames have been fed in. Returns every track,
        including single-observation ones (a real object seen only once
        is still a valid object ΓÇö it just has observations=1)."""
        return list(self._tracks.values())
