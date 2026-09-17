"""
mock.py

Mock mode: returns a schema-identical AIResult without running any model.
Enables offline development and testing of the visualization interface before
real detection is running, and lets the pipeline be smoke-tested with
USE_MOCK_AI=true and no GPU.

The mock output MUST use exactly the same schema as the real output —
see schemas.py. Treat demo/objects.json as part of the contract.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from schemas import AIResult, AIStatistics, TrackedObject

logger = logging.getLogger("ai.mock")

DEFAULT_MOCK_PATH = Path(__file__).parent / "demo" / "objects.json"


def load_mock_result(project_id: str, mock_path: Optional[str] = None) -> AIResult:
    path = Path(mock_path) if mock_path else DEFAULT_MOCK_PATH

    if not path.exists():
        logger.warning("Mock file not found at %s ΓÇö returning empty result", path)
        return AIResult(
            status="COMPLETED",
            project_id=project_id,
            objects=[],
            summary={},
            statistics=AIStatistics(),
        )

    with open(path, "r") as f:
        raw = json.load(f)

    objects = [
        TrackedObject(
            id=o["id"],
            class_name=o["class"],
            category=o["category"],
            confidence=o["confidence"],
            observations=o["observations"],
            frames_seen=o.get("frames_seen", []),
            bbox_history=o.get("bbox_history"),
            estimated_3d_position=o.get("estimated_3d_position"),
            semantic_region=o.get("semantic_region"),
            association_confidence=o.get("association_confidence"),
        )
        for o in raw.get("objects", [])
    ]

    stats_raw = raw.get("statistics", {})
    statistics = AIStatistics(
        frames_processed=stats_raw.get("frames_processed", 0),
        detections=stats_raw.get("detections", 0),
        unique_objects=stats_raw.get("unique_objects", len(objects)),
        average_detection_confidence=stats_raw.get("average_detection_confidence", 0.0),
        processing_time_seconds=stats_raw.get("processing_time_seconds", 0.0),
    )

    return AIResult(
        status="COMPLETED",
        project_id=project_id,
        objects=objects,
        summary=raw.get("summary", {}),
        statistics=statistics,
    )
