"""
AI / Semantic module adapter.

Wraps the `ai` package. That package uses flat imports internally,
so it expects its own directory on sys.path. We add it here and import the
public entry point:

    ai.interface.AIProcessor.process(
        frames_directory, project_id, reconstruction_metadata
    ) -> AIResult   (a dataclass with .to_dict())

Category normalisation: Maps lowercase categories (structure/vehicle/human/environment)
to standard SemanticCategory (Vehicle/Building/Person/Vegetation).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import get_settings
from ..models import ProcessingError, SemanticCategory, SemanticObject, Vec3
from ..utils import get_logger, log_module_event

logger = get_logger("backend.services.ai")

_AI_DIR = get_settings().repo_root / "ai"

# Category normalisation -> frontend SemanticCategory
_CATEGORY_MAP = {
    "structure": SemanticCategory.BUILDING,
    "building": SemanticCategory.BUILDING,
    "vehicle": SemanticCategory.VEHICLE,
    "human": SemanticCategory.PERSON,
    "person": SemanticCategory.PERSON,
    "environment": SemanticCategory.VEGETATION,
    "vegetation": SemanticCategory.VEGETATION,
}


class AIFailure(Exception):
    def __init__(self, error: ProcessingError):
        self.error = error
        super().__init__(error.detail)


def _ensure_importable() -> None:
    ai_path = str(_AI_DIR)
    if ai_path not in sys.path:
        sys.path.insert(0, ai_path)


def _to_semantic_object(raw: Dict[str, Any]) -> SemanticObject:
    category = _CATEGORY_MAP.get(
        str(raw.get("category", "")).lower(), SemanticCategory.VEHICLE
    )
    pos = raw.get("estimated_3d_position")
    vec: Optional[Vec3] = None
    if isinstance(pos, (list, tuple)) and len(pos) >= 3:
        vec = Vec3(x=float(pos[0]), y=float(pos[1]), z=float(pos[2]))
    return SemanticObject(
        id=str(raw.get("id")),
        **{"class": str(raw.get("class", "object"))},
        category=category,
        confidence=float(raw.get("confidence", 0.0)),
        observations=int(raw.get("observations", 0)),
        estimated_3d_position=vec,
    )


def process(
    frames_directory: str,
    project_id: str,
    reconstruction_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    settings = get_settings()
    start = time.perf_counter()
    _ensure_importable()

    try:
        from interface import AIProcessor  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise AIFailure(
            ProcessingError(
                title="AI module unavailable",
                detail=str(exc),
                cause="import_error",
                suggestion="Install ai/requirements.txt or set USE_MOCK_AI=true.",
            )
        ) from exc

    processor = AIProcessor(use_mock=settings.use_mock_ai)
    try:
        result = processor.process(
            frames_directory=frames_directory,
            project_id=project_id,
            reconstruction_metadata=reconstruction_metadata,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed = round(time.perf_counter() - start, 2)
        log_module_event(
            logger,
            project_id=project_id,
            module="ai",
            status="FAILED",
            processing_time=elapsed,
            error=str(exc),
        )
        raise AIFailure(
            ProcessingError(
                title="Semantic analysis failed",
                detail=str(exc),
                cause="ai_exception",
                suggestion="Run in mock mode (USE_MOCK_AI=true) for the demo.",
            )
        ) from exc

    data = result.to_dict() if hasattr(result, "to_dict") else dict(result)

    objects: List[SemanticObject] = [_to_semantic_object(o) for o in data.get("objects", [])]

    elapsed = round(time.perf_counter() - start, 2)
    log_module_event(
        logger,
        project_id=project_id,
        module="ai" + ("(mock)" if settings.use_mock_ai else ""),
        status="COMPLETED",
        processing_time=elapsed,
    )
    return {
        "objects": objects,
        "summary": data.get("summary", {}),
        "statistics": data.get("statistics", {}),
        "is_mock": settings.use_mock_ai,
        "adapter_time_seconds": elapsed,
    }
