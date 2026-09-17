"""
Vision module adapter (Person 2's territory).

Contract (guide section 6):
    vision.process(video_path) -> FrameSelectionResult

Person 2 has not delivered a real module into this tree yet, so only the
MOCK implementation exists here and it is clearly labelled as such
(guide section 17: never present a mock as real). When Person 2 ships a
`vision/` package with an interface, wire it into `_process_real` and
flip USE_MOCK_VISION=false.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from ..config import get_settings
from ..utils import get_logger, log_module_event

logger = get_logger("backend.services.vision")

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class FrameSelectionResult:
    frames_directory: str
    total_frames: int
    selected_frames: int
    average_quality: float
    processing_time_seconds: float
    is_mock: bool = False
    frame_files: List[str] = field(default_factory=list)


def _count_images(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for p in directory.iterdir() if p.suffix.lower() in _IMAGE_EXTS)


def _run_mock(video_path: str, project_id: str) -> FrameSelectionResult:
    """MOCK: does not decode the video. It ensures a frames directory
    exists and reports plausible, clearly-mock statistics so the rest of
    the pipeline (reconstruction + AI) has a directory to point at."""
    start = time.perf_counter()
    settings = get_settings()
    frames_dir = settings.project_frames_dir(project_id) / "selected"
    frames_dir.mkdir(parents=True, exist_ok=True)

    existing = _count_images(frames_dir)
    if not existing:
        # Write a handful of tiny placeholder frame files so downstream
        # modules (mock reconstruction/AI) that count images in this
        # directory never see an empty dir (which would divide-by-zero).
        # These are clearly-labelled placeholders, not real frames.
        for i in range(12):
            (frames_dir / f"frame_{i:06d}.jpg").write_bytes(b"MOCK_FRAME_PLACEHOLDER")
        existing = _count_images(frames_dir)
    # Report the demo-plausible count (176) for UI display, but the real
    # on-disk count is what reconstruction/AI will actually see.
    selected = 176

    elapsed = round(time.perf_counter() - start, 3)
    log_module_event(
        logger,
        project_id=project_id,
        module="vision(mock)",
        status="COMPLETED",
        processing_time=elapsed,
    )
    return FrameSelectionResult(
        frames_directory=str(frames_dir),
        total_frames=5832,
        selected_frames=selected,
        average_quality=91.7,
        processing_time_seconds=elapsed,
        is_mock=True,
    )


def process(video_path: str, project_id: str) -> FrameSelectionResult:
    settings = get_settings()
    if settings.use_mock_vision:
        return _run_mock(video_path, project_id)
    return _process_real(video_path, project_id)


def _process_real(video_path: str, project_id: str) -> FrameSelectionResult:
    """Real frame extraction via Person 2's vision module (vision/).

    Decodes the actual video, scores each frame for sharpness (variance of
    the Laplacian), and keeps the sharpest, evenly-spaced frames. Raises a
    clear error the orchestrator turns into a user-facing message if the
    module or OpenCV is unavailable, or the video can't be processed."""
    settings = get_settings()
    frames_dir = settings.project_frames_dir(project_id) / "selected"

    try:
        from vision import VisionProcessor
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Real vision mode requires the vision/ module and OpenCV. "
            "Install vision/requirements.txt or set USE_MOCK_VISION=true."
        ) from exc

    processor = VisionProcessor()
    result = processor.process(video_path, project_id, frames_dir)

    if result.status == "FAILED":
        raise RuntimeError(result.error or "Frame extraction failed.")

    log_module_event(
        logger,
        project_id=project_id,
        module="vision",
        status="COMPLETED",
        processing_time=result.processing_time_seconds,
    )
    return FrameSelectionResult(
        frames_directory=result.frames_directory,
        total_frames=result.total_frames,
        selected_frames=result.selected_frames,
        average_quality=result.average_quality,
        processing_time_seconds=result.processing_time_seconds,
        is_mock=False,
        frame_files=result.frame_files,
    )
