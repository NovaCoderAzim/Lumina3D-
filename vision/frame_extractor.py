"""
frame_extractor.py

Real video -> selected frames, using OpenCV. This is the genuine Person 2
deliverable: it decodes an actual video, scores every candidate frame for
sharpness, and keeps the best, most evenly-spaced frames for reconstruction.

Quality metric: variance of the Laplacian (a standard, well-understood
focus/sharpness measure). Blurry frames have low Laplacian variance; sharp,
detailed frames have high variance. This is a real signal, not a mock.

Selection strategy:
  1. Walk the video, sampling candidate frames at a stride so we don't
     score all N frames of a long clip.
  2. Score each candidate's sharpness.
  3. Divide the timeline into `target_frames` buckets and keep the sharpest
     frame in each bucket -> even coverage + best focus. Overlapping,
     evenly-spaced, in-focus frames are exactly what SfM/COLMAP wants.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Tuple

from .schemas import FrameSelectionResult

logger = logging.getLogger("vision.frame_extractor")


def _laplacian_sharpness(gray) -> float:
    """Variance of the Laplacian — higher means sharper/more in focus."""
    import cv2

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_frames(
    video_path: str,
    project_id: str,
    output_dir: str,
    target_frames: int = 120,
    candidate_stride: int = 3,
    min_sharpness: float = 20.0,
) -> FrameSelectionResult:
    """Extract quality-selected frames from a real video.

    Args:
        video_path: path to the input video file.
        project_id: owning project id (for logging).
        output_dir: where selected frame JPGs are written.
        target_frames: approximate number of frames to keep.
        candidate_stride: score every Nth frame (speed vs. thoroughness).
        min_sharpness: drop candidates below this Laplacian variance.
    """
    import cv2

    start = time.perf_counter()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return FrameSelectionResult(
            status="FAILED",
            project_id=project_id,
            frames_directory=str(out),
            total_frames=0,
            selected_frames=0,
            average_quality=0.0,
            processing_time_seconds=round(time.perf_counter() - start, 2),
            error=f"Could not open video: {video_path}",
        )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    # ---- Pass 1: score candidate frames ----
    # (frame_index, sharpness, encoded_jpg_bytes-less: we re-read on write to
    #  avoid holding every frame in RAM for a long video, so store the image.)
    candidates: List[Tuple[int, float]] = []
    images = {}  # frame_index -> BGR image (only for kept candidates)

    idx = 0
    read_ok = True
    while read_ok:
        read_ok, frame = cap.read()
        if not read_ok:
            break
        if idx % candidate_stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharp = _laplacian_sharpness(gray)
            if sharp >= min_sharpness:
                candidates.append((idx, sharp))
                images[idx] = frame
        idx += 1
    cap.release()

    if total_frames == 0:
        total_frames = idx  # some containers don't report a frame count

    if not candidates:
        return FrameSelectionResult(
            status="FAILED",
            project_id=project_id,
            frames_directory=str(out),
            total_frames=total_frames,
            selected_frames=0,
            average_quality=0.0,
            processing_time_seconds=round(time.perf_counter() - start, 2),
            error="No sufficiently sharp frames found. The video may be too "
            "blurry, too short, or corrupt.",
        )

    # ---- Pass 2: bucket by time, keep the sharpest per bucket ----
    n_buckets = min(target_frames, len(candidates))
    bucket_size = max(1, len(candidates) // n_buckets)

    selected: List[Tuple[int, float]] = []
    for b in range(n_buckets):
        chunk = candidates[b * bucket_size : (b + 1) * bucket_size]
        if not chunk:
            continue
        best = max(chunk, key=lambda c: c[1])
        selected.append(best)
    # Include any tail candidates the bucketing missed by picking the sharpest.
    tail = candidates[n_buckets * bucket_size :]
    if tail:
        selected.append(max(tail, key=lambda c: c[1]))

    # ---- Write selected frames ----
    selected.sort(key=lambda c: c[0])  # chronological order for SfM
    frame_files: List[str] = []
    sharpness_scores: List[float] = []
    for out_idx, (frame_index, sharp) in enumerate(selected):
        img = images.get(frame_index)
        if img is None:
            continue
        fname = out / f"frame_{out_idx:06d}.jpg"
        cv2.imwrite(str(fname), img)
        frame_files.append(str(fname))
        sharpness_scores.append(sharp)

    # Normalise sharpness into a friendly 0..100 "quality" score for the UI.
    # (Laplacian variance is unbounded; we clamp/scale for display only.)
    if sharpness_scores:
        max_s = max(sharpness_scores) or 1.0
        avg_quality = sum(min(100.0, 100.0 * s / max_s) for s in sharpness_scores) / len(
            sharpness_scores
        )
    else:
        avg_quality = 0.0

    elapsed = round(time.perf_counter() - start, 2)
    logger.info(
        "vision: project=%s total=%d selected=%d avg_quality=%.1f time=%.2fs",
        project_id,
        total_frames,
        len(frame_files),
        avg_quality,
        elapsed,
    )

    return FrameSelectionResult(
        status="COMPLETED" if frame_files else "FAILED",
        project_id=project_id,
        frames_directory=str(out),
        total_frames=total_frames,
        selected_frames=len(frame_files),
        average_quality=avg_quality,
        processing_time_seconds=elapsed,
        frame_files=frame_files,
        error="" if frame_files else "No frames could be written.",
    )
