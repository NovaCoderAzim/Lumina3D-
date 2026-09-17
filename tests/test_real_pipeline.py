"""
REAL pipeline integration test (no mocks for vision + AI).

Generates a real .mp4 with OpenCV, runs the REAL vision frame extractor and
the REAL YOLO detector, and asserts genuine outputs. Skips automatically if
the heavy deps (opencv/torch/ultralytics) aren't installed, so the mock CI
run is unaffected.

This proves the real vision -> real AI chain works on this machine.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("torch")
pytest.importorskip("ultralytics")

import numpy as np  # noqa: E402


def _make_real_video(path: Path, n: int = 60, w: int = 640, h: int = 480) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, 20, (w, h))
    for i in range(n):
        t = i / n
        img = np.full((h, w, 3), 30, np.uint8)
        for y in range(0, h, 16):
            cv2.line(img, (0, y), (w, y), (60, 60, 60), 1)
        cx = int(80 + t * (w - 160))
        cv2.rectangle(img, (cx - 40, 180), (cx + 40, 300), (0, 140, 255), -1)
        cv2.circle(img, (int(w * 0.7), 120), 35, (0, 200, 80), -1)
        cv2.putText(img, f"F{i}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        vw.write(img)
    vw.release()


def test_real_vision_then_real_ai(tmp_path):
    # ---- REAL vision: decode + sharpness selection ----
    from vision import VisionProcessor

    video = tmp_path / "clip.mp4"
    _make_real_video(video)
    assert video.exists() and video.stat().st_size > 1000

    frames_dir = tmp_path / "frames"
    vres = VisionProcessor(target_frames=20, candidate_stride=2).process(
        str(video), "realtest", frames_dir
    )
    assert vres.status == "COMPLETED", vres.error
    assert vres.selected_frames > 0
    on_disk = list(frames_dir.glob("*.jpg"))
    assert len(on_disk) == vres.selected_frames  # frames really written
    assert vres.average_quality > 0

    # ---- REAL AI: YOLO on the extracted frames ----
    import sys

    ai_dir = str(Path(__file__).resolve().parents[1] / "ai")
    if ai_dir not in sys.path:
        sys.path.insert(0, ai_dir)
    from interface import AIProcessor  # type: ignore

    use_gpu = os.environ.get("USE_GPU", "true").lower() == "true"
    result = AIProcessor(use_mock=False, use_gpu=use_gpu).process(
        frames_directory=str(frames_dir), project_id="realtest", reconstruction_metadata=None
    )
    d = result.to_dict()
    assert d["status"] == "COMPLETED"
    # Real inference ran over every selected frame.
    assert d["statistics"]["frames_processed"] == vres.selected_frames
    # Detections is an int >= 0; the model really executed.
    assert isinstance(d["statistics"]["detections"], int)
