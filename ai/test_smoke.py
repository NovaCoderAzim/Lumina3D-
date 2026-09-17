"""
test_smoke.py

Fast sanity check that doesn't require torch/ultralytics to be installed.
Run with: python test_smoke.py

Covers:
- mock mode end-to-end (interface -> mock -> output files)
- empty-frames-directory failure handling (no crash, valid COMPLETED result)
- semantic classification
"""

import json
import shutil
import tempfile
from pathlib import Path

from interface import AIProcessor
from semantic import classify


def test_mock_mode():
    processor = AIProcessor(use_mock=True)
    result = processor.process(
        frames_directory="ignored_in_mock_mode",
        project_id="demo_001",
    )
    assert result.status == "COMPLETED"
    assert len(result.objects) > 0
    assert result.summary.get("buildings") == 4
    assert result.summary.get("vehicles") == 12
    assert result.summary.get("people") == 7
    assert result.summary.get("vegetation") == 16

    out_dir = Path("data/projects/demo_001/ai")
    for name in ("detections.json", "objects.json", "semantic.json"):
        path = out_dir / name
        assert path.exists(), f"missing output: {path}"
        json.loads(path.read_text())  # must be valid JSON

    print("[PASS] mock mode produces valid, schema-correct outputs")


def test_empty_frames_dir():
    with tempfile.TemporaryDirectory() as tmp:
        processor = AIProcessor(use_mock=False)
        result = processor.process(frames_directory=tmp, project_id="empty_proj")
        assert result.status == "COMPLETED"
        assert result.objects == []
        assert result.summary == {}
        print("[PASS] empty frames directory returns a valid empty COMPLETED result")


def test_missing_frames_dir_does_not_crash():
    processor = AIProcessor(use_mock=False)
    result = processor.process(
        frames_directory="/nonexistent/path/whatever", project_id="missing_proj"
    )
    assert result.status == "FAILED"
    assert result.error is not None
    print("[PASS] missing frames directory fails gracefully with status=FAILED")


def test_semantic_classify():
    assert classify("car") == "vehicle"
    assert classify("person") == "human"
    assert classify("tree") == "environment"
    assert classify("building") == "structure"
    assert classify("spaceship") == "other"  # unmapped class never crashes
    print("[PASS] semantic classification covers known and unknown classes")


if __name__ == "__main__":
    test_mock_mode()
    test_empty_frames_dir()
    test_missing_frames_dir_does_not_crash()
    test_semantic_classify()
    for temp_dir in ("data/projects/empty_proj", "data/projects/missing_proj"):
        shutil.rmtree(temp_dir, ignore_errors=True)
    print("\nAll smoke tests passed.")
