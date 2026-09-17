"""
Reconstruction unit tests:
Dataset A (good), Dataset B (poor overlap), Dataset C (too few images).

These tests use synthetic images, which is enough to exercise
every code path and error contract WITHOUT needing a GPU or a real
drone dataset. What they guarantee: the schema contract, the fail-fast behavior,
and that mock mode and the real pipeline never crash the caller.

Run with: pytest reconstruction/test_reconstruction.py -v
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from PIL import Image

from .interface import ReconstructionEngine
from .schemas import ReconstructionConfig, ReconstructionResult, ReconstructionStatus


@pytest.fixture
def tmp_frames(tmp_path):
    def _make(n: int, size=(400, 300)) -> Path:
        frames_dir = tmp_path / f"frames_{n}"
        frames_dir.mkdir()
        for i in range(n):
            Image.new("RGB", size, color=(i * 20 % 255, 100, 150)).save(
                frames_dir / f"frame_{i:06d}.jpg"
            )
        return frames_dir

    return _make


# --------------------------------------------------------------------------
# Dataset C - too few images: must fail BEFORE COLMAP is even invoked
# --------------------------------------------------------------------------

def test_dataset_c_too_few_images(tmp_frames):
    frames_dir = tmp_frames(2)
    engine = ReconstructionEngine(ReconstructionConfig(use_mock_reconstruction=False))
    result = engine.process(str(frames_dir), "dataset_c")

    assert result.status == ReconstructionStatus.FAILED
    assert result.error.error_code == "TOO_FEW_IMAGES"
    assert result.registered_images == 0


# --------------------------------------------------------------------------
# Dataset B - poor overlap / no real texture: SfM registers nothing,
# early-stop / SFM_FAILED must fire, and it must not attempt dense/mesh.
# --------------------------------------------------------------------------

def test_dataset_b_poor_registration(tmp_frames, monkeypatch):
    frames_dir = tmp_frames(6)
    engine = ReconstructionEngine(ReconstructionConfig(use_mock_reconstruction=False))

    from reconstruction import colmap_runner

    dense_called = {"value": False}
    original = colmap_runner.run_dense_reconstruction

    def _spy(*args, **kwargs):
        dense_called["value"] = True
        return original(*args, **kwargs)

    monkeypatch.setattr(colmap_runner, "run_dense_reconstruction", _spy)

    result = engine.process(str(frames_dir), "dataset_b")

    assert result.status == ReconstructionStatus.FAILED
    assert result.error.error_code in ("SFM_FAILED", "INSUFFICIENT_REGISTRATION")
    assert dense_called["value"] is False, (
        "dense reconstruction must never run after a failed/low registration"
    )


# --------------------------------------------------------------------------
# Dataset A stand-in - schema contract via mock mode.
# --------------------------------------------------------------------------

def test_dataset_a_schema_contract_via_mock(tmp_frames):
    frames_dir = tmp_frames(20)
    engine = ReconstructionEngine(ReconstructionConfig(use_mock_reconstruction=True))
    result = engine.process(str(frames_dir), "dataset_a")

    assert result.status == ReconstructionStatus.COMPLETED
    assert result.registration_rate > 90
    assert Path(result.model).exists()
    assert Path(result.model).stat().st_size > 0


# --------------------------------------------------------------------------
# Mock/real schema parity
# --------------------------------------------------------------------------

def test_mock_and_real_share_schema(tmp_frames):
    frames_dir = tmp_frames(3)

    mock_engine = ReconstructionEngine(ReconstructionConfig(use_mock_reconstruction=True))
    mock_result = mock_engine.process(str(frames_dir), "schema_check")

    real_engine = ReconstructionEngine(ReconstructionConfig(use_mock_reconstruction=False))
    real_result = real_engine.process(str(frames_dir), "schema_check")

    assert isinstance(mock_result, ReconstructionResult)
    assert isinstance(real_result, ReconstructionResult)
    assert set(mock_result.model_dump().keys()) == set(real_result.model_dump().keys())
